use crate::error::{BackendError, Error};
use crate::tensor::{Tensor, TensorType, ToTensor};
use crate::utils::SharedSlice;

/// Describes the encoding of the graph. This allows the API to be implemented by various backends
/// that encode (i.e., serialize) their graph IR with different formats.
/// Now the available backends are `Openvino`, `Onnx`, `Tensorflow`, `Pytorch`, `TensorflowLite`
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(C)]
pub enum GraphEncoding {
    Openvino,
    Onnx,
    Tensorflow,
    Pytorch,
    TensorflowLite,
}

/// Define where the graph should be executed.
/// Now the available devices are `CPU`, `GPU`, `TPU`
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(C)]
pub enum GraphExecutionTarget {
    CPU,
    GPU,
    TPU,
}

/// Graph factory, which can be used in order to configure the properties of a new graph.
/// Methods can be chained on it in order to configure it.
///
/// ### Examples
///
/// #### build a graph with default config ( `CPU` + `TensorflowLite` )
/// ```rust
/// use wasi_nn_safe::GraphBuilder;
///
/// let model_file = "./test.tflite";
/// let graph = GraphBuilder::default().build_from_files([model_file])?;
/// ```
///
/// #### build a graph with onnx backend and gpu device target
/// ```rust
/// use wasi_nn_safe::{GraphBuilder, GraphEncoding, GraphExecutionTarget};
///
/// let model_file = "./test.onnx";
/// let graph = GraphBuilder::new(GraphEncoding::Onnx, GraphExecutionTarget::GPU).build_from_files([model_file])?;
/// ```
///
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    encoding: GraphEncoding,
    target: GraphExecutionTarget,
}

impl Default for GraphBuilder {
    /// Create default GraphBuild
    /// Default encoding is ```TensorflowLite```
    /// Default execution target is ```CPU```
    #[inline(always)]
    fn default() -> Self {
        Self::new(GraphEncoding::TensorflowLite, GraphExecutionTarget::CPU)
    }
}

impl GraphBuilder {
    #[inline(always)]
    pub fn new(encoding: GraphEncoding, target: GraphExecutionTarget) -> Self {
        Self { encoding, target }
    }

    #[inline(always)]
    pub fn encoding(mut self, encoding: GraphEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    #[inline(always)]
    pub fn execution_target(mut self, execution_target: GraphExecutionTarget) -> Self {
        self.target = execution_target;
        self
    }

    #[inline(always)]
    pub fn cpu(mut self) -> Self {
        self.target = GraphExecutionTarget::CPU;
        self
    }

    #[inline(always)]
    pub fn gpu(mut self) -> Self {
        self.target = GraphExecutionTarget::GPU;
        self
    }

    #[inline(always)]
    pub fn tpu(mut self) -> Self {
        self.target = GraphExecutionTarget::TPU;
        self
    }

    #[cfg(target_arch = "wasm32")]
    #[inline]
    fn wasi_nn_syscall_load(&self, graph_builder_array: &[&[u8]]) -> Result<u32, Error> {
        let mut graph_handle = 0;
        let res = unsafe {
            crate::wasi_nn_sys_call::load(
                graph_builder_array.as_ptr() as usize,
                graph_builder_array.len(),
                self.encoding as u32,
                self.target as u32,
                &mut graph_handle as *mut _ as usize,
            )
        };

        if res == 0 {
            Ok(graph_handle)
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    #[inline(always)]
    pub fn build_from_shared_slices(
        self,
        graph_builder_slices: impl AsRef<[SharedSlice<u8>]>,
    ) -> Result<Graph, Error> {
        let graph_contents = Vec::from(graph_builder_slices.as_ref());
        let graph_builder_array: Vec<&[u8]> = graph_contents.iter().map(|s| s.as_ref()).collect();

        let graph_handle = self.wasi_nn_syscall_load(graph_builder_array.as_slice())?;
        Ok(Graph {
            build_info: self,
            graph_handle,
            graph_contents,
        })
    }

    /// If a memory chunk such as `Vec<u8>` has more than one graph blob or other information,
    /// [`SharedSlice`] could be used to avoid copy.
    #[inline(always)]
    pub fn build_from_bytes(
        self,
        bytes_array: impl Iterator<Item = Vec<u8>>,
    ) -> Result<Graph, Error> {
        let graph_contents: Vec<SharedSlice<u8>> =
            bytes_array.map(|v| SharedSlice::from(v)).collect();
        let graph_builder_array: Vec<&[u8]> = graph_contents.iter().map(|s| s.as_ref()).collect();
        let graph_handle = self.wasi_nn_syscall_load(graph_builder_array.as_slice())?;
        Ok(Graph {
            build_info: self,
            graph_handle,
            graph_contents,
        })
    }

    #[inline(always)]
    pub fn build_from_files<P>(self, files: impl AsRef<[P]>) -> Result<Graph, Error>
    where
        P: AsRef<std::path::Path> + 'static,
    {
        let mut bytes_array = Vec::with_capacity(files.as_ref().len());
        for file in files.as_ref() {
            bytes_array.push(std::fs::read(file).map_err(Into::<Error>::into)?);
        }
        self.build_from_bytes(bytes_array.into_iter())
    }
}

/// An execution graph for performing inference (i.e., a model), which can create instances of [`GraphExecutionContext`].
/// Graph must has ownership of graph content bytes.
///
/// ### Example
/// ```rust
/// use wasi_nn_safe::GraphBuilder;
///
/// let model_file = "./test.tflite";
/// // create a graph using `GraphBuilder`
/// let graph = GraphBuilder::default().build_from_files([model_file])?;
/// // create an execution context using this graph
/// let mut graph_exec_ctx = graph.init_execution_context()?;
/// // set input tensors
/// // ......
///
/// // compute the inference on the given inputs
/// graph_exec_ctx.compute()?;
///
/// // get output tensors and do post-processing
/// // ......
/// ```
pub struct Graph {
    build_info: GraphBuilder,
    graph_contents: Vec<SharedSlice<u8>>,
    graph_handle: u32,
}

impl Graph {
    #[inline(always)]
    pub fn encoding(&self) -> GraphEncoding {
        self.build_info.encoding
    }

    #[inline(always)]
    pub fn execution_target(&self) -> GraphExecutionTarget {
        self.build_info.target
    }

    #[inline(always)]
    pub fn graph_contents(&self) -> &Vec<SharedSlice<u8>> {
        &self.graph_contents
    }

    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn init_execution_context(&self) -> Result<GraphExecutionContext, Error> {
        let mut ctx_handle = 0;
        let res = unsafe {
            crate::wasi_nn_sys_call::init_execution_context(
                self.graph_handle,
                &mut ctx_handle as *mut _ as usize,
            )
        };

        if res == 0 {
            Ok(GraphExecutionContext {
                graph: self,
                ctx_handle,
            })
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }
}

/// Bind a [`Graph`] to the input and output [`ToTensor`]s for an inference.
pub struct GraphExecutionContext<'a> {
    graph: &'a Graph,
    ctx_handle: u32,
}

impl<'a> GraphExecutionContext<'a> {
    #[inline(always)]
    pub fn graph(&self) -> &Graph {
        self.graph
    }

    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn wasi_nn_syscall_set_input(&mut self, index: usize, tensor: Tensor) -> Result<(), Error> {
        let res = unsafe {
            crate::wasi_nn_sys_call::set_input(self.ctx_handle, index, &tensor as *const _ as usize)
        };
        if res == 0 {
            Ok(())
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    /// Set input uses the `data`, not only [u8], but also [f32], [i32], etc.
    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: &[T],
    ) -> Result<(), Error> {
        let buf = unsafe {
            core::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };
        let tensor_for_call = Tensor::new(dimensions, tensor_type, buf);
        self.wasi_nn_syscall_set_input(index, tensor_for_call)
    }

    /// Copy the tensor contents to model buffer.
    #[inline(always)]
    pub fn set_input_tensor(&mut self, index: usize, tensor: &impl ToTensor) -> Result<(), Error> {
        let tensor_for_call = Tensor::from(tensor);
        self.wasi_nn_syscall_set_input(index, tensor_for_call)
    }

    #[inline(always)]
    pub fn set_input_tensors<'t, T>(
        &mut self,
        tensors: impl AsRef<[(usize, &'t T)]>,
    ) -> Result<(), Error>
    where
        T: ToTensor + 't,
    {
        for (index, tensor) in tensors.as_ref() {
            let tensor_for_call = Tensor::from(*tensor);
            self.wasi_nn_syscall_set_input(*index, tensor_for_call)?;
        }
        Ok(())
    }

    /// Compute the inference on the given inputs.
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn compute(&mut self) -> Result<(), Error> {
        let res = unsafe { crate::wasi_nn_sys_call::compute(self.ctx_handle) };
        if res == 0 {
            Ok(())
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn wasi_nn_syscall_get_output(&self, index: usize, out_buf: &mut [u8]) -> Result<usize, Error> {
        let mut out_size = 0;
        let res = unsafe {
            crate::wasi_nn_sys_call::get_output(
                self.ctx_handle,
                index,
                out_buf.as_mut_ptr() as usize,
                out_buf.len(),
                &mut out_size as *mut _ as usize,
            )
        };

        if res == 0 {
            Ok(out_size)
        } else {
            Err(Error::BackendError(BackendError::from(res)))
        }
    }

    /// Copy output tensor to `out_buffer`, return the out **byte size**.
    #[inline(always)]
    pub fn get_output<T: Sized>(&self, index: usize, out_buffer: &mut [T]) -> Result<usize, Error> {
        let out_buf = unsafe {
            core::slice::from_raw_parts_mut(
                out_buffer.as_mut_ptr() as *mut u8,
                out_buffer.len() * std::mem::size_of::<T>(),
            )
        };
        self.wasi_nn_syscall_get_output(index, out_buf)
    }

    /// Extract the outputs after inference and save to [`ToTensor`].
    #[inline(always)]
    pub fn output_to_tensor(&self, index: usize, tensor: &mut impl ToTensor) -> Result<(), Error> {
        let expect_out_size = tensor
            .dimensions()
            .iter()
            .fold(tensor.tensor_type().byte_size(), |mul, val| mul * val);
        let out_buf = tensor.buffer_for_write();

        // check buf size
        if expect_out_size > out_buf.len() {
            return Err(Error::InvalidTensorError {
                expect: expect_out_size,
                actual: out_buf.len(),
            });
        }

        let out_size = self.wasi_nn_syscall_get_output(index, out_buf)?;
        if expect_out_size != out_size {
            Err(Error::OutputLengthError {
                expect: expect_out_size,
                got: out_size,
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {
    use super::{GraphBuilder, GraphEncoding, GraphExecutionTarget};

    #[test]
    fn test_enum_graph_encoding() {
        assert_eq!(GraphEncoding::Openvino as u32, 0);
        assert_eq!(GraphEncoding::Onnx as u32, 1);
        assert_eq!(GraphEncoding::Tensorflow as u32, 2);
        assert_eq!(GraphEncoding::Pytorch as u32, 3);
        assert_eq!(GraphEncoding::TensorflowLite as u32, 4);
    }

    #[test]
    fn test_graph_encoding_with_wasi_nn() {
        assert_eq!(
            GraphEncoding::Onnx as u32,
            wasi_nn::GRAPH_ENCODING_ONNX.raw() as u32
        );
        assert_eq!(
            GraphEncoding::Openvino as u32,
            wasi_nn::GRAPH_ENCODING_OPENVINO.raw() as u32
        );
        assert_eq!(
            GraphEncoding::Pytorch as u32,
            wasi_nn::GRAPH_ENCODING_PYTORCH.raw() as u32
        );
        assert_eq!(
            GraphEncoding::Tensorflow as u32,
            wasi_nn::GRAPH_ENCODING_TENSORFLOW.raw() as u32
        );
        assert_eq!(
            GraphEncoding::TensorflowLite as u32,
            wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE.raw() as u32
        );
    }

    #[test]
    fn test_enum_graph_execution_target() {
        assert_eq!(GraphExecutionTarget::CPU as u32, 0);
        assert_eq!(GraphExecutionTarget::GPU as u32, 1);
        assert_eq!(GraphExecutionTarget::TPU as u32, 2);
    }

    #[test]
    fn test_graph_execution_target_with_wasi_nn() {
        assert_eq!(
            GraphExecutionTarget::CPU as u32,
            wasi_nn::EXECUTION_TARGET_CPU.raw() as u32
        );
        assert_eq!(
            GraphExecutionTarget::GPU as u32,
            wasi_nn::EXECUTION_TARGET_GPU.raw() as u32
        );
        assert_eq!(
            GraphExecutionTarget::TPU as u32,
            wasi_nn::EXECUTION_TARGET_TPU.raw() as u32
        );
    }

    #[test]
    fn test_graph_builder() {
        assert_eq!(
            GraphBuilder::default().encoding,
            GraphEncoding::TensorflowLite
        );

        assert_eq!(GraphBuilder::default().target, GraphExecutionTarget::CPU);
        assert_eq!(
            GraphBuilder::default().gpu().target,
            GraphExecutionTarget::GPU
        );
        assert_eq!(
            GraphBuilder::default().tpu().target,
            GraphExecutionTarget::TPU
        );
        assert_eq!(
            GraphBuilder::default().tpu().cpu().target,
            GraphExecutionTarget::CPU
        );
        assert_eq!(
            GraphBuilder::default()
                .execution_target(GraphExecutionTarget::GPU)
                .target,
            GraphExecutionTarget::GPU
        );
    }

    const TEST_TFLITE_MODEL_FILE: &'static str =
        "./assets/mobilenet_v1_0.25_224_1_default_1.tflite";

    #[test]
    fn test_build_graph() {
        let graph = GraphBuilder::default().build_from_files([TEST_TFLITE_MODEL_FILE]);
        assert!(graph.is_ok());

        let graph_err = GraphBuilder::default().build_from_bytes([vec![0]].into_iter());
        assert!(graph_err.is_err());

        let bytes = std::fs::read(TEST_TFLITE_MODEL_FILE).unwrap();
        let graph_from_bytes = GraphBuilder::default().build_from_bytes([bytes].into_iter());
        assert!(graph_from_bytes.is_ok());
    }

    #[test]
    fn test_graph_compute() {
        let graph = GraphBuilder::default()
            .build_from_files([TEST_TFLITE_MODEL_FILE])
            .unwrap();
        let mut exec_ctx = graph.init_execution_context().unwrap();
        exec_ctx.compute().unwrap();
    }
}
