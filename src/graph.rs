use crate::tensor::Tensor;
use crate::{syscall, Error, TensorType};

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
pub enum ExecutionTarget {
    CPU,
    GPU,
    TPU,
}

/// Graph factory, which can be used in order to configure the properties of a new graph.
/// Methods can be chained on it in order to configure it.
/// * Default Graph Encoding: `Openvino`.
/// * Default Execution Target: `CPU`.
///
/// ### Examples
///
/// #### build a graph with default config ( `CPU` + `Openvino` )
/// ```rust
/// use wasi_nn_safe::GraphBuilder;
///
/// let xml = "./mobilenet.xml";
/// let weight = "./mobilenet.bin";
/// let graph = GraphBuilder::default().build_from_files([xml, weight])?;
/// ```
///
/// #### build a graph with onnx backend and gpu device target
/// ```rust
/// use wasi_nn_safe::{GraphBuilder, GraphEncoding, ExecutionTarget};
///
/// let model_file = "./test.onnx";
/// let graph = GraphBuilder::new(GraphEncoding::Onnx, ExecutionTarget::GPU).build_from_files([model_file])?;
/// ```
///
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    encoding: GraphEncoding,
    target: ExecutionTarget,
}

impl Default for GraphBuilder {
    /// Create default GraphBuild
    /// * Default Graph Encoding: `Openvino`.
    /// * Default Execution Target: `CPU`.
    #[inline(always)]
    fn default() -> Self {
        Self::new(GraphEncoding::Openvino, ExecutionTarget::CPU)
    }
}

impl GraphBuilder {
    /// Create a new [```GraphBuilder```].
    #[inline(always)]
    pub fn new(encoding: GraphEncoding, target: ExecutionTarget) -> Self {
        Self { encoding, target }
    }

    /// Set graph encoding.
    #[inline(always)]
    pub fn encoding(mut self, encoding: GraphEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Set graph execution target.
    #[inline(always)]
    pub fn execution_target(mut self, execution_target: ExecutionTarget) -> Self {
        self.target = execution_target;
        self
    }

    /// Set graph execution target to `CPU`.
    #[inline(always)]
    pub fn cpu(mut self) -> Self {
        self.target = ExecutionTarget::CPU;
        self
    }

    /// Set graph execution target to `GPU`.
    #[inline(always)]
    pub fn gpu(mut self) -> Self {
        self.target = ExecutionTarget::GPU;
        self
    }

    /// Set graph execution target to `TPU`.
    #[inline(always)]
    pub fn tpu(mut self) -> Self {
        self.target = ExecutionTarget::TPU;
        self
    }

    #[inline(always)]
    pub fn build_from_bytes<B>(self, bytes_array: impl AsRef<[B]>) -> Result<Graph, Error>
    where
        B: AsRef<[u8]>,
    {
        let graph_builder_array: Vec<&[u8]> =
            bytes_array.as_ref().iter().map(|s| s.as_ref()).collect();
        let graph_handle =
            syscall::load(graph_builder_array.as_slice(), self.encoding, self.target)?;
        Ok(Graph {
            build_info: self,
            graph_handle,
        })
    }

    #[inline(always)]
    pub fn build_from_files<P>(self, files: impl AsRef<[P]>) -> Result<Graph, Error>
    where
        P: AsRef<std::path::Path>,
    {
        let mut graph_contents = Vec::with_capacity(files.as_ref().len());
        for file in files.as_ref() {
            graph_contents.push(std::fs::read(file).map_err(Into::<Error>::into)?);
        }
        let graph_builder_array: Vec<&[u8]> = graph_contents.iter().map(|s| s.as_ref()).collect();
        let graph_handle =
            syscall::load(graph_builder_array.as_slice(), self.encoding, self.target)?;
        Ok(Graph {
            build_info: self,
            graph_handle,
        })
    }
}

/// An execution graph for performing inference (i.e., a model), which can create instances of [`GraphExecutionContext`].
///
/// ### Example
/// ```rust
/// use wasi_nn_safe::{ExecutionTarget, GraphBuilder, GraphEncoding};
///
/// let model_file = "./test.tflite";
/// // create a graph using `GraphBuilder`
/// let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU).build_from_files([model_file])?;
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
    graph_handle: syscall::GraphHandle,
}

impl Graph {
    /// Get the graph encoding.
    #[inline(always)]
    pub fn encoding(&self) -> GraphEncoding {
        self.build_info.encoding
    }

    /// Get the graph execution target.
    #[inline(always)]
    pub fn execution_target(&self) -> ExecutionTarget {
        self.build_info.target
    }

    /// Use this graph to create a new instances of [`GraphExecutionContext`].
    #[inline(always)]
    pub fn init_execution_context(&self) -> Result<GraphExecutionContext, Error> {
        let ctx_handle = syscall::init_execution_context(&self.graph_handle)?;
        Ok(GraphExecutionContext {
            graph: self,
            ctx_handle,
        })
    }
}

/// Bind a [`Graph`] to the input and output for an inference.
pub struct GraphExecutionContext<'a> {
    graph: &'a Graph,
    ctx_handle: syscall::GraphExecutionContextHandle,
}

impl<'a> GraphExecutionContext<'a> {
    /// Get the [`Graph`] instance for this [`GraphExecutionContext`] instance.
    #[inline(always)]
    pub fn graph(&self) -> &Graph {
        self.graph
    }

    /// Set input uses the `data`, not only [u8], but also [f32], [i32], etc.
    pub fn set_input<T: Sized>(
        &mut self,
        index: usize,
        tensor_type: TensorType,
        dimensions: &[usize],
        data: impl AsRef<[T]>,
    ) -> Result<(), Error> {
        let data_slice = data.as_ref();
        let buf = unsafe {
            core::slice::from_raw_parts(
                data_slice.as_ptr() as *const u8,
                data_slice.len() * std::mem::size_of::<T>(),
            )
        };
        let tensor_for_call = Tensor::new(dimensions, tensor_type, buf);
        syscall::set_input(&mut self.ctx_handle, index, tensor_for_call)
    }

    /// Compute the inference on the given inputs.
    #[inline(always)]
    pub fn compute(&mut self) -> Result<(), Error> {
        syscall::compute(&mut self.ctx_handle)
    }

    /// Copy output tensor to `out_buffer`, return the out **byte size**.
    #[inline(always)]
    pub fn get_output<T: Sized>(
        &self,
        index: usize,
        out_buffer: &mut impl AsMut<[T]>,
    ) -> Result<usize, Error> {
        let output_slice = out_buffer.as_mut();
        let out_buf = unsafe {
            core::slice::from_raw_parts_mut(
                output_slice.as_mut_ptr() as *mut u8,
                output_slice.len() * std::mem::size_of::<T>(),
            )
        };
        syscall::get_output(&self.ctx_handle, index, out_buf)
    }
}

#[cfg(test)]
mod test {
    use super::{ExecutionTarget, GraphBuilder, GraphEncoding};

    #[test]
    fn test_enum_graph_encoding() {
        assert_eq!(GraphEncoding::Openvino as u32, 0);
        assert_eq!(GraphEncoding::Onnx as u32, 1);
        assert_eq!(GraphEncoding::Tensorflow as u32, 2);
        assert_eq!(GraphEncoding::Pytorch as u32, 3);
        assert_eq!(GraphEncoding::TensorflowLite as u32, 4);
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
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
        assert_eq!(ExecutionTarget::CPU as u32, 0);
        assert_eq!(ExecutionTarget::GPU as u32, 1);
        assert_eq!(ExecutionTarget::TPU as u32, 2);
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    fn test_graph_execution_target_with_wasi_nn() {
        assert_eq!(
            ExecutionTarget::CPU as u32,
            wasi_nn::EXECUTION_TARGET_CPU.raw() as u32
        );
        assert_eq!(
            ExecutionTarget::GPU as u32,
            wasi_nn::EXECUTION_TARGET_GPU.raw() as u32
        );
        assert_eq!(
            ExecutionTarget::TPU as u32,
            wasi_nn::EXECUTION_TARGET_TPU.raw() as u32
        );
    }

    #[test]
    fn test_graph_builder() {
        assert_eq!(GraphBuilder::default().encoding, GraphEncoding::Openvino);

        assert_eq!(GraphBuilder::default().target, ExecutionTarget::CPU);
        assert_eq!(GraphBuilder::default().gpu().target, ExecutionTarget::GPU);
        assert_eq!(GraphBuilder::default().tpu().target, ExecutionTarget::TPU);
        assert_eq!(
            GraphBuilder::default().tpu().cpu().target,
            ExecutionTarget::CPU
        );
        assert_eq!(
            GraphBuilder::default()
                .execution_target(ExecutionTarget::GPU)
                .target,
            ExecutionTarget::GPU
        );
    }

    const TEST_TFLITE_MODEL_FILE: &'static str =
        "./assets/mobilenet_v1_0.25_224_1_default_1.tflite";

    #[test]
    fn test_build_graph() {
        let tflite_cpu_builder =
            GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU);

        let graph = tflite_cpu_builder
            .clone()
            .build_from_files([TEST_TFLITE_MODEL_FILE]);
        assert!(graph.is_ok());

        let graph_err = tflite_cpu_builder.clone().build_from_bytes([vec![0]]);
        assert!(graph_err.is_err());

        let bytes = std::fs::read(TEST_TFLITE_MODEL_FILE).unwrap();
        let graph_from_bytes = tflite_cpu_builder.clone().build_from_bytes(&[bytes]);
        assert!(graph_from_bytes.is_ok());
    }

    #[test]
    fn test_graph_compute() {
        let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, ExecutionTarget::CPU)
            .build_from_files([TEST_TFLITE_MODEL_FILE])
            .unwrap();
        let mut exec_ctx = graph.init_execution_context().unwrap();
        exec_ctx.compute().unwrap();
    }
}
