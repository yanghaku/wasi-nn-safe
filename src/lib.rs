//! wasi-nn-safe
//!
//! ## Introduction
//! This library provides some convenient and safe wrapper APIs for **wasi-nn system calls**, which can
//! replace the unsafe [wasi-nn](https://github.com/bytecodealliance/wasi-nn) APIs.
//!
//! ## Note
//! This crate is experimental and will change to adapt the upstream
//! [wasi-nn specification](https://github.com/WebAssembly/wasi-nn/).
//! Now version is based on git commit ```0f77c48ec195748990ff67928a4b3eef5f16c2de```
//!
//! ## Quick Start
//! ```rust
//! use wasi_nn_safe::{GraphBuilder, TensorType};
//!
//! fn test(model_path: &'static str) -> Result<(), wasi_nn_safe::Error> {
//!     // prepare input and output buffer.
//!     let input = vec![0f32; 224 * 224 * 3];
//!     let input_dim = vec![1, 224, 224, 3];
//!     let mut output_buffer = vec![0f32; 1001];
//!
//!     // build a tflite graph from file. (graph builder default with tflite and cpu).
//!     let graph = GraphBuilder::default().build_from_files([model_path])?;
//!     // init graph execution context for this graph.
//!     let mut ctx = graph.init_execution_context()?;
//!     // set input
//!     ctx.set_input(0, TensorType::F32, &input_dim, &input)?;
//!     // do inference
//!     ctx.compute()?;
//!     // copy output to buffer
//!     let output_bytes = ctx.get_output(0, &mut output_buffer)?;
//!
//!     assert_eq!(output_bytes, output_buffer.len() * std::mem::size_of::<f32>());
//!     Ok(())
//! }
//! ```
//!
//! ## Use custom tensor object as input or output
//! ```rust
//! use wasi_nn_safe::{GraphBuilder, GraphExecutionContext, TensorType, ToTensor};
//!
//! struct MyTensor {
//!     tp: TensorType,
//!     dim: Vec<usize>,
//!     data: Vec<u8>,
//! }
//!
//! // impl trait `ToTensor`, and it can used for inference.
//! impl ToTensor for MyTensor {
//!     fn tensor_type(&self) -> TensorType {
//!         self.tp
//!     }
//!     fn dimensions(&self) -> &[usize] {
//!         &self.dim
//!     }
//!     fn buffer_for_read(&self) -> &[u8] {
//!         &self.data
//!     }
//!     fn buffer_for_write(&mut self) -> &mut [u8] {
//!         &mut self.data
//!     }
//! }
//!
//! // do inference using input tensors and output tensors
//! fn do_inference(
//!     ctx: &mut GraphExecutionContext,
//!     inputs: &[(usize, &MyTensor)],
//!     outputs: &mut [(usize, &mut MyTensor)],
//! ) -> Result<(), wasi_nn_safe::Error> {
//!     ctx.set_input_tensors(inputs)?;
//!     ctx.compute()?;
//!     for (index, tensor) in outputs.iter_mut() {
//!         ctx.output_to_tensor(*index, *tensor)?;
//!     }
//!     Ok(())
//! }
//!
//! // if only one input tensor and one output tensor:
//! fn do_inference_(
//!     ctx: &mut GraphExecutionContext,
//!     input: &MyTensor,
//!     output: &mut MyTensor,
//! ) -> Result<(), wasi_nn_safe::Error> {
//!     ctx.set_input_tensor(0, input)?;
//!     ctx.compute()?;
//!     ctx.output_to_tensor(0, output)
//! }
//! ```
//!

mod error;
mod graph;
mod tensor;
mod utils;

#[cfg(target_arch = "wasm32")]
mod wasi_nn_sys_call {
    #[link(wasm_import_module = "wasi_ephemeral_nn")]
    extern "C" {
        pub fn load(
            graph_builder_array_ptr: usize,
            graph_builder_array_len: usize,
            encoding: u32,
            target: u32,
            graph_result_ptr: usize,
        ) -> u32;
        pub fn init_execution_context(graph: u32, context_result_ptr: usize) -> u32;
        pub fn set_input(context: u32, index: usize, tensor_ptr: usize) -> u32;
        pub fn get_output(
            context: u32,
            index: usize,
            out_buffer_ptr: usize,
            out_buffer_max_size: usize,
            result_buffer_size_ptr: usize,
        ) -> u32;
        pub fn compute(context: u32) -> u32;
    }

    #[cfg(test)]
    mod test {
        #[test]
        fn test_bytes_in_wasm32() {
            assert_eq!(std::mem::size_of::<usize>(), std::mem::size_of::<u32>());
            assert_eq!(std::mem::size_of::<u32>(), std::mem::size_of::<i32>());
            assert_eq!(
                std::mem::size_of::<usize>(),
                std::mem::size_of::<*const u8>()
            );
        }
    }
}

pub use error::Error;
pub use graph::{Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, GraphExecutionTarget};
pub use tensor::{TensorType, ToTensor};
pub use utils::SharedSlice;
