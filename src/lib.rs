//! wasi-nn-safe
//!
//! ## Introduction
//! This library provides some convenient and safe wrapper APIs for **wasi-nn system calls**, which can
//! replace the unsafe [wasi-nn](https://github.com/bytecodealliance/wasi-nn) APIs.
//!
//! ## Quick Start
//! ```rust
//! use wasi_nn_safe::{GraphBuilder, TensorType};
//!
//! fn test(model_path: &'static str) -> Result<(), wasi_nn_safe::Error> {
//!     // prepare input and output buffer.
//!     let input = vec![0f32; 224 * 224 * 3];
//!     let input_dim = vec![1, 224, 224, 3];
//!     // the input and output buffer can be any sized type, such as u8, f32, etc.
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
//! ## Note
//! This crate is experimental and will change to adapt the upstream
//! [wasi-nn specification](https://github.com/WebAssembly/wasi-nn/).
//!
//! Now version is based on git commit ```0f77c48ec195748990ff67928a4b3eef5f16c2de```
//!

mod error;
mod graph;
mod syscall;
mod tensor;
mod utils;

pub use error::Error;
pub use graph::{Graph, GraphBuilder, GraphEncoding, GraphExecutionContext, GraphExecutionTarget};
pub use tensor::TensorType;
pub use utils::SharedSlice;

/// re-export ```thiserror``` crate
pub use thiserror;
