<div align="center">
  <h1><code>wasi-nn-safe</code></h1>
  <p>
    <a href="https://github.com/yanghaku/wasi-nn-safe/actions?query=workflow%3ACI">
      <img src="https://github.com/yanghaku/wasi-nn-safe/workflows/CI/badge.svg" alt="CI status" height="20"/>
    </a>
    <a href="https://crates.io/crates/wasi-nn-safe">
      <img src="https://img.shields.io/crates/v/wasi-nn-safe.svg" alt="crates.io status" height="20"/>
    </a>
    <a href="https://docs.rs/wasi-nn-safe">
      <img src="https://img.shields.io/docsrs/wasi-nn-safe" alt="doc.rs status" height="20"/>
    </a>
  </p>
</div>

## Introduction

This library provides some convenient and safe wrapper APIs for **wasi-nn system calls**, which can replace the
unsafe [wasi-nn] APIs.

```toml
[dependencies]
wasi-nn-safe = "0.0.0"
```

> Note
>
> This crate is experimental and will change to adapt the upstream [wasi-nn specification].
>
> Now version is based on git commit ```0f77c48ec195748990ff67928a4b3eef5f16c2de```

## Examples

### Quick Start

```rust
use wasi_nn_safe::{GraphBuilder, TensorType};

fn test(model_path: &'static str) -> Result<(), wasi_nn_safe::Error> {
    // prepare input and output buffer.
    let input = vec![0f32; 224 * 224 * 3];
    let input_dim = vec![1, 224, 224, 3];
    let mut output_buffer = vec![0f32; 1001];

    // build a tflite graph from file.  (graph builder default with tflite and cpu).
    let graph = GraphBuilder::default().build_from_files([model_path])?;
    // init graph execution context for this graph.
    let mut ctx = graph.init_execution_context()?;
    // set input
    ctx.set_input(0, TensorType::F32, &input_dim, &input)?;
    // do inference
    ctx.compute()?;
    // copy output to buffer
    let output_bytes = ctx.get_output(0, &mut output_buffer)?;

    assert_eq!(output_bytes, output_buffer.len() * std::mem::size_of::<f32>());
    Ok(())
}
```

### Use custom tensor object as input or output

```rust
use wasi_nn_safe::{GraphBuilder, GraphExecutionContext, TensorType, ToTensor};

pub struct MyMedia {
    // some fields
}

impl ToTensor for MyMedia {
    fn tensor_type(&self) -> TensorType {
        TensorType::F32
    }

    fn dimensions(&self) -> &[usize] {
        &[]
    }

    /// Media to tensor data
    fn buffer_for_read(&self) -> &[u8] {
        unimplemented!()
    }
}

fn do_inference(
    ctx: &mut GraphExecutionContext,
    input_media: &MyMedia,
    output_len: usize,
) -> Result<Vec<f32>, wasi_nn_safe::Error> {
    // just use `MyMedia` as input.
    ctx.set_input_tensor(0, input_media)?;
    ctx.compute()?;

    let mut buf = vec![0f32; output_len];
    ctx.get_output(0, &mut buf)?;
    Ok(buf)
}
```

## Related Links

- [wasi-nn specification]
- [wasi-nn]

[wasi-nn]: https://github.com/bytecodealliance/wasi-nn

[wasi-nn specification]: https://github.com/WebAssembly/wasi-nn/

## License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE
