mod common;
use common::*;
use wasi_nn_safe::{GraphBuilder, GraphEncoding, GraphExecutionTarget, TensorType};

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

    assert_eq!(
        output_bytes,
        output_buffer.len() * std::mem::size_of::<f32>()
    );
    Ok(())
}

#[test]
fn test_doc_example() {
    test(MODEL_FILE).unwrap();
}

#[test]
fn test_inference() {
    // load and build graph
    let model_binary = std::fs::read(MODEL_FILE).unwrap();
    let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, GraphExecutionTarget::CPU)
        .build_from_bytes([model_binary.clone()].into_iter())
        .unwrap();

    // prepare inputs and outputs buffer
    let input_dimensions = [1, 3, 224, 224];
    let input_data = generate_random_input(
        input_dimensions.iter().fold(1, |mul, val| mul * val),
        0.0,
        255.0,
    );
    let output_len = 1001;
    let mut output_buffer = vec![0f32; output_len];

    // do inference
    let mut graph_exec_ctx = graph.init_execution_context().unwrap();
    graph_exec_ctx
        .set_input(0, TensorType::F32, &input_dimensions, &input_data)
        .unwrap();
    graph_exec_ctx.compute().unwrap();
    let out_bytes = graph_exec_ctx.get_output(0, &mut output_buffer).unwrap();
    assert_eq!(out_bytes, output_buffer.len() * std::mem::size_of::<f32>());

    // for test
    let wasi_nn_output = unsafe {
        get_wasi_nn_output(
            &model_binary,
            &input_data,
            &input_dimensions.to_vec(),
            output_len,
        )
    };

    // check output
    // for convenience, cast to u8 to check eq
    let ans_1 = change_f32_to_u8(&wasi_nn_output);
    let ans_2 = change_f32_to_u8(&output_buffer);
    assert_eq!(ans_1, ans_2);
}
