use common::*;
use wasi_nn_safe::{
    GraphBuilder, GraphEncoding, GraphExecutionContext, GraphExecutionTarget, TensorType, ToTensor,
};

mod common;

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

#[should_panic]
#[test]
fn test_to_tensor_impl() {
    // load and build graph
    let model_binary = std::fs::read(MODEL_FILE).unwrap();
    let graph = GraphBuilder::new(GraphEncoding::TensorflowLite, GraphExecutionTarget::CPU)
        .build_from_bytes([model_binary.clone()].into_iter())
        .unwrap();

    // prepare inputs and outputs buffer
    let input_data = MyMedia {};
    let output_len = 1001;

    // do inference
    let mut graph_exec_ctx = graph.init_execution_context().unwrap();
    do_inference(&mut graph_exec_ctx, &input_data, output_len).unwrap();
}
