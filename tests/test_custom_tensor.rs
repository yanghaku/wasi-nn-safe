use common::*;
use wasi_nn_safe::{
    GraphBuilder, GraphEncoding, GraphExecutionContext, GraphExecutionTarget, TensorType, ToTensor,
};

mod common;

struct MyTensor {
    tp: TensorType,
    dim: Vec<usize>,
    data: Vec<f32>,
}

impl ToTensor for MyTensor {
    fn tensor_type(&self) -> TensorType {
        self.tp
    }

    fn dimensions(&self) -> &[usize] {
        &self.dim
    }

    fn buffer_for_read(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        }
    }

    fn buffer_for_write(&mut self) -> &mut [u8] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        }
    }
}

fn do_inference(
    ctx: &mut GraphExecutionContext,
    inputs: &[(usize, &MyTensor)],
    outputs: &mut [(usize, &mut MyTensor)],
) -> Result<(), Box<dyn std::error::Error>> {
    ctx.set_input_tensors(inputs)?;
    ctx.compute()?;
    for (index, tensor) in outputs.iter_mut() {
        ctx.output_to_tensor(*index, *tensor)?;
    }
    Ok(())
}

#[test]
fn test_custom_tensor() {
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
    let input_tensor = MyTensor {
        tp: TensorType::F32,
        dim: input_dimensions.to_vec(),
        data: input_data,
    };
    let mut output_tensor = MyTensor {
        tp: TensorType::F32,
        dim: vec![1, output_len],
        data: vec![0f32; output_len],
    };

    // do inference
    let mut graph_exec_ctx = graph.init_execution_context().unwrap();
    do_inference(
        &mut graph_exec_ctx,
        &[(0, &input_tensor)],
        &mut [(0, &mut output_tensor)],
    )
    .unwrap();

    // for test
    let wasi_nn_output = unsafe {
        get_wasi_nn_output(
            &model_binary,
            &input_tensor.data,
            &input_dimensions.to_vec(),
            output_len,
        )
    };
    // check output
    // for convenience, cast to u8 to check eq
    let ans_1 = change_f32_to_u8(&wasi_nn_output);
    let ans_2 = change_f32_to_u8(&output_tensor.data);
    assert_eq!(ans_1, ans_2);
}
