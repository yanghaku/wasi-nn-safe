#![allow(unused)]

use rand::distributions::Uniform;
use rand::Rng;

pub const MODEL_FILE: &'static str = "./assets/mobilenet_v1_0.25_224_1_default_1.tflite";

pub fn generate_random_input(size: usize, min: f32, max: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let u = Uniform::new(min, max);
    let mut res = Vec::with_capacity(size);
    for _ in 0..size {
        res.push(rng.sample(u));
    }
    res
}

// change f32 to bytes
pub fn change_f32_to_u8(input: &Vec<f32>) -> Vec<u8> {
    let mut v = Vec::with_capacity(input.len() * 4);
    for f in input {
        for b in f.to_ne_bytes() {
            v.push(b);
        }
    }
    v
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
pub unsafe fn get_wasi_nn_output(
    graph_binary: &Vec<u8>,
    input: &Vec<f32>,
    input_dim: &Vec<usize>,
    output_len: usize,
) -> Vec<f32> {
    let input_dim: Vec<u32> = input_dim.iter().map(|d| *d as u32).collect();
    let input = change_f32_to_u8(input);

    let graph = wasi_nn::load(
        &[&graph_binary],
        wasi_nn::GRAPH_ENCODING_TENSORFLOWLITE,
        wasi_nn::EXECUTION_TARGET_CPU,
    )
    .unwrap();

    let context = wasi_nn::init_execution_context(graph).unwrap();
    let tensor = wasi_nn::Tensor {
        dimensions: &input_dim,
        type_: wasi_nn::TENSOR_TYPE_F32,
        data: &input,
    };
    wasi_nn::set_input(context, 0, tensor).unwrap();
    wasi_nn::compute(context).unwrap();

    let mut output_buffer = vec![0f32; output_len];
    let recv_num = wasi_nn::get_output(
        context,
        0,
        &mut output_buffer[..] as *mut [f32] as *mut u8,
        (output_buffer.len() * std::mem::size_of::<f32>())
            .try_into()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(recv_num as usize, output_len * std::mem::size_of::<f32>());

    output_buffer
}
