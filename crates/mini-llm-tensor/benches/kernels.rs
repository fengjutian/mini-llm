use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mini_llm_tensor::Tensor;

fn matrix(size: usize, phase: f32) -> Tensor {
    let values = (0..size * size)
        .map(|index| ((index as f32 + phase) * 0.013).sin())
        .collect();
    Tensor::from_2d(size, size, values).expect("square benchmark tensor")
}

fn matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");
    for size in [32usize, 64, 128, 256] {
        let left = matrix(size, 0.0);
        let right = matrix(size, 1.0);
        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(&left).matmul(black_box(&right)).unwrap())
        });
    }
    group.finish();
}

fn softmax(c: &mut Criterion) {
    let input = Tensor::new(
        (0..128 * 512)
            .map(|index| (index as f32 * 0.01).sin())
            .collect(),
        vec![128, 512],
    )
    .unwrap();
    c.bench_function("softmax_rows/128x512", |b| {
        b.iter(|| black_box(&input).softmax_rows().unwrap())
    });
}

criterion_group!(kernels, matmul, softmax);
criterion_main!(kernels);
