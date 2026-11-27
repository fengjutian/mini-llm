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

fn affine(c: &mut Criterion) {
    let input = Tensor::new(vec![0.25; 128 * 256], vec![128, 256]).unwrap();
    let weight = Tensor::new(vec![0.01; 256 * 256], vec![256, 256]).unwrap();
    let bias = Tensor::new(vec![0.1; 256], vec![256]).unwrap();
    let mut group = c.benchmark_group("affine_f32/128x256");
    group.bench_function("separate", |b| {
        b.iter(|| {
            black_box(&input)
                .matmul(black_box(&weight))
                .unwrap()
                .add(black_box(&bias))
                .unwrap()
        })
    });
    group.bench_function("fused", |b| {
        b.iter(|| {
            black_box(&input)
                .matmul_add_bias(black_box(&weight), black_box(&bias))
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(kernels, matmul, softmax, affine);
criterion_main!(kernels);
