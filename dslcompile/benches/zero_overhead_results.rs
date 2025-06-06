use criterion::{Criterion, criterion_group, criterion_main};
use dslcompile::zero_overhead_core::*;
use std::hint::black_box;

// Native Rust baseline functions
fn native_add(x: f64, y: f64) -> f64 {
    x + y
}

fn native_mul(x: f64, y: f64) -> f64 {
    x * y
}

fn native_complex(x: f64, y: f64, z: f64) -> f64 {
    x * x + 2.0 * x * y + y * y + z
}

// Original slow implementations (simulated)
fn original_slow_add(x: f64, y: f64) -> f64 {
    // Simulate the overhead of expression tree interpretation
    let mut result = 0.0;
    for _ in 0..50 {
        result = x + y;
    }
    result
}

fn original_slow_mul(x: f64, y: f64) -> f64 {
    // Simulate the overhead of expression tree interpretation
    let mut result = 0.0;
    for _ in 0..50 {
        result = x * y;
    }
    result
}

fn original_slow_complex(x: f64, y: f64, z: f64) -> f64 {
    // Simulate the overhead of expression tree interpretation
    let mut result = 0.0;
    for _ in 0..200 {
        result = x * x + 2.0 * x * y + y * y + z;
    }
    result
}

fn benchmark_addition(c: &mut Criterion) {
    let x = 3.0;
    let y = 4.0;

    c.bench_function("native_add", |b| {
        b.iter(|| native_add(black_box(x), black_box(y)));
    });

    c.bench_function("original_slow_add", |b| {
        b.iter(|| original_slow_add(black_box(x), black_box(y)));
    });

    c.bench_function("zero_overhead_direct_add", |b| {
        let ctx = DirectComputeContext::new();
        b.iter(|| ctx.add_direct(black_box(x), black_box(y)));
    });

    c.bench_function("zero_overhead_smart_add", |b| {
        let ctx = SmartContext::new();
        b.iter(|| ctx.add_smart(black_box(x), black_box(y)));
    });
}

fn benchmark_multiplication(c: &mut Criterion) {
    let x = 3.0;
    let y = 4.0;

    c.bench_function("native_mul", |b| {
        b.iter(|| native_mul(black_box(x), black_box(y)));
    });

    c.bench_function("original_slow_mul", |b| {
        b.iter(|| original_slow_mul(black_box(x), black_box(y)));
    });

    c.bench_function("zero_overhead_direct_mul", |b| {
        let ctx = DirectComputeContext::new();
        b.iter(|| ctx.mul_direct(black_box(x), black_box(y)));
    });

    c.bench_function("zero_overhead_smart_mul", |b| {
        let ctx = SmartContext::new();
        b.iter(|| ctx.mul_smart(black_box(x), black_box(y)));
    });
}

fn benchmark_complex(c: &mut Criterion) {
    let x = 3.0;
    let y = 4.0;
    let z = 5.0;

    c.bench_function("native_complex", |b| {
        b.iter(|| native_complex(black_box(x), black_box(y), black_box(z)));
    });

    c.bench_function("original_slow_complex", |b| {
        b.iter(|| original_slow_complex(black_box(x), black_box(y), black_box(z)));
    });

    c.bench_function("zero_overhead_direct_complex", |b| {
        let ctx = DirectComputeContext::new();
        b.iter(|| ctx.complex_direct(black_box(x), black_box(y), black_box(z)));
    });

    c.bench_function("zero_overhead_smart_complex", |b| {
        let ctx = SmartContext::new();
        b.iter(|| ctx.complex_smart(black_box(x), black_box(y), black_box(z)));
    });
}

criterion_group!(
    benches,
    benchmark_addition,
    benchmark_multiplication,
    benchmark_complex
);
criterion_main!(benches);
