//! Simple Performance Benchmark for Frunk-based Unified API
//!
//! This benchmark compares the frunk-based approach against naive implementations
//! to verify the zero-cost abstraction claims.

use criterion::{Criterion, criterion_group, criterion_main};
use frunk::hlist;
use std::collections::HashMap;
use std::hint::black_box;

// ============================================================================
// BASELINE: Direct Hand-Written Operations
// ============================================================================

fn direct_sum_3(a: f64, b: f64, c: f64) -> f64 {
    a + b + c
}

fn direct_sum_mixed(scalar1: f64, vec_sum: f64, scalar2: f64, index: f64) -> f64 {
    scalar1 + vec_sum + scalar2 + index
}

// ============================================================================
// NAIVE HASHMAP APPROACH (What We Want to Beat)
// ============================================================================

fn hashmap_sum(values: &HashMap<&str, f64>) -> f64 {
    values.values().sum()
}

// ============================================================================
// FRUNK APPROACH TEST
// ============================================================================

fn frunk_sum_3(args: frunk::HCons<f64, frunk::HCons<f64, frunk::HCons<f64, frunk::HNil>>>) -> f64 {
    args.head + args.tail.head + args.tail.tail.head
}

fn frunk_sum_mixed(
    args: frunk::HCons<
        f64,
        frunk::HCons<Vec<f64>, frunk::HCons<f64, frunk::HCons<usize, frunk::HNil>>>,
    >,
) -> f64 {
    let scalar1 = args.head;
    let vec_sum: f64 = args.tail.head.iter().sum();
    let scalar2 = args.tail.tail.head;
    let index = args.tail.tail.tail.head as f64;
    scalar1 + vec_sum + scalar2 + index
}

// ============================================================================
// BENCHMARKS
// ============================================================================

fn bench_direct_operations(c: &mut Criterion) {
    c.bench_function("direct_sum_3_scalars", |b| {
        b.iter(|| {
            let result = direct_sum_3(black_box(1.5), black_box(2.7), black_box(3.2));
            black_box(result)
        });
    });

    c.bench_function("direct_sum_mixed", |b| {
        b.iter(|| {
            let result = direct_sum_mixed(
                black_box(1.5),
                black_box(6.0), // Pre-computed vector sum
                black_box(3.2),
                black_box(42.0),
            );
            black_box(result)
        });
    });
}

fn bench_hashmap_approach(c: &mut Criterion) {
    c.bench_function("hashmap_sum_3_scalars", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            map.insert("a", black_box(1.5));
            map.insert("b", black_box(2.7));
            map.insert("c", black_box(3.2));
            let result = hashmap_sum(&map);
            black_box(result)
        });
    });

    c.bench_function("hashmap_sum_mixed", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            map.insert("scalar1", black_box(1.5));
            map.insert("vec_sum", black_box(6.0)); // Pre-computed
            map.insert("scalar2", black_box(3.2));
            map.insert("index", black_box(42.0));
            let result = hashmap_sum(&map);
            black_box(result)
        });
    });
}

fn bench_frunk_approach(c: &mut Criterion) {
    c.bench_function("frunk_sum_3_scalars", |b| {
        b.iter(|| {
            let args = hlist![black_box(1.5_f64), black_box(2.7_f64), black_box(3.2_f64)];
            let result = frunk_sum_3(args);
            black_box(result)
        });
    });

    c.bench_function("frunk_sum_mixed", |b| {
        let test_vector = vec![1.0, 2.0, 3.0];
        b.iter(|| {
            let args = hlist![
                black_box(1.5_f64),
                black_box(test_vector.clone()),
                black_box(3.2_f64),
                black_box(42_usize)
            ];
            let result = frunk_sum_mixed(args);
            black_box(result)
        });
    });
}

fn bench_hlist_creation(c: &mut Criterion) {
    c.bench_function("hlist_creation", |b| {
        b.iter(|| {
            let args = hlist![
                black_box(1.5_f64),
                black_box(vec![1.0, 2.0, 3.0]),
                black_box(42_usize),
                black_box(true)
            ];
            black_box(args)
        });
    });

    c.bench_function("hashmap_creation", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            map.insert("scalar", black_box(1.5));
            map.insert("vec_len", black_box(3.0)); // Simulate processing vector
            map.insert("index", black_box(42.0));
            map.insert("flag", black_box(1.0));
            black_box(map)
        });
    });
}

fn bench_scaling(c: &mut Criterion) {
    // Test with larger argument lists
    c.bench_function("direct_sum_10_args", |b| {
        b.iter(|| {
            let result = black_box(1.0)
                + black_box(2.0)
                + black_box(3.0)
                + black_box(4.0)
                + black_box(5.0)
                + black_box(6.0)
                + black_box(7.0)
                + black_box(8.0)
                + black_box(9.0)
                + black_box(10.0);
            black_box(result)
        });
    });

    c.bench_function("hashmap_sum_10_args", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for i in 1..=10 {
                map.insert(format!("arg_{i}"), black_box(f64::from(i)));
            }
            let result: f64 = map.values().sum();
            black_box(result)
        });
    });
}

criterion_group!(
    benches,
    bench_direct_operations,
    bench_hashmap_approach,
    bench_frunk_approach,
    bench_hlist_creation,
    bench_scaling
);

criterion_main!(benches);
