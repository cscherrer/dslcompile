//! Quick Frunk Performance Test

use criterion::{Criterion, criterion_group, criterion_main};
use frunk::hlist;
use std::collections::HashMap;
use std::hint::black_box;

fn direct_sum(a: f64, b: f64, c: f64) -> f64 {
    a + b + c
}

fn hashmap_sum() -> f64 {
    let mut map = HashMap::new();
    map.insert("a", 1.5);
    map.insert("b", 2.7);
    map.insert("c", 3.2);
    map.values().sum()
}

fn frunk_sum() -> f64 {
    let args = hlist![1.5_f64, 2.7_f64, 3.2_f64];
    args.head + args.tail.head + args.tail.tail.head
}

fn bench_comparison(c: &mut Criterion) {
    c.bench_function("direct_sum", |b| {
        b.iter(|| {
            let result = direct_sum(black_box(1.5), black_box(2.7), black_box(3.2));
            black_box(result)
        });
    });

    c.bench_function("hashmap_sum", |b| {
        b.iter(|| {
            let result = hashmap_sum();
            black_box(result)
        });
    });

    c.bench_function("frunk_sum", |b| {
        b.iter(|| {
            let result = frunk_sum();
            black_box(result)
        });
    });
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
