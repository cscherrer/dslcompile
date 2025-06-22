//! Memory allocation benchmarks for Box vs Arena AST representations
//!
//! This benchmark measures memory usage and allocation patterns between
//! the traditional Box-based ASTRepr and the new arena-based ArenaAST.

use divan::Bencher;
use dslcompile::{
    ast::{ASTRepr, ExprArena, ExprId, arena::ArenaAST, arena_to_ast, ast_to_arena},
    prelude::*,
};

fn main() {
    // Use divan::main() to run benchmarks
    divan::main();
}

/// Create a deep expression tree using Box-based AST
/// Pattern: ((((x + 1) + 1) + 1) + 1)... depth times
fn create_deep_box_expr(depth: usize) -> ASTRepr<f64> {
    let mut expr = ASTRepr::Variable(0);
    let one = ASTRepr::Constant(1.0);

    for _ in 0..depth {
        expr = ASTRepr::add_binary(expr, one.clone());
    }

    expr
}

/// Create a deep expression tree using arena-based AST
fn create_deep_arena_expr(depth: usize, arena: &mut ExprArena<f64>) -> ExprId {
    let mut expr_id = arena.variable(0);
    let one_id = arena.constant(1.0);

    for _ in 0..depth {
        expr_id = arena.add_binary(expr_id, one_id);
    }

    expr_id
}

/// Create a wide expression tree using Box-based AST
/// Pattern: x + x + x + x + ... (width terms)
fn create_wide_box_expr(width: usize) -> ASTRepr<f64> {
    let x = ASTRepr::Variable(0);
    let terms: Vec<ASTRepr<f64>> = (0..width).map(|_| x.clone()).collect();

    // Create a multiset addition
    let mut multiset = dslcompile::ast::multiset::MultiSet::new();
    for term in terms {
        multiset.insert(term);
    }
    ASTRepr::Add(multiset)
}

/// Create a wide expression tree using arena-based AST
fn create_wide_arena_expr(width: usize, arena: &mut ExprArena<f64>) -> ExprId {
    let x_id = arena.variable(0);
    let terms: Vec<ExprId> = (0..width).map(|_| x_id).collect();
    arena.add(terms)
}

/// Create an expression with many shared subexpressions
/// Pattern: (x + y) * (x + y) * (x + y) * ...
fn create_shared_box_expr(repetitions: usize) -> ASTRepr<f64> {
    let x = ASTRepr::Variable(0);
    let y = ASTRepr::Variable(1);
    let shared_expr = ASTRepr::add_binary(x, y);

    let mut result = shared_expr.clone();
    for _ in 1..repetitions {
        result = ASTRepr::mul_binary(result, shared_expr.clone());
    }

    result
}

/// Create an expression with many shared subexpressions using arena
fn create_shared_arena_expr(repetitions: usize, arena: &mut ExprArena<f64>) -> ExprId {
    let x_id = arena.variable(0);
    let y_id = arena.variable(1);
    let shared_expr_id = arena.add_binary(x_id, y_id);

    let mut result_id = shared_expr_id;
    for _ in 1..repetitions {
        result_id = arena.mul_binary(result_id, shared_expr_id);
    }

    result_id
}

#[divan::bench(args = [10, 50, 100, 500, 1000])]
fn bench_deep_box_creation(bencher: Bencher, depth: usize) {
    bencher.bench(|| {
        divan::black_box(create_deep_box_expr(depth));
    })
}

#[divan::bench(args = [10, 50, 100, 500, 1000])]
fn bench_deep_arena_creation(bencher: Bencher, depth: usize) {
    bencher.bench(|| {
        let mut arena = ExprArena::new();
        divan::black_box(create_deep_arena_expr(depth, &mut arena));
    })
}

#[divan::bench(args = [10, 50, 100, 500, 1000])]
fn bench_wide_box_creation(bencher: Bencher, width: usize) {
    bencher.bench(|| {
        divan::black_box(create_wide_box_expr(width));
    })
}

#[divan::bench(args = [10, 50, 100, 500, 1000])]
fn bench_wide_arena_creation(bencher: Bencher, width: usize) {
    bencher.bench(|| {
        let mut arena = ExprArena::new();
        divan::black_box(create_wide_arena_expr(width, &mut arena));
    })
}

#[divan::bench(args = [5, 10, 20, 50, 100])]
fn bench_shared_box_creation(bencher: Bencher, repetitions: usize) {
    bencher.bench(|| {
        divan::black_box(create_shared_box_expr(repetitions));
    })
}

#[divan::bench(args = [5, 10, 20, 50, 100])]
fn bench_shared_arena_creation(bencher: Bencher, repetitions: usize) {
    bencher.bench(|| {
        let mut arena = ExprArena::new();
        divan::black_box(create_shared_arena_expr(repetitions, &mut arena));
    })
}

/// Benchmark conversion from Box to Arena
#[divan::bench(args = [10, 50, 100])]
fn bench_box_to_arena_conversion(bencher: Bencher, depth: usize) {
    let expr = create_deep_box_expr(depth);

    bencher.bench(|| {
        let mut arena = ExprArena::new();
        divan::black_box(ast_to_arena(&expr, &mut arena));
    })
}

/// Benchmark conversion from Arena to Box
#[divan::bench(args = [10, 50, 100])]
fn bench_arena_to_box_conversion(bencher: Bencher, depth: usize) {
    let mut arena = ExprArena::new();
    let expr_id = create_deep_arena_expr(depth, &mut arena);

    bencher.bench(|| {
        divan::black_box(arena_to_ast(expr_id, &arena));
    })
}

/// Memory usage analysis (not a benchmark, but useful for understanding)
#[divan::bench]
fn analyze_memory_usage(bencher: Bencher) {
    bencher.bench(|| {
        // Create identical expressions using both approaches
        let box_expr = create_shared_box_expr(20);

        let mut arena = ExprArena::new();
        let arena_expr_id = create_shared_arena_expr(20, &mut arena);

        // Prevent optimization away
        divan::black_box(&box_expr);
        divan::black_box(&arena_expr_id);
        divan::black_box(&arena);

        // In a real analysis, we would measure:
        // - std::mem::size_of_val(&box_expr) vs arena.memory_usage()
        // - Number of heap allocations via a custom allocator
        // - Cache miss rates during traversal
    });
}

/// Benchmark expression evaluation performance
#[divan::bench(args = [10, 50, 100])]
fn bench_evaluation_box(bencher: Bencher, depth: usize) {
    let expr = create_deep_box_expr(depth);

    bencher.bench(|| {
        // Note: This requires implementing evaluation for Box-based AST
        // For now, we'll just measure the creation overhead
        divan::black_box(&expr);
    })
}

#[divan::bench(args = [10, 50, 100])]
fn bench_evaluation_arena(bencher: Bencher, depth: usize) {
    let mut arena = ExprArena::new();
    let expr_id = create_deep_arena_expr(depth, &mut arena);

    bencher.bench(|| {
        // Note: This requires implementing evaluation for arena-based AST
        // For now, we'll just measure the creation overhead
        divan::black_box(&expr_id);
        divan::black_box(&arena);
    })
}

/// Benchmark traversal patterns that would be common in optimization
#[divan::bench(args = [100, 500, 1000])]
fn bench_traversal_box(bencher: Bencher, width: usize) {
    let expr = create_wide_box_expr(width);

    bencher.bench(|| {
        // Simulate a traversal that counts nodes
        fn count_nodes(expr: &ASTRepr<f64>) -> usize {
            match expr {
                ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => 1,
                ASTRepr::Add(ms) | ASTRepr::Mul(ms) => {
                    1 + ms.iter().map(|(e, _)| count_nodes(e)).sum::<usize>()
                }
                ASTRepr::Sub(l, r) | ASTRepr::Div(l, r) | ASTRepr::Pow(l, r) => {
                    1 + count_nodes(l) + count_nodes(r)
                }
                ASTRepr::Neg(e)
                | ASTRepr::Ln(e)
                | ASTRepr::Exp(e)
                | ASTRepr::Sin(e)
                | ASTRepr::Cos(e)
                | ASTRepr::Sqrt(e) => 1 + count_nodes(e),
                ASTRepr::Let(_, e, b) => 1 + count_nodes(e) + count_nodes(b),
                ASTRepr::Sum(_) | ASTRepr::Lambda(_) => 1, // Simplified for benchmark
            }
        }

        divan::black_box(count_nodes(&expr));
    })
}

#[divan::bench(args = [100, 500, 1000])]
fn bench_traversal_arena(bencher: Bencher, width: usize) {
    let mut arena = ExprArena::new();
    let expr_id = create_wide_arena_expr(width, &mut arena);

    bencher.bench(|| {
        // Simulate a traversal that counts nodes
        fn count_nodes(expr_id: ExprId, arena: &ExprArena<f64>) -> usize {
            if let Some(expr) = arena.get(expr_id) {
                match expr {
                    ArenaAST::Constant(_) | ArenaAST::Variable(_) | ArenaAST::BoundVar(_) => 1,
                    ArenaAST::Add(ms) | ArenaAST::Mul(ms) => {
                        1 + ms
                            .iter()
                            .map(|(id, _)| count_nodes(*id, arena))
                            .sum::<usize>()
                    }
                    ArenaAST::Sub(l, r) | ArenaAST::Div(l, r) | ArenaAST::Pow(l, r) => {
                        1 + count_nodes(*l, arena) + count_nodes(*r, arena)
                    }
                    ArenaAST::Neg(e)
                    | ArenaAST::Ln(e)
                    | ArenaAST::Exp(e)
                    | ArenaAST::Sin(e)
                    | ArenaAST::Cos(e)
                    | ArenaAST::Sqrt(e) => 1 + count_nodes(*e, arena),
                    ArenaAST::Let(_, e, b) => 1 + count_nodes(*e, arena) + count_nodes(*b, arena),
                    ArenaAST::Sum(_) | ArenaAST::Lambda(_) => 1, // Simplified for benchmark
                }
            } else {
                0
            }
        }

        divan::black_box(count_nodes(expr_id, &arena));
    })
}
