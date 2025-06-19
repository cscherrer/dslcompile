//! Benchmark: String Conversion Overhead Analysis
//!
//! This benchmark measures the performance overhead of converting AST to egglog
//! S-expressions and back, to quantify the cost of the current string-based approach.

use dslcompile::{ast::ASTRepr, symbolic::native_egglog::NativeEgglogOptimizer};
use std::time::Instant;

/// Generate a complex nested expression for benchmarking
fn create_complex_expression(depth: usize, width: usize) -> ASTRepr<f64> {
    if depth == 0 {
        if width % 2 == 0 {
            ASTRepr::Variable(width % 5)
        } else {
            ASTRepr::Constant(width as f64)
        }
    } else {
        // Create a simpler binary structure to avoid array size issues
        let left = create_complex_expression(depth - 1, width);
        let right = create_complex_expression(depth - 1, width + 1);

        if depth % 2 == 0 {
            ASTRepr::add_from_array([left, right])
        } else {
            ASTRepr::mul_from_array([left, right])
        }
    }
}

/// Benchmark string conversion performance
pub fn benchmark_string_conversion() {
    println!("🔬 Benchmarking String Conversion Overhead");
    println!("=========================================");

    let optimizer = match NativeEgglogOptimizer::new() {
        Ok(opt) => opt,
        Err(e) => {
            println!("❌ Failed to create optimizer: {}", e);
            return;
        }
    };

    // Test different expression complexities
    let test_cases = vec![
        ("Simple", 2, 2),       // Small expression
        ("Medium", 3, 3),       // Medium complexity
        ("Complex", 4, 3),      // Higher complexity
        ("Very Complex", 5, 2), // Deep nesting
    ];

    for (name, depth, width) in test_cases {
        println!(
            "\n📊 Testing {} Expression (depth={}, width={})",
            name, depth, width
        );

        let expr = create_complex_expression(depth, width);

        // Measure AST to egglog conversion
        let start = Instant::now();
        let mut conversion_times = Vec::new();

        for _ in 0..1000 {
            let conversion_start = Instant::now();
            let _egglog_str = optimizer.ast_to_egglog(&expr);
            conversion_times.push(conversion_start.elapsed());
        }

        let total_conversion = start.elapsed();
        let avg_conversion =
            conversion_times.iter().sum::<std::time::Duration>() / conversion_times.len() as u32;
        let min_conversion = conversion_times.iter().min().unwrap();
        let max_conversion = conversion_times.iter().max().unwrap();

        println!("   AST → egglog conversion:");
        println!(
            "     Total (1000 iterations): {:.3}ms",
            total_conversion.as_secs_f64() * 1000.0
        );
        println!(
            "     Average per conversion: {:.3}μs",
            avg_conversion.as_secs_f64() * 1_000_000.0
        );
        println!(
            "     Min: {:.3}μs, Max: {:.3}μs",
            min_conversion.as_secs_f64() * 1_000_000.0,
            max_conversion.as_secs_f64() * 1_000_000.0
        );

        // Measure the generated string size
        if let Ok(egglog_str) = optimizer.ast_to_egglog(&expr) {
            println!(
                "     Generated string length: {} characters",
                egglog_str.len()
            );

            // Estimate parsing overhead (simplified simulation)
            let parse_start = Instant::now();
            for _ in 0..1000 {
                let _simulated_parse = egglog_str.chars().count(); // Simulate string processing
            }
            let parse_time = parse_start.elapsed();

            println!(
                "     Simulated parse overhead: {:.3}μs/parse",
                (parse_time.as_secs_f64() * 1_000_000.0) / 1000.0
            );
        }
    }

    // Benchmark full optimization cycle
    println!("\n🚀 Full Optimization Cycle Benchmark");
    println!("=====================================");

    let test_expr = create_complex_expression(3, 3);
    let mut full_times = Vec::new();

    for i in 0..10 {
        let mut opt = NativeEgglogOptimizer::new().unwrap();
        let start = Instant::now();

        match opt.optimize(&test_expr) {
            Ok(_result) => {
                let duration = start.elapsed();
                full_times.push(duration);
                println!("   Run {}: {:.3}ms", i + 1, duration.as_secs_f64() * 1000.0);
            }
            Err(e) => {
                println!("   Run {} failed: {}", i + 1, e);
            }
        }
    }

    if !full_times.is_empty() {
        let avg_time = full_times.iter().sum::<std::time::Duration>() / full_times.len() as u32;
        let min_time = full_times.iter().min().unwrap();
        let max_time = full_times.iter().max().unwrap();

        println!("\n📈 Full Optimization Statistics:");
        println!("   Average: {:.3}ms", avg_time.as_secs_f64() * 1000.0);
        println!(
            "   Min: {:.3}ms, Max: {:.3}ms",
            min_time.as_secs_f64() * 1000.0,
            max_time.as_secs_f64() * 1000.0
        );
    }
}

/// Analyze the complexity of the current string conversion implementation
pub fn analyze_conversion_complexity() {
    println!("\n🔍 String Conversion Implementation Analysis");
    println!("===========================================");

    // Create expressions of increasing complexity to measure scaling
    let sizes = vec![1, 2, 4, 8, 16];

    for &size in &sizes {
        let expr = create_complex_expression(2, size);

        let optimizer = NativeEgglogOptimizer::new().unwrap();

        // Measure conversion time and output size
        let start = Instant::now();
        let egglog_result = optimizer.ast_to_egglog(&expr);
        let conversion_time = start.elapsed();

        match egglog_result {
            Ok(egglog_str) => {
                println!(
                    "   Expression width {}: {:.3}μs, {} chars",
                    size,
                    conversion_time.as_secs_f64() * 1_000_000.0,
                    egglog_str.len()
                );
            }
            Err(e) => {
                println!("   Expression width {}: Conversion failed: {}", size, e);
            }
        }
    }

    println!("\n💡 Key Observations:");
    println!("   - Current implementation: ~580 lines of conversion logic");
    println!("   - String generation overhead scales with expression complexity");
    println!("   - Memory allocations for string building and parsing");
    println!("   - Serialization/deserialization creates intermediate representations");
    println!("   - Error handling requires string parsing validation");
}

/// Compare theoretical performance of different approaches
pub fn compare_approach_performance() {
    println!("\n⚖️  Theoretical Performance Comparison");
    println!("====================================");

    println!("Current egglog String-Based Approach:");
    println!("   ➕ Leverages mature egglog optimization engine");
    println!("   ➕ Rich rule language with built-in features");
    println!("   ➖ String conversion overhead (measured above)");
    println!("   ➖ Memory allocation for string generation");
    println!("   ➖ String parsing validation overhead");
    println!("   ➖ Limited cost function customization");

    println!("\nDirect Egg Integration:");
    println!("   ➕ Zero string conversion overhead");
    println!("   ➕ Native Rust type integration");
    println!("   ➕ Custom cost functions with full control");
    println!("   ➕ Better debugging with Rust tools");
    println!("   ➖ Need to reimplement dependency analysis");
    println!("   ➖ Rule migration effort required");

    println!("\nCustom E-Graph Implementation:");
    println!("   ➕ Perfect fit for mathematical expressions");
    println!("   ➕ Minimal overhead, maximum performance");
    println!("   ➕ Complete control over algorithms");
    println!("   ➕ Domain-specific optimizations possible");
    println!("   ➖ Significant implementation effort");
    println!("   ➖ Need to implement core e-graph algorithms");

    println!("\nDirect egglog-rust Integration (Research Finding):");
    println!("   ➕ Partial string conversion elimination via TermDag");
    println!("   ➖ Still requires string-based rules");
    println!("   ➖ Limited API coverage and documentation");
    println!("   ➖ Rules still need string format");
}

/// Main benchmark runner
pub fn run_conversion_benchmark() {
    println!("🎯 String Conversion Performance Analysis");
    println!("========================================");

    benchmark_string_conversion();
    analyze_conversion_complexity();
    compare_approach_performance();

    println!("\n🎯 Benchmark Summary:");
    println!("   The string conversion overhead is measurable but not necessarily");
    println!("   the primary bottleneck. The main benefits of direct integration");
    println!("   would be better cost function control and native debugging.");
    println!("   ");
    println!("   For DSLCompile's mathematical optimization domain, either");
    println!("   direct egg integration or a custom e-graph implementation");
    println!("   could provide significant benefits beyond just performance.");
}

fn main() {
    run_conversion_benchmark();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runs() {
        // Just ensure the benchmark functions execute without panicking
        // We don't assert on specific timings since they're machine-dependent
        benchmark_string_conversion();
    }

    #[test]
    fn test_complex_expression_generation() {
        let expr = create_complex_expression(3, 2);
        // Verify it's actually complex
        assert!(matches!(expr, ASTRepr::Add(_) | ASTRepr::Mul(_)));
    }
}
