use dslcompile::prelude::*;
use std::time::Instant;

/// Debug function to isolate JIT vs interpretation issues
pub fn debug_interpretation_only() {
    println!("🔍 Debug: Testing interpretation-only mode");
    println!("==========================================");
    
    // Create context with interpretation-only strategy
    let math = DynamicContext::new_interpreter();
    let x = math.var();
    let y = math.var();

    // Test the failing expression with interpretation only
    let expr = &x * &x + 2.0 * &x * &y + &y * &y; // (x + y)²
    
    println!("Expression AST: {:?}", expr.as_ast());
    println!("JIT Strategy: {:?}", math.jit_stats().strategy);
    
    let result = math.eval(&expr, &[3.0, 4.0]);
    println!("Result with interpretation-only: {}", result);
    println!("Expected: 49.0 (since (3+4)² = 49)");
    
    // Test simpler expressions to isolate the issue
    let simple_add = &x + &y;
    let simple_mul = &x * &y;
    let x_squared = &x * &x;
    let y_squared = &y * &y;
    let cross_term = 2.0 * &x * &y;
    
    println!("\nBreaking down the expression:");
    println!("x + y = {}", math.eval(&simple_add, &[3.0, 4.0]));
    println!("x * y = {}", math.eval(&simple_mul, &[3.0, 4.0]));
    println!("x² = {}", math.eval(&x_squared, &[3.0, 4.0]));
    println!("y² = {}", math.eval(&y_squared, &[3.0, 4.0]));
    println!("2xy = {}", math.eval(&cross_term, &[3.0, 4.0]));
    
    // Manual reconstruction
    let manual_expr = x_squared + cross_term + y_squared;
    println!("Manual reconstruction (x² + 2xy + y²) = {}", math.eval(&manual_expr, &[3.0, 4.0]));
}

/// Debug function to analyze the failing test case
pub fn debug_failing_test() {
    println!("🔍 Debug: Analyzing failing test case");
    println!("=====================================");
    
    let math = DynamicContext::new();
    let x = math.var();
    let y = math.var();

    // The failing expression from the test
    let expr = &x * &x + 2.0 * &x * &y + &y * &y;
    
    println!("Variables created:");
    println!("  x: Variable(0)");
    println!("  y: Variable(1)");
    
    println!("\nExpression AST: {:?}", expr.as_ast());
    
    // Test simple expressions first
    println!("\nTesting simple expressions:");
    let simple_x = &x;
    let simple_y = &y;
    println!("x = {}", math.eval(simple_x, &[3.0, 4.0]));
    println!("y = {}", math.eval(simple_y, &[3.0, 4.0]));
    
    let add_expr = &x + &y;
    let mul_expr = &x * &y;
    println!("x + y = {}", math.eval(&add_expr, &[3.0, 4.0]));
    println!("x * y = {}", math.eval(&mul_expr, &[3.0, 4.0]));
    
    // Test the complex expression
    println!("\nTesting complex expression:");
    let result = math.eval(&expr, &[3.0, 4.0]);
    println!("x² + 2xy + y² = {}", result);
    println!("Expected: 49.0 (since (3+4)² = 49)");
    
    if (result - 49.0).abs() < 1e-10 {
        println!("✅ Test PASSED!");
    } else {
        println!("❌ Test FAILED! Got {} instead of 49.0", result);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Zero-Overhead Core Optimization Demo");
    println!("=======================================");

    // Run debug analysis first
    debug_failing_test();
    println!();
    debug_interpretation_only();

    Ok(())
}

/// Direct computation context - performs immediate computation without expression trees
#[derive(Debug, Clone)]
pub struct DirectComputeContext;

impl DirectComputeContext {
    pub fn new() -> Self {
        Self
    }
    
    /// Direct computation: x² + 2xy + y²
    pub fn quadratic_expansion(&self, x: f64, y: f64) -> f64 {
        x * x + 2.0 * x * y + y * y
    }
    
    /// Direct computation: ax² + bx + c
    pub fn quadratic(&self, a: f64, b: f64, c: f64, x: f64) -> f64 {
        a * x * x + b * x + c
    }
}

/// Performance test comparing different approaches
pub fn performance_test() {
    println!("⚡ Performance Comparison");
    println!("========================");
    
    let iterations = 1_000_000;
    let x = 3.0;
    let y = 4.0;
    
    // Direct Rust baseline
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += x * x + 2.0 * x * y + y * y;
    }
    let direct_time = start.elapsed();
    println!("Direct Rust: {:.2?} (result: {})", direct_time, sum / iterations as f64);
    
    // DirectComputeContext
    let direct_ctx = DirectComputeContext::new();
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += direct_ctx.quadratic_expansion(x, y);
    }
    let direct_ctx_time = start.elapsed();
    println!("DirectComputeContext: {:.2?} (result: {})", direct_ctx_time, sum / iterations as f64);
    
    // DynamicContext (interpretation)
    let math = DynamicContext::new_interpreter();
    let x_var = math.var();
    let y_var = math.var();
    let expr = &x_var * &x_var + 2.0 * &x_var * &y_var + &y_var * &y_var;
    
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += math.eval(&expr, &[x, y]);
    }
    let dynamic_time = start.elapsed();
    println!("DynamicContext (interp): {:.2?} (result: {})", dynamic_time, sum / iterations as f64);
    
    // Performance ratios
    let direct_ratio = direct_ctx_time.as_nanos() as f64 / direct_time.as_nanos() as f64;
    let dynamic_ratio = dynamic_time.as_nanos() as f64 / direct_time.as_nanos() as f64;
    
    println!("\nPerformance ratios (vs Direct Rust):");
    println!("DirectComputeContext: {:.2}x", direct_ratio);
    println!("DynamicContext: {:.2}x", dynamic_ratio);
} 