//! Unified Variable API Demo
//!
//! This demo shows the new type-driven variable API where:
//! - ctx.var::<f64>() → Scalar variable for arithmetic
//! - ctx.var::<Vec<f64>>() → Collection variable for iteration

use dslcompile::prelude::*;

fn main() {
    println!("🚀 Unified Variable API Demo");
    println!("============================\n");

    let ctx = DynamicContext::new();

    // ============================================================================
    // DEMO 1: Scalar Variables (f64) - Arithmetic Operations
    // ============================================================================
    println!("📊 Demo 1: Scalar Variables");
    println!("---------------------------");
    
    // Create scalar variables using unified API
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    
    // Build arithmetic expression
    let expr = &x * 2.0 + &y * 3.0;
    println!("Expression: x * 2.0 + y * 3.0");
    println!("AST: {:?}", expr.as_ast());
    
    // Evaluate with concrete values
    let result = ctx.eval(&expr, &[5.0, 10.0]);
    println!("Result with x=5.0, y=10.0: {}\n", result);

    // ============================================================================
    // DEMO 2: Operator Overloading - Natural Mathematical Syntax
    // ============================================================================
    println!("📊 Demo 2: Operator Overloading");
    println!("--------------------------------");
    
    // Natural mathematical syntax with operator overloading
    let polynomial = &x * &x + 2.0 * &x + 1.0;
    println!("Expression: x² + 2x + 1");
    println!("AST: {:?}", polynomial.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&polynomial));

    // ============================================================================
    // DEMO 3: Complex Expressions - Transcendental Functions
    // ============================================================================
    println!("📊 Demo 3: Complex Expressions");
    println!("------------------------------");
    
    // Complex expression with transcendental functions
    let z = ctx.var::<f64>();
    let complex_expr = z.clone().sin() + z.clone().cos().exp();
    println!("Expression: sin(z) + exp(cos(z))");
    println!("AST: {:?}", complex_expr.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&complex_expr));

    // ============================================================================
    // DEMO 4: Mixed Operations - Multiple Variables
    // ============================================================================
    println!("📊 Demo 4: Mixed Operations");
    println!("---------------------------");
    
    // Multiple variables in complex expression
    let a = ctx.var::<f64>();
    let b = ctx.var::<f64>();
    let c = ctx.var::<f64>();
    
    // Quadratic expression: a*x² + b*x + c
    let quadratic = &a * &x * &x + &b * &x + &c;
    println!("Expression: a*x² + b*x + c");
    println!("AST: {:?}", quadratic.as_ast());
    println!("Pretty: {}\n", ctx.pretty_print(&quadratic));

    println!("✅ SUCCESS: Unified variable API working!");
    println!("   🎯 Single ctx.var::<T>() method for all variable types");
    println!("   🎯 Automatic operator overloading for natural syntax");
    println!("   🎯 Clean type-safe expression building");
    println!("   🎯 Support for transcendental functions");
    println!("   🎯 Extensible for future types (Matrix<f64>, etc.)");
} 