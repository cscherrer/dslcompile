//! Concept Demo: Sqrt Pre/Post Processing
//!
//! This demo illustrates the pre/post processing approach for sqrt() optimization.
//! We demonstrate the key ideas without running the full compilation stack.

use dslcompile::ast::ast_repr::ASTRepr;

fn main() {
    println!("=== Sqrt Pre/Post Processing Concept ===\n");

    println!("**BEFORE**: Traditional approach with separate Sqrt variant");
    println!("  User code: x.sqrt()");
    println!("  Internal: ASTRepr::Sqrt(x)");
    println!("  Problem: Need separate handling in ~30+ places");
    
    println!("\n**AFTER**: New unified approach with pre/post processing");
    
    // 1. Pre-processing: sqrt() creates Pow(x, 0.5)
    println!("\n1. PRE-PROCESSING STEP:");
    println!("   User writes: x.sqrt()");
    
    let x = ASTRepr::<f64>::Variable(0);
    let sqrt_expr = x.sqrt(); // This creates Pow(x, 0.5) internally
    
    match &sqrt_expr {
        ASTRepr::Pow(base, exp) => {
            println!("   ✅ Internal AST: Pow(x, 0.5)");
            if let (ASTRepr::Variable(0), ASTRepr::Constant(exp_val)) = (base.as_ref(), exp.as_ref()) {
                println!("      Base: Variable(0)");
                println!("      Exponent: {}", exp_val);
                
                if *exp_val == 0.5 {
                    println!("      ✅ Exponent is exactly 0.5");
                } else {
                    println!("      ❌ Exponent mismatch: {}", exp_val);
                }
            }
        }
        ASTRepr::Sqrt(_) => {
            println!("   ❌ Still using old Sqrt variant!");
        }
        _ => {
            println!("   ❓ Unexpected AST structure: {:?}", sqrt_expr);
        }
    }

    println!("\n2. UNIFIED PROCESSING:");
    println!("   All power operations (x^2, x^0.5, x^-1) handled by same code");
    println!("   - Binary exponentiation for integer powers");
    println!("   - Domain analysis for safety");
    println!("   - Common subexpression elimination");
    println!("   - Optimization passes");

    println!("\n3. POST-PROCESSING STEP (Code Generation):");
    println!("   Pattern recognition: Pow(expr, 0.5) → expr.sqrt()");
    
    // Simulate the post-processing logic
    let optimized_code = if let ASTRepr::Pow(base, exp) = &sqrt_expr {
        if let ASTRepr::Constant(exp_val) = exp.as_ref() {
            if exp_val.fract() == 0.0 && *exp_val >= i32::MIN as f64 && *exp_val <= i32::MAX as f64 {
                // Integer exponent: use powi()
                let int_exp = *exp_val as i32;
                format!("({}).powi({})", format_expr(base), int_exp)
            } else if *exp_val == 0.5 {
                // Exact 0.5: use sqrt()
                format!("({}).sqrt()", format_expr(base))
            } else {
                // Fractional exponent: use powf()
                format!("({}).powf({})", format_expr(base), exp_val)
            }
        } else {
            format!("({}).powf({})", format_expr(base), format_expr(exp))
        }
    } else {
        "unknown".to_string()
    };

    println!("   Generated code: {}", optimized_code);
    
    if optimized_code.contains(".sqrt()") {
        println!("   ✅ Post-processing successful: generates .sqrt()");
    } else {
        println!("   ❌ Post-processing failed");
    }

    println!("\n=== BENEFITS ===");
    println!("✅ User Experience: Natural sqrt() API");
    println!("✅ Code Simplification: No duplicate Sqrt handling");
    println!("✅ Performance: Both .sqrt() calls AND power optimizations");
    println!("✅ Maintainability: Single power optimization codebase");
    println!("✅ Extensibility: Easy to add x^(1/3), x^(2/3), etc.");

    println!("\n=== OPTIMIZATION EXAMPLES ===");
    
    // Example 1: Integer power optimization
    let x1 = ASTRepr::<f64>::Variable(0);
    let square_expr = ASTRepr::Pow(Box::new(x1), Box::new(ASTRepr::Constant(2.0)));
    println!("x^2 → {}", format_expr(&square_expr));
    
    // Example 2: Square root optimization  
    let x2 = ASTRepr::<f64>::Variable(0);
    let sqrt_simple = x2.sqrt();
    println!("x.sqrt() → {}", format_expr(&sqrt_simple));
    
    // Example 3: Complex expression with both optimizations
    let x_clone = ASTRepr::<f64>::Variable(0);
    let complex_expr = ASTRepr::Add(
        Box::new(ASTRepr::Pow(Box::new(x_clone), Box::new(ASTRepr::Constant(2.0)))),
        Box::new(ASTRepr::Constant(1.0))
    ).sqrt();

    println!("sqrt(x^2 + 1) → {}", format_expr(&complex_expr));
    
    if let ASTRepr::Pow(inner, exp) = &complex_expr {
        if let ASTRepr::Constant(exp_val) = exp.as_ref() {
            if *exp_val == 0.5 {
                println!("✅ Complex sqrt becomes Pow(expr, 0.5) with nested powi() optimization");
            }
        }
    }

    println!("\nThis approach eliminates the Sqrt enum variant while");
    println!("preserving all user-facing functionality and performance!");
}

fn format_expr(expr: &ASTRepr<f64>) -> String {
    match expr {
        ASTRepr::Constant(val) => format!("{}", val),
        ASTRepr::Variable(idx) => format!("x{}", idx),
        ASTRepr::Add(left, right) => format!("({} + {})", format_expr(left), format_expr(right)),
        ASTRepr::Mul(left, right) => format!("({} * {})", format_expr(left), format_expr(right)),
        ASTRepr::Pow(base, exp) => {
            // Post-processing optimization: detect patterns for specialized functions
            if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                if exp_val.fract() == 0.0 && *exp_val >= i32::MIN as f64 && *exp_val <= i32::MAX as f64 {
                    // Integer exponent: use powi()
                    let int_exp = *exp_val as i32;
                    format!("{}.powi({})", format_expr(base), int_exp)
                } else if *exp_val == 0.5 {
                    // Exact 0.5: use sqrt()
                    format!("{}.sqrt()", format_expr(base))
                } else {
                    // Fractional exponent: use powf()
                    format!("{}.powf({})", format_expr(base), exp_val)
                }
            } else {
                format!("{}.powf({})", format_expr(base), format_expr(exp))
            }
        },
        _ => "expr".to_string(),
    }
} 