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
                
                if (exp_val - 0.5).abs() < 1e-15 {
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
            if (exp_val - 0.5).abs() < 1e-15 {
                format!("({}).sqrt()", format_expr(base))
            } else {
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

    println!("\n=== COMPLEX EXAMPLE ===");
    let complex_expr = ASTRepr::Add(
        Box::new(ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::Constant(2.0)))),
        Box::new(ASTRepr::Constant(1.0))
    ).sqrt();

    println!("Expression: sqrt(x^2 + 1)");
    if let ASTRepr::Pow(inner, exp) = &complex_expr {
        if let ASTRepr::Constant(exp_val) = exp.as_ref() {
            if (exp_val - 0.5).abs() < 1e-15 {
                println!("✅ Complex sqrt also becomes Pow(expr, 0.5)");
                println!("   Can be optimized to: ({}).sqrt()", format_expr(inner));
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
        ASTRepr::Pow(base, exp) => format!("{}.powf({})", format_expr(base), format_expr(exp)),
        _ => "expr".to_string(),
    }
} 