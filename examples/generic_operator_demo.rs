//! Generic Operator Overloading Demo
//!
//! This example demonstrates the new generic operator overloading capabilities
//! of MathCompile, showing how the same natural mathematical syntax works across
//! different numeric types like f64, f32, and potentially other numeric types.

use mathcompile::final_tagless::{ASTRepr, DirectEval};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Generic Operator Overloading Demo");
    println!("=====================================\n");

    // Demonstrate f64 expressions
    println!("📊 Working with f64 expressions:");
    demo_f64_expressions()?;

    println!("\n📊 Working with f32 expressions:");
    demo_f32_expressions()?;

    println!("\n🔄 Comparing f64 vs f32 precision:");
    compare_precision()?;

    println!("\n✅ All demonstrations completed successfully!");
    Ok(())
}

fn demo_f64_expressions() -> Result<(), Box<dyn std::error::Error>> {
    // Create variables
    let x = ASTRepr::<f64>::Variable(0);
    let y = ASTRepr::<f64>::Variable(1);
    let z = ASTRepr::<f64>::Variable(2);

    // Create constants
    let two = ASTRepr::<f64>::Constant(2.0);
    let pi = ASTRepr::<f64>::Constant(std::f64::consts::PI);

    // Demonstrate natural operator syntax
    println!("  Building expressions with natural syntax:");
    
    // Linear combination: 2x + 3y
    let linear = &two * &x + &ASTRepr::<f64>::Constant(3.0) * &y;
    println!("    Linear: 2x + 3y");
    
    // Quadratic: x² + y²
    let quadratic = &x * &x + &y * &y;
    println!("    Quadratic: x² + y²");
    
    // Complex expression: sin(πx) + cos(y) - z²
    let complex = pi.sin() + y.cos() - &z * &z;
    println!("    Complex: sin(πx) + cos(y) - z²");
    
    // Evaluate expressions
    println!("\n  Evaluating with x=1.0, y=2.0, z=3.0:");
    let vars = vec![1.0_f64, 2.0, 3.0];
    
    let linear_result = DirectEval::eval_with_vars(&linear, &vars);
    let quadratic_result = DirectEval::eval_with_vars(&quadratic, &vars);
    let complex_result = DirectEval::eval_with_vars(&complex, &vars);
    
    println!("    Linear result: {:.6}", linear_result);
    println!("    Quadratic result: {:.6}", quadratic_result);
    println!("    Complex result: {:.6}", complex_result);

    Ok(())
}

fn demo_f32_expressions() -> Result<(), Box<dyn std::error::Error>> {
    // Create variables with f32 type
    let x = ASTRepr::<f32>::Variable(0);
    let y = ASTRepr::<f32>::Variable(1);

    // Create constants
    let two = ASTRepr::<f32>::Constant(2.0_f32);
    let half = ASTRepr::<f32>::Constant(0.5_f32);

    println!("  Building f32 expressions:");
    
    // Demonstrate the same operators work with f32
    let expr1 = &two * &x + &half;
    println!("    Expression 1: 2x + 0.5");
    
    let expr2 = (&x + &y) * (&x - &y); // (x + y)(x - y) = x² - y²
    println!("    Expression 2: (x + y)(x - y)");
    
    let expr3 = x.exp() + y.ln(); // exp(x) + ln(y)
    println!("    Expression 3: exp(x) + ln(y)");
    
    // Evaluate with f32 values
    println!("\n  Evaluating with x=2.0, y=3.0:");
    let vars = vec![2.0_f32, 3.0_f32];
    
    let result1 = DirectEval::eval_with_vars(&expr1, &vars);
    let result2 = DirectEval::eval_with_vars(&expr2, &vars);
    let result3 = DirectEval::eval_with_vars(&expr3, &vars);
    
    println!("    Expression 1 result: {:.6}", result1);
    println!("    Expression 2 result: {:.6}", result2);
    println!("    Expression 3 result: {:.6}", result3);

    Ok(())
}

fn compare_precision() -> Result<(), Box<dyn std::error::Error>> {
    // Create equivalent expressions in f64 and f32
    let x_f64 = ASTRepr::<f64>::Variable(0);
    let x_f32 = ASTRepr::<f32>::Variable(0);
    
    // Build the same mathematical expression: x^10 (where precision differences show)
    let mut expr_f64 = x_f64.clone();
    let mut expr_f32 = x_f32.clone();
    
    // Build x^10 by repeated multiplication
    for _ in 1..10 {
        expr_f64 = &expr_f64 * &x_f64;
        expr_f32 = &expr_f32 * &x_f32;
    }
    
    println!("  Comparing precision for x^10:");
    
    // Test with a value that will show precision differences
    let test_value_f64 = 1.1_f64;
    let test_value_f32 = 1.1_f32;
    
    let result_f64 = DirectEval::eval_with_vars(&expr_f64, &[test_value_f64]);
    let result_f32 = DirectEval::eval_with_vars(&expr_f32, &[test_value_f32]);
    
    // Also compute the reference value
    let reference = test_value_f64.powi(10);
    
    println!("    Input value: 1.1");
    println!("    f64 result:  {:.15}", result_f64);
    println!("    f32 result:  {:.15}", result_f32 as f64);
    println!("    Reference:   {:.15}", reference);
    println!("    f64 error:   {:.2e}", (result_f64 - reference).abs());
    println!("    f32 error:   {:.2e}", (result_f32 as f64 - reference).abs());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_operators_f64() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let c = ASTRepr::<f64>::Constant(5.0);
        
        // Test all operators
        let add_expr = &x + &y;
        let sub_expr = &x - &y;
        let mul_expr = &x * &c;
        let div_expr = &x / &c;
        let neg_expr = -&x;
        
        // Verify structure
        assert!(matches!(add_expr, ASTRepr::Add(_, _)));
        assert!(matches!(sub_expr, ASTRepr::Sub(_, _)));
        assert!(matches!(mul_expr, ASTRepr::Mul(_, _)));
        assert!(matches!(div_expr, ASTRepr::Div(_, _)));
        assert!(matches!(neg_expr, ASTRepr::Neg(_)));
    }

    #[test]
    fn test_generic_operators_f32() {
        let x = ASTRepr::<f32>::Variable(0);
        let y = ASTRepr::<f32>::Variable(1);
        let c = ASTRepr::<f32>::Constant(5.0_f32);
        
        // Test all operators work with f32
        let add_expr = &x + &y;
        let sub_expr = &x - &y;
        let mul_expr = &x * &c;
        let div_expr = &x / &c;
        let neg_expr = -&x;
        
        // Verify structure
        assert!(matches!(add_expr, ASTRepr::Add(_, _)));
        assert!(matches!(sub_expr, ASTRepr::Sub(_, _)));
        assert!(matches!(mul_expr, ASTRepr::Mul(_, _)));
        assert!(matches!(div_expr, ASTRepr::Div(_, _)));
        assert!(matches!(neg_expr, ASTRepr::Neg(_)));
    }

    #[test]
    fn test_transcendental_functions() {
        let x_f64 = ASTRepr::<f64>::Variable(0);
        let x_f32 = ASTRepr::<f32>::Variable(0);
        
        // Test transcendental functions work with both types
        let sin_f64 = x_f64.sin();
        let cos_f32 = x_f32.cos();
        let exp_f64 = x_f64.exp();
        let ln_f32 = x_f32.ln();
        let sqrt_f64 = x_f64.sqrt();
        
        assert!(matches!(sin_f64, ASTRepr::Sin(_)));
        assert!(matches!(cos_f32, ASTRepr::Cos(_)));
        assert!(matches!(exp_f64, ASTRepr::Exp(_)));
        assert!(matches!(ln_f32, ASTRepr::Ln(_)));
        assert!(matches!(sqrt_f64, ASTRepr::Sqrt(_)));
    }

    #[test]
    fn test_mixed_operations() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        
        // Complex expression combining multiple operations
        let expr = (&x + &y).sin() * (&x - &y).cos() + (&x * &y).sqrt();
        
        // Should evaluate without panicking
        let result = DirectEval::eval_with_vars(&expr, &[2.0, 3.0]);
        assert!(result.is_finite());
    }
} 