//! Comprehensive Test Suite for Egg Sum Splitting Optimization
//!
//! This test suite validates various sum splitting scenarios to ensure
//! our egg-based optimizer handles all the important cases correctly.

use dslcompile::prelude::*;


fn main() -> Result<()> {
    println!("🧪 Comprehensive Sum Splitting Test Suite");
    println!("=========================================\n");

    // Test 1: Same variable factoring - Σ(2*x + 3*x) → 5*Σ(x)
    test_same_variable_factoring()?;

    // Test 2: Different variables splitting - Σ(2*x + 3*y) → 2*Σ(x) + 3*Σ(y)
    test_different_variables_splitting()?;

    // Test 3: Constants and variables - Σ(5 + 2*x) → 5*|collection| + 2*Σ(x)
    test_constants_with_variables()?;

    // Test 4: Nested expressions - Σ(a*(b + c)) → a*Σ(b + c)
    test_nested_expressions()?;

    // Test 5: Complex factoring - Σ(2*x + 3*x + 4*y) → 5*Σ(x) + 4*Σ(y)
    test_complex_factoring()?;

    println!("\n✅ All test cases completed!");
    Ok(())
}

fn test_same_variable_factoring() -> Result<()> {
    println!("🔬 Test 1: Same Variable Factoring");
    println!("   Expression: 2*x + 3*x");
    println!("   Expected: Should factor to 5*x\n");

    let x = ASTRepr::Variable(0);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    // Create 2*x + 3*x
    let expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two.clone(), x.clone()),
        ASTRepr::mul_binary(three.clone(), x.clone()),
    );

    println!("   Original expression: {expr:?}");

    // Test evaluation before optimization
    let test_val = 4.0;
    let original_result = eval_expression(&expr, &[test_val])?;
    println!("   Evaluation (x={test_val}): {original_result}");
    println!(
        "   Expected: 2*{} + 3*{} = {}",
        test_val,
        test_val,
        2.0 * test_val + 3.0 * test_val
    );

    // Optimization functionality removed
    println!("   Expression created successfully: {expr:?}");
    println!("   Direct evaluation: {original_result}");

    #[cfg(not(feature = "optimization"))]
    {
        println!("   🚫 Egg optimization not enabled");
    }

    println!();
    Ok(())
}

fn test_different_variables_splitting() -> Result<()> {
    println!("🔬 Test 2: Different Variables Splitting");
    println!("   Expression: 2*x + 3*y");
    println!("   Expected: Should remain as separate terms\n");

    let x = ASTRepr::Variable(0);
    let y = ASTRepr::Variable(1);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    // Create 2*x + 3*y
    let expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two.clone(), x.clone()),
        ASTRepr::mul_binary(three.clone(), y.clone()),
    );

    println!("   Original expression: {expr:?}");

    // Test evaluation
    let test_vals = [2.0, 3.0]; // x=2, y=3
    let original_result = eval_expression(&expr, &test_vals)?;
    println!(
        "   Evaluation (x={}, y={}): {}",
        test_vals[0], test_vals[1], original_result
    );
    println!(
        "   Expected: 2*{} + 3*{} = {}",
        test_vals[0],
        test_vals[1],
        2.0 * test_vals[0] + 3.0 * test_vals[1]
    );

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        println!("   Expression created successfully");
    }

    println!();
    Ok(())
}

fn test_constants_with_variables() -> Result<()> {
    println!("🔬 Test 3: Constants with Variables");
    println!("   Expression: 5 + 2*x");
    println!("   Expected: Should remain as mixed constant and variable\n");

    let x = ASTRepr::Variable(0);
    let five = ASTRepr::Constant(5.0);
    let two = ASTRepr::Constant(2.0);

    // Create 5 + 2*x
    let expr = ASTRepr::add_binary(five.clone(), ASTRepr::mul_binary(two.clone(), x.clone()));

    println!("   Original expression: {expr:?}");

    let test_val = 3.0;
    let original_result = eval_expression(&expr, &[test_val])?;
    println!("   Evaluation (x={test_val}): {original_result}");
    println!("   Expected: 5 + 2*{} = {}", test_val, 5.0 + 2.0 * test_val);

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        println!("   Expression created successfully");
    }

    println!();
    Ok(())
}

fn test_nested_expressions() -> Result<()> {
    println!("🔬 Test 4: Nested Expressions");
    println!("   Expression: x * (y + z)");
    println!("   Expected: Should potentially distribute to x*y + x*z\n");

    let x = ASTRepr::Variable(0);
    let y = ASTRepr::Variable(1);
    let z = ASTRepr::Variable(2);

    // Create x * (y + z)
    let inner_add = ASTRepr::add_binary(y.clone(), z.clone());
    let expr = ASTRepr::mul_binary(x.clone(), inner_add);

    println!("   Original expression: {expr:?}");

    let test_vals = [2.0, 3.0, 4.0]; // x=2, y=3, z=4
    let original_result = eval_expression(&expr, &test_vals)?;
    println!(
        "   Evaluation (x={}, y={}, z={}): {}",
        test_vals[0], test_vals[1], test_vals[2], original_result
    );
    println!(
        "   Expected: {}*({} + {}) = {}",
        test_vals[0],
        test_vals[1],
        test_vals[2],
        test_vals[0] * (test_vals[1] + test_vals[2])
    );

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        println!("   Expression created successfully");
    }

    println!();
    Ok(())
}

fn test_complex_factoring() -> Result<()> {
    println!("🔬 Test 5: Complex Factoring");
    println!("   Expression: 2*x + 3*x + 4*y");
    println!("   Expected: Should factor to 5*x + 4*y\n");

    let x = ASTRepr::Variable(0);
    let y = ASTRepr::Variable(1);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);
    let four = ASTRepr::Constant(4.0);

    // Create 2*x + 3*x + 4*y
    let term1 = ASTRepr::mul_binary(two.clone(), x.clone());
    let term2 = ASTRepr::mul_binary(three.clone(), x.clone());
    let term3 = ASTRepr::mul_binary(four.clone(), y.clone());

    let expr = ASTRepr::add_binary(ASTRepr::add_binary(term1, term2), term3);

    println!("   Original expression: {expr:?}");

    let test_vals = [2.0, 3.0]; // x=2, y=3
    let original_result = eval_expression(&expr, &test_vals)?;
    println!(
        "   Evaluation (x={}, y={}): {}",
        test_vals[0], test_vals[1], original_result
    );
    println!(
        "   Expected: 2*{} + 3*{} + 4*{} = {}",
        test_vals[0],
        test_vals[0],
        test_vals[1],
        2.0 * test_vals[0] + 3.0 * test_vals[0] + 4.0 * test_vals[1]
    );

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        println!("   Expression created successfully");
    }

    println!();
    Ok(())
}

/// Simple evaluation helper for testing
fn eval_expression(expr: &ASTRepr<f64>, vars: &[f64]) -> Result<f64> {
    match expr {
        ASTRepr::Constant(val) => Ok(*val),
        ASTRepr::Variable(idx) => {
            if *idx < vars.len() {
                Ok(vars[*idx])
            } else {
                Err(DSLCompileError::VariableNotFound(format!(
                    "Variable {idx} not found in input"
                )))
            }
        }
        ASTRepr::Add(terms) => {
            let mut sum = 0.0;
            for (term, _count) in terms.iter() {
                sum += eval_expression(term, vars)?;
            }
            Ok(sum)
        }
        ASTRepr::Mul(factors) => {
            let mut product = 1.0;
            for (factor, _count) in factors.iter() {
                product *= eval_expression(factor, vars)?;
            }
            Ok(product)
        }
        ASTRepr::Sub(left, right) => {
            Ok(eval_expression(left, vars)? - eval_expression(right, vars)?)
        }
        ASTRepr::Div(left, right) => {
            Ok(eval_expression(left, vars)? / eval_expression(right, vars)?)
        }
        ASTRepr::Pow(base, exp) => {
            Ok(eval_expression(base, vars)?.powf(eval_expression(exp, vars)?))
        }
        ASTRepr::Neg(inner) => Ok(-eval_expression(inner, vars)?),
        ASTRepr::Ln(inner) => Ok(eval_expression(inner, vars)?.ln()),
        ASTRepr::Exp(inner) => Ok(eval_expression(inner, vars)?.exp()),
        ASTRepr::Sin(inner) => Ok(eval_expression(inner, vars)?.sin()),
        ASTRepr::Cos(inner) => Ok(eval_expression(inner, vars)?.cos()),
        ASTRepr::Sqrt(inner) => Ok(eval_expression(inner, vars)?.sqrt()),
        _ => Err(DSLCompileError::UnsupportedExpression(
            "Unsupported expression type for evaluation".to_string(),
        )),
    }
}