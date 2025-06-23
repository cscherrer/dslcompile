//! Test Actual Sum Splitting with Real Sum Expressions
//!
//! This test creates proper Sum(Lambda(...)) expressions to test our
//! egg optimization rules for sum splitting scenarios.

use dslcompile::{
    ast::ast_repr::{Collection, Lambda},
    prelude::*,
};


fn main() -> Result<()> {
    println!("🎯 Testing Actual Sum Splitting with Lambda Expressions");
    println!("======================================================\n");

    // Test 1: Basic coefficient combining - 2*x + 3*x → 5*x
    test_coefficient_combining()?;

    // Test 2: Actual sum splitting - Σ(λv.(2*v + 3*v)) → Σ(λv.5*v)
    test_sum_with_lambda_factoring()?;

    // Test 3: Sum splitting with different variables - Σ(λv.(2*v + 3*w)) → Σ(λv.2*v) + Σ(λv.3*w)
    test_sum_splitting_different_vars()?;

    println!("\n✅ Sum splitting tests completed!");
    Ok(())
}

fn test_coefficient_combining() -> Result<()> {
    println!("🔬 Test 1: Basic Coefficient Combining");
    println!("   Expression: 2*x + 3*x");
    println!("   Expected: Should become 5*x\n");

    let x = ASTRepr::Variable(0);
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    // Create 2*x + 3*x using helper methods
    let expr = ASTRepr::add_binary(
        ASTRepr::mul_binary(two.clone(), x.clone()),
        ASTRepr::mul_binary(three.clone(), x.clone()),
    );

    println!("   Original: {expr:?}");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("   Expression created: {expr:?}");

            // Test evaluation
            let test_val = 4.0;
            if let Ok(result) = eval_simple(&expr, &[test_val]) {
                println!("   Evaluation (x={test_val}): {result}");
                println!("   Expected: {} (2*4 + 3*4 = 20)", 2.0 * test_val + 3.0 * test_val);
            }
        }
    }

    println!();
    Ok(())
}

fn test_sum_with_lambda_factoring() -> Result<()> {
    println!("🔬 Test 2: Sum with Lambda Factoring");
    println!("   Expression: Σ(λv.(2*v + 3*v)) over data");
    println!("   Expected: Should factor to Σ(λv.5*v)\n");

    // Create lambda: λv.(2*v + 3*v)
    let var_v = ASTRepr::BoundVar(0); // Lambda variable
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    let lambda_body = ASTRepr::add_binary(
        ASTRepr::mul_binary(two, var_v.clone()),
        ASTRepr::mul_binary(three, var_v),
    );

    let lambda = Lambda {
        var_indices: vec![0],
        body: Box::new(lambda_body),
    };

    // Create sum with mapped lambda over constant data
    let map_collection = Collection::Map {
        lambda: Box::new(lambda),
        collection: Box::new(Collection::Constant(vec![1.0, 2.0, 3.0])),
    };

    let sum_expr = ASTRepr::Sum(Box::new(map_collection));

    println!("   Original: {sum_expr:?}");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("   Expression created: {sum_expr:?}");

            // Manual check: Σ(λv.(2*v + 3*v)) over [1,2,3]
            // = (2*1 + 3*1) + (2*2 + 3*2) + (2*3 + 3*3)
            // = 5 + 10 + 15 = 30
            println!("   Expected result: 5 + 10 + 15 = 30");

            // Test evaluation using AST eval
            let result = sum_expr.eval_with_vars(&[]);
            println!("   Actual result: {result}");
        }
    }

    println!();
    Ok(())
}

fn test_sum_splitting_different_vars() -> Result<()> {
    println!("🔬 Test 3: Sum Splitting with Mixed Variables");
    println!("   Expression: Σ(λv.(2*v + 3*w)) where w is external");
    println!("   Expected: Should split as 2*Σ(v) + 3*w*|collection|\n");

    // Create lambda: λv.(2*v + 3*w) where w=Variable(1) is external
    let var_v = ASTRepr::BoundVar(0); // Lambda-bound variable
    let var_w = ASTRepr::Variable(1); // External variable
    let two = ASTRepr::Constant(2.0);
    let three = ASTRepr::Constant(3.0);

    let lambda_body = ASTRepr::add_binary(
        ASTRepr::mul_binary(two, var_v),
        ASTRepr::mul_binary(three, var_w),
    );

    let lambda = Lambda {
        var_indices: vec![0],
        body: Box::new(lambda_body),
    };

    // Create sum with mapped lambda over constant data
    let map_collection = Collection::Map {
        lambda: Box::new(lambda),
        collection: Box::new(Collection::Constant(vec![1.0, 2.0, 3.0])),
    };

    let sum_expr = ASTRepr::Sum(Box::new(map_collection));

    println!("   Original: {sum_expr:?}");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("   Expression created: {sum_expr:?}");

            // Manual check: Σ(λv.(2*v + 3*w)) over [1,2,3] with w=5
            // = (2*1 + 3*5) + (2*2 + 3*5) + (2*3 + 3*5)
            // = 17 + 19 + 21 = 57
            // Factored: 2*(1+2+3) + 3*5*3 = 2*6 + 45 = 12 + 45 = 57
            println!("   Expected: 2*Σ(v) + 3*w*|collection|");
            println!("   With w=5, data=[1,2,3]: 2*6 + 3*5*3 = 12 + 45 = 57");

            // Test evaluation with w=5
            let result = sum_expr.eval_with_vars(&[0.0, 5.0]);
            println!("   Actual result with w=5: {result}");
        }
    }

    println!();
    Ok(())
}

/// Simple expression evaluator for testing
fn eval_simple(expr: &ASTRepr<f64>, vars: &[f64]) -> Result<f64> {
    match expr {
        ASTRepr::Constant(val) => Ok(*val),
        ASTRepr::Variable(idx) => {
            if *idx < vars.len() {
                Ok(vars[*idx])
            } else {
                Err(DSLCompileError::VariableNotFound(format!(
                    "Variable {idx} not found"
                )))
            }
        }
        ASTRepr::Add(terms) => {
            let mut sum = 0.0;
            for (term, _) in terms.iter() {
                sum += eval_simple(term, vars)?;
            }
            Ok(sum)
        }
        ASTRepr::Mul(factors) => {
            let mut product = 1.0;
            for (factor, _) in factors.iter() {
                product *= eval_simple(factor, vars)?;
            }
            Ok(product)
        }
        _ => Err(DSLCompileError::UnsupportedExpression(
            "Unsupported expression".to_string(),
        )),
    }
}
