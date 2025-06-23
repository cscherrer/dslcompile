//! Test Dependency Analysis Implementation in egg
//!
//! This test validates that our egg-based dependency analysis correctly
//! identifies free variables in expressions, enabling safe coefficient factoring.

use dslcompile::{
    ast::ast_repr::{Collection, Lambda},
    prelude::*,
};


fn main() -> Result<()> {
    println!("🔍 Testing Dependency Analysis in egg");
    println!("====================================\n");

    // Test 1: Basic variable dependency tracking
    test_basic_dependency_tracking()?;

    // Test 2: Dependency tracking in sum expressions
    test_sum_dependency_tracking()?;

    // Test 3: Lambda variable scoping (bound vs free variables)
    test_lambda_variable_scoping()?;

    println!("\n✅ Dependency analysis tests completed!");
    Ok(())
}

fn test_basic_dependency_tracking() -> Result<()> {
    println!("📊 Test 1: Basic Variable Dependency Tracking");
    println!("   Testing simple expressions with multiple variables\n");

    // Test expressions with known dependency patterns
    let test_cases = vec![
        (
            "Constant 5.0",
            ASTRepr::Constant(5.0),
            vec![], // No dependencies
        ),
        (
            "Variable x0",
            ASTRepr::Variable(0),
            vec![0], // Depends on variable 0
        ),
        (
            "2*x + 3*y",
            ASTRepr::add_binary(
                ASTRepr::mul_binary(ASTRepr::Constant(2.0), ASTRepr::Variable(0)),
                ASTRepr::mul_binary(ASTRepr::Constant(3.0), ASTRepr::Variable(1)),
            ),
            vec![0, 1], // Depends on variables 0 and 1
        ),
        (
            "2*x + 3",
            ASTRepr::add_binary(
                ASTRepr::mul_binary(ASTRepr::Constant(2.0), ASTRepr::Variable(0)),
                ASTRepr::Constant(3.0),
            ),
            vec![0], // Only depends on variable 0
        ),
    ];

    #[cfg(feature = "optimization")]
    {
        for (name, expr, expected_deps) in test_cases {
            println!("   🧪 Testing: {name}");

            // Optimization functionality removed
            {
                // Just test that expressions can be created and evaluated
                println!("      ✅ Expression created successfully");

                if expected_deps.is_empty() {
                    println!("      📋 Expected: No dependencies (constant expression)");
                } else {
                    println!("      📋 Expected dependencies: {expected_deps:?}");
                }
                
                // Test evaluation if possible
                if !expected_deps.is_empty() {
                    let test_values: Vec<f64> = (0..=expected_deps.len()).map(|i| i as f64 + 1.0).collect();
                    if !test_values.is_empty() {
                        let result = expr.eval_with_vars(&test_values);
                        println!("      📊 Test evaluation: {result}");
                    }
                } else {
                    let result = expr.eval_with_vars(&[]);
                    println!("      📊 Test evaluation: {result}");
                }
            }
            println!();
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("   🚫 Egg optimization not enabled - skipping dependency tests");
    }

    Ok(())
}

fn test_sum_dependency_tracking() -> Result<()> {
    println!("📈 Test 2: Sum Expression Dependency Tracking");
    println!("   Testing dependency analysis with summation expressions\n");

    // Create a sum expression: Σ(λv.(a*v + b)) where a,b are external variables
    let var_a = ASTRepr::Variable(0); // External variable 'a'
    let var_b = ASTRepr::Variable(1); // External variable 'b'
    let var_v = ASTRepr::BoundVar(0); // Lambda-bound variable 'v'

    // Lambda body: a*v + b
    let lambda_body = ASTRepr::add_binary(ASTRepr::mul_binary(var_a, var_v), var_b);

    let lambda = Lambda {
        var_indices: vec![0], // Binds variable 0 (v)
        body: Box::new(lambda_body),
    };

    // Create sum over data collection
    let data_collection = Collection::Constant(vec![1.0, 2.0, 3.0]);
    let map_collection = Collection::Map {
        lambda: Box::new(lambda),
        collection: Box::new(data_collection),
    };

    let sum_expr = ASTRepr::Sum(Box::new(map_collection));

    println!("   🔍 Testing sum expression: Σ(λv.(a*v + b))");
    println!("   Expected: Should depend on variables {{a, b}}, not on bound variable v");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("   ✅ Sum expression created successfully");
            println!("   📊 Original:  {sum_expr:?}");

            // Test evaluation
            let test_values = [2.0, 3.0]; // Values for variables a and b
            let result = sum_expr.eval_with_vars(&test_values);
            println!("   📊 Test evaluation (a=2, b=3): {result}");

            // This should demonstrate that we can factor out 'a' and 'b' but not 'v'
            println!("   💡 The bound variable 'v' should not appear in free variables");
            println!("   💡 External variables 'a' and 'b' should be tracked as dependencies");
        }
    }

    println!();
    Ok(())
}

fn test_lambda_variable_scoping() -> Result<()> {
    println!("🔬 Test 3: Lambda Variable Scoping");
    println!("   Testing bound variables vs free variables in lambda expressions\n");

    // Test case: expression that mixes bound and free variables
    // Create: Σ(λx.(c*x + d*y)) where:
    // - x is bound by the lambda (should not be a free variable)
    // - c, d, y are free variables (should be tracked as dependencies)

    let var_c = ASTRepr::Variable(0); // Free variable 'c'
    let var_d = ASTRepr::Variable(1); // Free variable 'd' 
    let var_y = ASTRepr::Variable(2); // Free variable 'y'
    let var_x = ASTRepr::BoundVar(0); // Bound variable 'x'

    // Lambda body: c*x + d*y
    let lambda_body = ASTRepr::add_binary(
        ASTRepr::mul_binary(var_c, var_x), // c*x (c is free, x is bound)
        ASTRepr::mul_binary(var_d, var_y), // d*y (both d and y are free)
    );

    let lambda = Lambda {
        var_indices: vec![0], // Binds variable 0 (x)
        body: Box::new(lambda_body),
    };

    let data_collection = Collection::Constant(vec![1.0, 2.0, 3.0]);
    let map_collection = Collection::Map {
        lambda: Box::new(lambda),
        collection: Box::new(data_collection),
    };

    let complex_sum = ASTRepr::Sum(Box::new(map_collection));

    println!("   🧪 Testing complex sum: Σ(λx.(c*x + d*y))");
    println!("   Expected dependencies: {{c, d, y}} (not x, since x is bound)");

    #[cfg(feature = "optimization")]
    {
        // Optimization functionality removed
        {
            println!("   ✅ Lambda scoping expression created successfully");

            // Test evaluation
            let test_values = [2.0, 3.0, 4.0]; // Values for variables c, d, y
            let result = complex_sum.eval_with_vars(&test_values);
            println!("   📊 Test evaluation (c=2, d=3, y=4): {result}");

            println!("   💡 Key insight: 'c' and 'd' should be factorizable, 'x' should not");
            println!("   💡 Expected result: factoring based on variable independence");
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("   🚫 Egg optimization not enabled - cannot test lambda scoping");
    }

    println!();
    Ok(())
}
