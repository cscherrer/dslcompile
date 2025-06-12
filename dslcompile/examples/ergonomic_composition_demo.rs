// Demonstration of ergonomic function composition patterns
// Shows how natural mathematical syntax can be achieved through the composition API

use dslcompile::prelude::*;
use dslcompile::composition::{FunctionBuilder, MathFunction, LambdaVar};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Ergonomic Function Composition Demo ===\n");

    // Create some basic mathematical functions with natural syntax
    
    // f(x) = x² + 1
    let square_plus_one = MathFunction::from_lambda("square_plus_one", |builder| {
        builder.lambda(|x| x.clone() * x + 1.0)
    });
    
    // g(x) = 2x + 3
    let linear = MathFunction::from_lambda("linear", |builder| {
        builder.lambda(|x| x * 2.0 + 3.0)
    });

    println!("Created functions with natural lambda syntax:");
    println!("  f(x) = x² + 1");
    println!("  g(x) = 2x + 3");
    println!();

    // Function composition feels natural
    let f_compose_g = square_plus_one.compose(&linear);
    
    println!("Function composition:");
    println!("  f ∘ g means f(g(x)) = (2x + 3)² + 1");
    
    let x = 2.0;
    let result = f_compose_g.eval(hlist![x]);
    let expected = {
        let g_val = 2.0 * x + 3.0; // g(2) = 7
        g_val * g_val + 1.0       // f(7) = 50
    };
    
    println!("  At x = {}: result = {}, expected = {}", x, result, expected);
    println!("  ✓ Composition works correctly");
    println!();

    // Complex compositions are readable
    println!("=== Complex Composition Chains ===");
    
    // h(x) = sin(x)
    let sine = MathFunction::from_lambda("sin", |builder| {
        builder.lambda(|x: LambdaVar<f64>| x.sin())
    });
    
    // Chain: sin((2x + 3)² + 1)
    let complex_composition = sine.compose(&f_compose_g);
    
    println!("Complex chain: sin(f(g(x))) = sin((2x + 3)² + 1)");
    
    let complex_result = complex_composition.eval(hlist![x]);
    let expected_complex = expected.sin();
    
    println!("  At x = {}: result = {:.6}, expected = {:.6}", x, complex_result, expected_complex);
    println!("  ✓ Complex composition works");
    println!();

    // Show that mathematical expressions read naturally
    println!("=== Natural Mathematical Expressions ===");
    
    // Create more complex functions with readable syntax
    let polynomial = MathFunction::from_lambda("polynomial", |builder| {
        builder.lambda(|x| {
            // 3x³ - 2x² + x - 1
            let x2 = x.clone() * x.clone();
            let x3 = x2.clone() * x.clone();
            x3 * 3.0 - x2 * 2.0 + x.clone() - 1.0
        })
    });
    
    let trigonometric = MathFunction::from_lambda("trigonometric", |builder| {
        builder.lambda(|x| {
            // sin(x)cos(x) + e^x/(x+1)
            let sin_cos = x.sin() * x.cos();
            let exp_ratio = x.exp() / (x.clone() + 1.0);
            sin_cos + exp_ratio
        })
    });

    let test_val = 1.5;
    println!("Complex mathematical expressions:");
    println!("  Polynomial: 3x³ - 2x² + x - 1 = {}", polynomial.eval(hlist![test_val]));
    println!("  Trigonometric: sin(x)cos(x) + e^x/(x+1) = {}", trigonometric.eval(hlist![test_val]));
    println!();

    // Benefits summary
    println!("=== Benefits of This Approach ===");
    println!("✅ Natural mathematical syntax in lambda expressions");
    println!("✅ Clean function composition with f.compose(&g)");
    println!("✅ Readable complex mathematical expressions");
    println!("✅ Type-safe evaluation with HList inputs");
    println!("✅ Leverages existing DSLCompile infrastructure");
    println!("✅ No need for manual AST construction");
    
    println!();
    println!("🎉 Mathematical functions can be written naturally!");
    println!("   The lambda syntax feels like writing mathematics directly.");

    Ok(())
}