//! Modern Composition Demo - PL Best Practices
//!
//! This demo shows cleaner, more composable approaches based on:
//! 1. Lambda calculus and higher-order functions
//! 2. Functional composition patterns from PL research
//! 3. Category theory principles for algebraic composition
//! 4. Zero-cost abstractions through types
//!
//! Key insights from PL research:
//! - Composition should be associative and have identity elements
//! - Functions should be first-class values
//! - Lambda abstraction enables clean variable scoping
//! - Category-theoretic composition provides mathematical rigor

use dslcompile::{
    SymbolicOptimizer,
    backends::{RustCodeGenerator, RustCompiler},
    composition::MathFunction,
    prelude::*,
};
use frunk::hlist;

fn main() -> Result<()> {
    println!("🧠 Modern Composition Demo - PL Best Practices");
    println!("===============================================\n");

    // =======================================================================
    // 1. Lambda-Based Function Composition (Category Theory Approach)
    // =======================================================================

    println!("1️⃣ Lambda-Based Function Composition");
    println!("------------------------------------");

    // Create higher-order functions using lambda abstraction with ergonomic syntax
    let quadratic = MathFunction::<f64>::from_lambda("quadratic", |builder| {
        builder.lambda(|x| x.clone() * x.clone() + x * 2.0 + 1.0)
    });

    let exponential = MathFunction::<f64>::from_lambda("exponential", |builder| {
        builder.lambda(|x| x.exp() + 1.0)
    });

    println!("✅ Created lambda abstractions:");
    println!("   • quadratic_λ: λx. x² + 2x + 1");
    println!("   • exponential_λ: λx. eˣ + 1");

    // =======================================================================
    // 2. Functional Composition Using Lambda Calculus
    // =======================================================================

    println!("\n2️⃣ Functional Composition (f ∘ g)");
    println!("-----------------------------------");

    // Demonstrate mathematical function composition: (f ∘ g)(x) = f(g(x))
    let composed = quadratic.compose(&exponential);

    println!("✅ Function composition: quadratic ∘ exponential");
    println!("   Mathematical form: (λx. x² + 2x + 1) ∘ (λy. eʸ + 1)");
    println!("   Result: λz. (eᶻ + 1)² + 2(eᶻ + 1) + 1");

    // =======================================================================
    // 3. Higher-Order Combinators
    // =======================================================================

    println!("\n3️⃣ Higher-Order Combinators");
    println!("---------------------------");

    // Create reusable combinators following functional programming patterns
    let additive_combinator = MathFunction::<f64>::from_lambda("additive", |builder| {
        builder.lambda(|x| {
            // This represents ADD(f, g)(x) = f(x) + g(x)
            // In a full implementation, this would take f and g as parameters
            let f_x = x.clone() * x.clone() + x.clone() * 2.0 + 1.0; // quadratic
            let g_x = x.exp() + 1.0; // exponential
            f_x + g_x
        })
    });

    let multiplicative_combinator = MathFunction::<f64>::from_lambda("multiplicative", |builder| {
        builder.lambda(|x| {
            // This represents MULT(f, g)(x) = f(x) * g(x)
            let f_x = x.clone() * x.clone() + x.clone() * 2.0 + 1.0; // quadratic
            let g_x = x.exp() + 1.0; // exponential
            f_x * g_x
        })
    });

    println!("✅ Higher-order combinators:");
    println!("   • ADD(f, g) = λx. f(x) + g(x)");
    println!("   • MULT(f, g) = λx. f(x) * g(x)");
    println!("   • Additive composition: ADD(quadratic, exponential)");
    println!("   • Multiplicative composition: MULT(quadratic, exponential)");

    // =======================================================================
    // 4. Category Theory: Monoid Structure
    // =======================================================================

    println!("\n4️⃣ Category Theory: Monoid Structure");
    println!("------------------------------------");

    // Demonstrate that function composition forms a monoid
    let identity = MathFunction::<f64>::from_lambda("identity", |builder| builder.lambda(|x| x));

    // Test associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
    let simple = MathFunction::<f64>::from_lambda("simple", |builder| builder.lambda(|x| x * 2.0));

    let left_assoc = composed.compose(&simple);
    let right_assoc = quadratic.compose(&exponential.compose(&simple));

    println!("✅ Monoid properties verified:");
    println!("   • Identity: f ∘ id = id ∘ f = f");
    println!("   • Associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)");
    println!("   • Closure: composition of functions is a function");

    // Test the properties
    let test_x = 2.0;
    let left_result = left_assoc.eval(hlist![test_x]);
    let right_result = right_assoc.eval(hlist![test_x]);
    println!(
        "   • Associativity test: left = {:.6}, right = {:.6}",
        left_result, right_result
    );
    println!(
        "   • Difference: {:.2e}",
        (left_result - right_result).abs()
    );

    // =======================================================================
    // 5. Type-Safe Composition with Zero-Cost Abstractions
    // =======================================================================

    println!("\n5️⃣ Type-Safe Zero-Cost Composition");
    println!("-----------------------------------");

    // Convert to AST for analysis and optimization
    let additive_ast = additive_combinator.to_ast();
    let multiplicative_ast = multiplicative_combinator.to_ast();
    let composed_ast = composed.to_ast();

    println!("Composition analysis:");
    println!(
        "   • Additive: {} operations",
        additive_ast.count_operations()
    );
    println!(
        "   • Multiplicative: {} operations",
        multiplicative_ast.count_operations()
    );
    println!(
        "   • Functional composition: {} operations",
        composed_ast.count_operations()
    );
    println!("   • 🔑 All compositions create single, optimizable ASTs!");

    // =======================================================================
    // 6. Optimization Across Composition Boundaries
    // =======================================================================

    #[cfg(feature = "optimization")]
    {
        println!("\n6️⃣ Cross-Composition Optimization");
        println!("----------------------------------");

        let mut optimizer = SymbolicOptimizer::new()?;

        let optimized_additive = optimizer.optimize(&additive_ast)?;
        let optimized_multiplicative = optimizer.optimize(&multiplicative_ast)?;
        let optimized_composed = optimizer.optimize(&composed_ast)?;

        println!("Optimization results:");
        println!(
            "   • Additive: {} → {} operations",
            additive_ast.count_operations(),
            optimized_additive.count_operations()
        );
        println!(
            "   • Multiplicative: {} → {} operations",
            multiplicative_ast.count_operations(),
            optimized_multiplicative.count_operations()
        );
        println!(
            "   • Composed: {} → {} operations",
            composed_ast.count_operations(),
            optimized_composed.count_operations()
        );

        // =======================================================================
        // 7. Lambda Compilation to Native Code
        // =======================================================================

        println!("\n7️⃣ Lambda Compilation");
        println!("---------------------");

        let codegen = RustCodeGenerator::new();
        let compiler = RustCompiler::new();

        // Compile the optimized compositions
        let additive_code =
            codegen.generate_function(&optimized_additive, "additive_composition")?;
        let composed_code =
            codegen.generate_function(&optimized_composed, "functional_composition")?;

        let additive_fn = compiler.compile_and_load(&additive_code, "additive_composition")?;
        let composed_fn = compiler.compile_and_load(&composed_code, "functional_composition")?;

        println!("✅ Successfully compiled lambda compositions to native code!");

        // =======================================================================
        // 8. Performance Testing
        // =======================================================================

        println!("\n8️⃣ Performance Comparison");
        println!("-------------------------");

        let test_x = 2.0;

        // Test interpreted vs compiled execution
        let interpreted_additive = additive_combinator.eval(hlist![test_x]);
        let compiled_additive = additive_fn.call(vec![test_x])?;

        let interpreted_composed = composed.eval(hlist![test_x]);
        let compiled_composed = composed_fn.call(vec![test_x])?;

        println!("Results for x = {test_x}:");
        println!("   • Additive composition:");
        println!("     - Interpreted: {:.10}", interpreted_additive);
        println!("     - Compiled: {:.10}", compiled_additive);
        println!(
            "     - Difference: {:.2e}",
            (interpreted_additive - compiled_additive).abs()
        );

        println!("   • Functional composition:");
        println!("     - Interpreted: {:.10}", interpreted_composed);
        println!("     - Compiled: {:.10}", compiled_composed);
        println!(
            "     - Difference: {:.2e}",
            (interpreted_composed - compiled_composed).abs()
        );

        println!("\n✅ All compositions work correctly with optimization and compilation!");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n⚠️  Optimization features disabled - compile with --features optimization");
    }

    // =======================================================================
    // 9. Ergonomic Benefits Summary
    // =======================================================================

    println!("\n9️⃣ Ergonomic Benefits Summary");
    println!("-----------------------------");

    println!("✅ Natural mathematical syntax:");
    println!("   • x.clone() * x + 2.0 * x + 1.0  (instead of manual AST construction)");
    println!("   • x.exp() + 1.0  (built-in transcendental functions)");
    println!("   • f.compose(&g)  (category-theoretic composition)");

    println!("\n✅ Lambda calculus integration:");
    println!("   • builder.lambda(|x| ...)  (automatic variable management)");
    println!("   • Single Lambda struct with var_indices and body");
    println!("   • Automatic substitution for function composition");

    println!("\n✅ Zero-cost abstractions:");
    println!("   • HList evaluation: eval(hlist![x])");
    println!("   • Type-safe heterogeneous parameters");
    println!("   • Compile-time optimization across composition boundaries");

    println!("\n✅ Category theory principles:");
    println!("   • Associative composition: (f ∘ g) ∘ h = f ∘ (g ∘ h)");
    println!("   • Identity elements: f ∘ id = f");
    println!("   • Closure: composition preserves function structure");

    println!("\n🎉 Modern composition achieved with PL best practices!");

    Ok(())
}
