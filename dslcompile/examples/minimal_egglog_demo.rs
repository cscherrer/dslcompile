use dslcompile::ast::ASTRepr;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Minimal Egglog Constant Propagation Demo ===");

    // Test expressions for constant propagation
    let test_cases = vec![
        // Basic constant folding
        (
            ASTRepr::Add(
                Box::new(ASTRepr::Constant(2.0)),
                Box::new(ASTRepr::Constant(3.0)),
            ),
            "2.0 + 3.0 should become 5.0",
        ),
        (
            ASTRepr::Mul(
                Box::new(ASTRepr::Constant(4.0)),
                Box::new(ASTRepr::Constant(5.0)),
            ),
            "4.0 * 5.0 should become 20.0",
        ),
        // Identity rules
        (
            ASTRepr::Add(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(0.0)),
            ),
            "x + 0.0 should become x",
        ),
        (
            ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(1.0)),
            ),
            "x * 1.0 should become x",
        ),
        // Zero elimination
        (
            ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(0.0)),
            ),
            "x * 0.0 should become 0.0",
        ),
        // NEW: Mixed constant/variable expression
        (
            ASTRepr::Add(
                Box::new(ASTRepr::Add(
                    Box::new(ASTRepr::Constant(2.0)),
                    Box::new(ASTRepr::Variable(0)),
                )),
                Box::new(ASTRepr::Constant(3.0)),
            ),
            "2.0 + x + 3.0 should become x + 5.0 (through commutativity and constant folding)",
        ),
    ];

    // Test each case
    for (i, (expr, description)) in test_cases.iter().enumerate() {
        println!("\nTest {}: {}", i + 1, description);
        println!("Original: {expr:?}");

        #[cfg(feature = "optimization")]
        {
            let mut optimizer = NativeEgglogOptimizer::new()?;
            match optimizer.optimize(expr) {
                Ok(optimized) => {
                    println!("Optimized: {optimized:?}");

                    // Simple validation
                    match (expr, &optimized, description) {
                        (ASTRepr::Add(..), ASTRepr::Constant(5.0), _)
                            if description.contains("2.0 + 3.0") =>
                        {
                            println!("✓ Optimization worked correctly")
                        }
                        (ASTRepr::Mul(..), ASTRepr::Constant(20.0), _)
                            if description.contains("4.0 * 5.0") =>
                        {
                            println!("✓ Optimization worked correctly")
                        }
                        (ASTRepr::Add(..), ASTRepr::Variable(0), _)
                            if description.contains("x + 0.0") =>
                        {
                            println!("✓ Optimization worked correctly")
                        }
                        (ASTRepr::Mul(..), ASTRepr::Variable(0), _)
                            if description.contains("x * 1.0") =>
                        {
                            println!("✓ Optimization worked correctly")
                        }
                        (ASTRepr::Mul(..), ASTRepr::Constant(0.0), _)
                            if description.contains("x * 0.0") =>
                        {
                            println!("✓ Optimization worked correctly")
                        }
                        // NEW: Check for 2 + x + 3 → x + 5 optimization
                        (_, ASTRepr::Add(left, right), _)
                            if description.contains("2.0 + x + 3.0") =>
                        {
                            match (&**left, &**right) {
                                (ASTRepr::Variable(0), ASTRepr::Constant(5.0)) => {
                                    println!("✓ Optimization worked correctly: x + 5.0")
                                }
                                (ASTRepr::Constant(5.0), ASTRepr::Variable(0)) => {
                                    println!("✓ Optimization worked correctly: 5.0 + x")
                                }
                                _ => println!("⚠ Partial optimization: {optimized:?}"),
                            }
                        }
                        _ => println!("⚠ Unexpected optimization result"),
                    }
                }
                Err(e) => println!("Error: {e:?}"),
            }
        }

        #[cfg(not(feature = "optimization"))]
        {
            println!("Optimization disabled (missing 'optimization' feature)");
        }
    }

    println!("\n=== Basic tests completed ===");

    // NEW: Test summation optimizations
    println!("\n=== Testing summation optimizations ===");

    // Note: We can't create Sum variants yet because they're not in ASTRepr
    // But we can outline what we want to test:

    println!("TODO: Add Sum variant to ASTRepr to test:");
    println!("1. Sum splitting: Σ(f + g) = Σ(f) + Σ(g)");
    println!("2. Constant factor: Σ(k * f) = k * Σ(f)");
    println!("3. Sum of constant: Σ(k) over [1,3] = k * 3 = 3k");
    println!("4. Sum with zero body: Σ(0) = 0");

    println!("\n=== Demo completed ===");
    Ok(())
}
