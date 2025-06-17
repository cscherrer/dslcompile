use dslcompile::ast::ASTRepr;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Minimal Egglog Constant Propagation Demo ===");

    // Test expressions for constant propagation
    let test_cases = vec![
        // Basic constant folding
        (
            ASTRepr::Add(vec![
                ASTRepr::Constant(2.0),
                ASTRepr::Constant(3.0),
            ]),
            "2.0 + 3.0 should become 5.0",
        ),
        (
            ASTRepr::Mul(vec![
                ASTRepr::Constant(4.0),
                ASTRepr::Constant(5.0),
            ]),
            "4.0 * 5.0 should become 20.0",
        ),
        // Identity rules
        (
            ASTRepr::Add(vec![
                ASTRepr::Variable(0),
                ASTRepr::Constant(0.0),
            ]),
            "x + 0.0 should become x",
        ),
        (
            ASTRepr::Mul(vec![
                ASTRepr::Variable(0),
                ASTRepr::Constant(1.0),
            ]),
            "x * 1.0 should become x",
        ),
        // Zero elimination
        (
            ASTRepr::Mul(vec![
                ASTRepr::Variable(0),
                ASTRepr::Constant(0.0),
            ]),
            "x * 0.0 should become 0.0",
        ),
        // NEW: Mixed constant/variable expression
        (
            ASTRepr::Add(vec![
                ASTRepr::Add(vec![
                    ASTRepr::Constant(2.0),
                    ASTRepr::Variable(0),
                ]),
                ASTRepr::Constant(3.0),
            ]),
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
                        (_, ASTRepr::Add(operands), _)
                            if description.contains("2.0 + x + 3.0") =>
                        {
                            match &operands[..] {
                                [ASTRepr::Variable(0), ASTRepr::Constant(c)] if *c == 5.0 => {
                                    println!("✓ Optimization worked correctly: x + 5.0")
                                }
                                [ASTRepr::Constant(c), ASTRepr::Variable(0)] if *c == 5.0 => {
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

    // Create simple mathematical expressions for testing
    let x = ASTRepr::Add(vec![
        ASTRepr::Constant(2.0),
        ASTRepr::Constant(3.0),
    ]);

    println!("Simple addition: 2 + 3");
    println!("Expression: {x:?}");

    let y = ASTRepr::Mul(vec![
        ASTRepr::Variable(0),
        ASTRepr::Constant(5.0),
    ]);

    println!("Variable multiplication: x * 5");
    println!("Expression: {y:?}");

    // More complex expression
    let complex = ASTRepr::Add(vec![
        ASTRepr::Mul(vec![
            ASTRepr::Constant(2.0),
            ASTRepr::Variable(0),
        ]),
        ASTRepr::Constant(1.0),
    ]);

    println!("Complex: 2*x + 1");
    println!("Expression: {complex:?}");

    // Test Mul with three factors
    let three_factor = ASTRepr::Mul(vec![
        ASTRepr::Variable(0),
        ASTRepr::Variable(1),
        ASTRepr::Constant(3.0),
    ]);

    println!("Three factors: x * y * 3");
    println!("Expression: {three_factor:?}");

    // Test nested Add
    let nested_add = ASTRepr::Add(vec![
        ASTRepr::Variable(0),
        ASTRepr::Variable(1),
        ASTRepr::Constant(2.0),
        ASTRepr::Constant(3.0),
    ]);

    println!("Nested addition: x + y + 2 + 3");
    println!("Expression: {nested_add:?}");

    // Test power operation (still binary)
    let power_expr = ASTRepr::Pow(
        Box::new(ASTRepr::Variable(0)),
        Box::new(ASTRepr::Constant(2.0)),
    );

    println!("Power: x^2");
    println!("Expression: {power_expr:?}");

    println!("\n=== Demo completed ===");
    Ok(())
}
