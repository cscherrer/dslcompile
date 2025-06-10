use dslcompile::ast::advanced::ast_from_expr;
use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CSE Rules Debug Test ===\n");

    let mut ctx: DynamicContext<f64, 0> = DynamicContext::new();
    let x = ctx.var();

    // Convert to ASTRepr for optimization
    let x_ast = ast_from_expr(&x).clone();

    // Test 1: Simple case - x * x (should trigger CSE Rule 1)
    println!("--- Test 1: Simple x * x ---");
    let x_squared = ASTRepr::Mul(Box::new(x_ast.clone()), Box::new(x_ast.clone()));
    println!("Original: {:?}", x_squared);

    // Check what egglog representation looks like
    let mut optimizer1 = NativeEgglogOptimizer::new()?;
    let egglog_repr = optimizer1.ast_to_egglog(&x_squared)?;
    println!("Egglog representation: {}", egglog_repr);

    let optimized1 = optimizer1.optimize(&x_squared)?;
    println!("Optimized: {:?}", optimized1);
    println!("Changed: {}", x_squared != optimized1);

    // Test 2: Even simpler - make sure ANY rule fires
    println!("\n--- Test 2: Basic Constant Folding (should work) ---");
    let simple_math = ASTRepr::Add(
        Box::new(ASTRepr::Constant(2.0)),
        Box::new(ASTRepr::Constant(3.0)),
    );
    println!("Original: {:?}", simple_math);

    let mut optimizer2 = NativeEgglogOptimizer::new()?;
    let egglog_repr2 = optimizer2.ast_to_egglog(&simple_math)?;
    println!("Egglog representation: {}", egglog_repr2);

    let optimized2 = optimizer2.optimize(&simple_math)?;
    println!("Optimized: {:?}", optimized2);
    println!("Changed: {}", simple_math != optimized2);

    // Test 3: Check if the issue is with variable indexing
    println!("\n--- Test 3: Different Variable Pattern ---");
    let different_pattern =
        ASTRepr::Mul(Box::new(ASTRepr::UserVar(0)), Box::new(ASTRepr::UserVar(0)));
    println!("Original: {:?}", different_pattern);

    let mut optimizer3 = NativeEgglogOptimizer::new()?;
    let egglog_repr3 = optimizer3.ast_to_egglog(&different_pattern)?;
    println!("Egglog representation: {}", egglog_repr3);

    let optimized3 = optimizer3.optimize(&different_pattern)?;
    println!("Optimized: {:?}", optimized3);
    println!("Changed: {}", different_pattern != optimized3);

    // Test 4: Manual verification of the exact CSE pattern
    println!("\n--- Test 4: Checking CSE Rule Pattern ---");
    println!("CSE Rule 1 expects: (Mul ?expr ?expr)");
    println!("We're providing: (Mul (UserVar 0) (UserVar 0))");
    println!("Should match: ?expr = (UserVar 0)");

    // Test 5: Test with the exact pattern from the ANF test
    println!("\n--- Test 5: Complex Gaussian Pattern from ANF Test ---");
    let exact_pattern = ASTRepr::Mul(
        Box::new(ASTRepr::Div(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::UserVar(0)),
                Box::new(ASTRepr::Neg(Box::new(ASTRepr::UserVar(1)))),
            )),
            Box::new(ASTRepr::UserVar(2)),
        )),
        Box::new(ASTRepr::Div(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::UserVar(0)),
                Box::new(ASTRepr::Neg(Box::new(ASTRepr::UserVar(1)))),
            )),
            Box::new(ASTRepr::UserVar(2)),
        )),
    );
    println!("Original: Complex Gaussian pattern");

    let mut optimizer5 = NativeEgglogOptimizer::new()?;
    let egglog_repr5 = optimizer5.ast_to_egglog(&exact_pattern)?;
    println!("Egglog representation: {}", egglog_repr5);

    let optimized5 = optimizer5.optimize(&exact_pattern)?;
    println!("Changed: {}", exact_pattern != optimized5);

    println!("\n=== Diagnosis ===");
    println!("1. If constant folding works, basic rules are firing");
    println!("2. If CSE rules aren't firing, there might be a pattern matching issue");
    println!("3. The egglog representations show what patterns we're actually generating");

    Ok(())
}

/// Check if an expression contains Let bindings (indicating CSE occurred)
fn contains_let_binding(expr: &ASTRepr<f64>) -> bool {
    match expr {
        // ASTRepr doesn't seem to have a Let variant currently
        // CSE would show up as transformed structure
        ASTRepr::Add(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => contains_let_binding(left) || contains_let_binding(right),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => contains_let_binding(inner),
        ASTRepr::Sum(collection) => {
            // Check the collection for Let bindings
            false // Simplified for now
        }
        _ => false,
    }
}
