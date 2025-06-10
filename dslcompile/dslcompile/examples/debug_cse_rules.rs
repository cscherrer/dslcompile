use dslcompile::ast::DynamicContext;
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CSE Rules Debug Test ===\n");

    // Create a simple test case: x * x (should trigger CSE Rule 1)
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let x_squared = &x * &x;
    
    println!("Original expression: {}", x_squared.pretty_print());
    println!("AST structure: {:?}", x_squared);
    
    // Test 1: Regular optimization pipeline
    println!("\n--- Test 1: Regular Optimization Pipeline ---");
    let mut optimizer = NativeEgglogOptimizer::new()?;
    let optimized_regular = optimizer.optimize(&x_squared)?;
    println!("After regular optimization: {}", optimized_regular.pretty_print());
    println!("AST structure: {:?}", optimized_regular);
    
    // Test 2: ANF-CSE optimization pipeline
    println!("\n--- Test 2: ANF-CSE Optimization Pipeline ---");
    let mut optimizer2 = NativeEgglogOptimizer::new()?;
    let optimized_anf_cse = optimizer2.optimize_with_anf_cse(&x_squared)?;
    println!("After ANF-CSE optimization: {}", optimized_anf_cse.pretty_print());
    println!("AST structure: {:?}", optimized_anf_cse);
    
    // Test 3: More complex Gaussian pattern: ((x - mu) / sigma)^2
    println!("\n--- Test 3: Gaussian Pattern ---");
    let mu = ctx.var();
    let sigma = ctx.var();
    let standardized = (&x - &mu) / &sigma;
    let gaussian_pattern = &standardized * &standardized;
    
    println!("Original Gaussian pattern: {}", gaussian_pattern.pretty_print());
    
    let mut optimizer3 = NativeEgglogOptimizer::new()?;
    let optimized_gaussian = optimizer3.optimize(&gaussian_pattern)?;
    println!("After optimization: {}", optimized_gaussian.pretty_print());
    
    // Test 4: Compare structures to see if CSE happened
    println!("\n--- Test 4: CSE Detection ---");
    let cse_detected_simple = contains_let_binding(&optimized_regular);
    let cse_detected_anf = contains_let_binding(&optimized_anf_cse);
    let cse_detected_gaussian = contains_let_binding(&optimized_gaussian);
    
    println!("CSE detected in regular optimization: {}", cse_detected_simple);
    println!("CSE detected in ANF-CSE optimization: {}", cse_detected_anf);
    println!("CSE detected in Gaussian optimization: {}", cse_detected_gaussian);
    
    // Test 5: Manual CSE rule check
    println!("\n--- Test 5: Manual Egglog Check ---");
    test_egglog_cse_directly()?;
    
    Ok(())
}

/// Check if an expression contains Let bindings (indicating CSE occurred)
fn contains_let_binding(expr: &dslcompile::ast::ASTRepr<f64>) -> bool {
    use dslcompile::ast::ASTRepr;
    
    match expr {
        ASTRepr::Let(_, _, _) => true,
        ASTRepr::Add(left, right) | 
        ASTRepr::Mul(left, right) | 
        ASTRepr::Div(left, right) | 
        ASTRepr::Pow(left, right) => {
            contains_let_binding(left) || contains_let_binding(right)
        }
        ASTRepr::Neg(inner) | 
        ASTRepr::Abs(inner) |
        ASTRepr::Ln(inner) |
        ASTRepr::Exp(inner) |
        ASTRepr::Sin(inner) |
        ASTRepr::Cos(inner) |
        ASTRepr::Sqrt(inner) => contains_let_binding(inner),
        ASTRepr::Sum { body, .. } => contains_let_binding(body),
        _ => false,
    }
}

/// Test CSE rules directly in egglog to see raw behavior
fn test_egglog_cse_directly() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing CSE rules directly in egglog...");
    
    // Create a fresh egglog instance
    let mut optimizer = NativeEgglogOptimizer::new()?;
    
    // Convert x * x to egglog format manually
    let test_expr = "(Mul (UserVar 0) (UserVar 0))";
    println!("Input expression: {}", test_expr);
    
    // This is tricky - we need to call the internal egglog directly
    // For now, let's see what the optimizer produces
    println!("(Need to implement direct egglog access for deeper debugging)");
    
    Ok(())
} 