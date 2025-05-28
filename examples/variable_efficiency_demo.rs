use mathjit::final_tagless::{ASTRepr, DirectEval, ASTEval, ASTMathExpr};

fn main() {
    println!("=== Variable Indexing Efficiency Demo ===\n");

    // Test 1: Using efficient indexed variables (Vector lookup)
    println!("1. Efficient Variable Indexing (Vector lookup):");
    let efficient_expr = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)), // x (index 0)
            Box::new(ASTRepr::Constant(2.0)),
        )),
        Box::new(ASTRepr::Variable(1)), // y (index 1)
    );
    
    let variables = [3.0, 4.0]; // x=3, y=4
    let result1 = DirectEval::eval_with_vars(&efficient_expr, &variables);
    println!("   Expression: 2*x + y where x=3, y=4");
    println!("   Result: {}", result1);
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result1, 10.0);
    println!("   ✓ Correct!");

    // Test 2: Using backwards compatible named variables (String lookup)
    println!("\n2. Named Variables (String lookup, backwards compatible):");
    let named_expr = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::VariableByName("x".to_string())),
            Box::new(ASTRepr::Constant(2.0)),
        )),
        Box::new(ASTRepr::VariableByName("y".to_string())),
    );
    
    let result2 = DirectEval::eval_with_vars(&named_expr, &variables);
    println!("   Expression: 2*x + y where x=3, y=4");
    println!("   Result: {}", result2);
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result2, 10.0);
    println!("   ✓ Correct!");

    // Test 3: Using the convenient API methods
    println!("\n3. Using ASTEval convenient methods:");
    let api_expr = ASTEval::add(
        ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)), // Efficient indexed
        ASTEval::var_by_name("y"), // Named for flexibility
    );
    
    let result3 = DirectEval::eval_with_vars(&api_expr, &variables);
    println!("   Mixed: ASTEval::var(0) * 2 + ASTEval::var_by_name(\"y\")");
    println!("   Result: {}", result3);
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result3, 10.0);
    println!("   ✓ Correct!");

    // Test 4: Backwards compatibility with traditional API
    println!("\n4. Traditional API (uses VariableByName internally):");
    let traditional_expr = <ASTEval as ASTMathExpr>::add(
        <ASTEval as ASTMathExpr>::mul(
            <ASTEval as ASTMathExpr>::var("x"),
            <ASTEval as ASTMathExpr>::constant(2.0),
        ),
        <ASTEval as ASTMathExpr>::var("y"),
    );
    
    let result4 = DirectEval::eval_two_vars(&traditional_expr, 3.0, 4.0);
    println!("   Traditional: 2*x + y where x=3, y=4");
    println!("   Result: {}", result4);
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result4, 10.0);
    println!("   ✓ Correct!");

    // Test 5: Performance comparison example with more variables
    println!("\n5. Multi-variable expression:");
    let multi_var_expr = ASTEval::add(
        ASTEval::add(
            ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)), // 2*x
            ASTEval::mul(ASTEval::var(1), ASTEval::constant(3.0)), // 3*y
        ),
        ASTEval::add(
            ASTEval::mul(ASTEval::var(2), ASTEval::constant(4.0)), // 4*z
            ASTEval::var(3), // w
        ),
    );
    
    let multi_vars = [1.0, 2.0, 3.0, 4.0]; // x=1, y=2, z=3, w=4
    let result5 = DirectEval::eval_with_vars(&multi_var_expr, &multi_vars);
    println!("   Expression: 2*x + 3*y + 4*z + w where x=1, y=2, z=3, w=4");
    println!("   Result: {}", result5);
    println!("   Expected: 2*1 + 3*2 + 4*3 + 4 = 2 + 6 + 12 + 4 = 24");
    assert_eq!(result5, 24.0);
    println!("   ✓ Correct!");

    println!("\n=== Summary ===");
    println!("✓ Variable(usize) - Most efficient, uses vector indexing O(1)");
    println!("✓ VariableByName(String) - Less efficient, but backwards compatible");
    println!("✓ Both types work seamlessly with DirectEval::eval_with_vars");
    println!("✓ Traditional API still works without changes");
    println!("✓ Can mix indexed and named variables in the same expression");
    println!("\n🚀 DirectEval can now use vector lookups for optimal performance!");
    println!("📈 For performance-critical code, use Variable(index) instead of VariableByName(name)");
} 