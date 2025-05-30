use mathcompile::final_tagless::{
    ASTEval, ASTMathExpr, ASTRepr, DirectEval, ExpressionBuilder, clear_global_registry,
    register_variable,
};

fn main() {
    println!("=== Variable Indexing Efficiency Demo ===\n");

    // Clear any existing variables for clean demo
    clear_global_registry();

    // Test 1: Using efficient indexed variables (Vector lookup)
    println!("1. Efficient Variable Indexing (Vector lookup):");
    let efficient_expr = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)), // Use index 0 for variable x
            Box::new(ASTRepr::Constant(2.0)),
        )),
        Box::new(ASTRepr::Variable(1)), // Use index 1 for variable y
    );

    let variables = [3.0, 4.0]; // x=3, y=4
    let result1 = DirectEval::eval_with_vars(&efficient_expr, &variables);
    println!("   Expression: 2*x + y where x=3, y=4");
    println!("   Result: {result1}");
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result1, 10.0);
    println!("   Correct");

    // Test 2: Using named variables with the new registry system
    println!("\n2. Named Variables (using global registry):");

    // Register variables to get their indices
    let x_index = register_variable("x");
    let y_index = register_variable("y");

    let named_expr = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Variable(x_index)),
            Box::new(ASTRepr::Constant(2.0)),
        )),
        Box::new(ASTRepr::Variable(y_index)),
    );

    let result2 = DirectEval::eval_with_vars(&named_expr, &variables);
    println!("   Expression: 2*x + y where x=3, y=4");
    println!("   Result: {result2}");
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result2, 10.0);
    println!("   Correct");

    // Test 3: Using the API methods with named variables
    println!("\n3. Using ASTEval methods:");
    let api_expr = ASTEval::add(
        ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)), // Efficient indexed
        ASTEval::var_by_name("y"),                             // Named for flexibility
    );

    let result3 = DirectEval::eval_with_vars(&api_expr, &variables);
    println!("   Mixed: ASTEval::var(0) * 2 + ASTEval::var_by_name(\"y\")");
    println!("   Result: {result3}");
    println!("   Expected: 2*3 + 4 = 10");
    assert_eq!(result3, 10.0);
    println!("   Correct");

    // Test 4: Named variable evaluation
    println!("\n4. Named Variable Evaluation:");
    let mut builder = ExpressionBuilder::new();
    let api_expr = ASTRepr::Add(Box::new(builder.var("x")), Box::new(builder.var("y")));
    let named_vars = vec![("x".to_string(), 3.0), ("y".to_string(), 4.0)];
    let result4 = builder.eval_with_named_vars(&api_expr, &named_vars);
    println!("   Result: {result4}");

    // Test 5: Performance comparison example with more variables
    println!("\n5. Multi-variable expression:");
    let multi_var_expr = ASTEval::add(
        ASTEval::add(
            ASTEval::mul(ASTEval::var(0), ASTEval::constant(2.0)), // 2*x
            ASTEval::mul(ASTEval::var(1), ASTEval::constant(3.0)), // 3*y
        ),
        ASTEval::add(
            ASTEval::mul(ASTEval::var(2), ASTEval::constant(4.0)), // 4*z
            ASTEval::var(3),                                       // w
        ),
    );

    let multi_vars = [1.0, 2.0, 3.0, 4.0]; // x=1, y=2, z=3, w=4
    let result5 = DirectEval::eval_with_vars(&multi_var_expr, &multi_vars);
    println!("   Expression: 2*x + 3*y + 4*z + w where x=1, y=2, z=3, w=4");
    println!("   Result: {result5}");
    println!("   Expected: 2*1 + 3*2 + 4*3 + 4 = 2 + 6 + 12 + 4 = 24");
    assert_eq!(result5, 24.0);
    println!("   Correct");

    println!("\n=== Summary ===");
    println!("- Variable(usize) - Most efficient, uses vector indexing O(1)");
    println!("- Named variables via global registry - User-friendly with good performance");
    println!("- Both types work seamlessly with DirectEval::eval_with_vars");
    println!("- eval_with_named_vars_f64 provides named variable evaluation");
    println!("- Can mix indexed and named variables in the same expression");
    println!("\nDirectEval can use vector lookups for optimal performance");
    println!("For performance-critical code, use Variable(index) instead of named variables");
}
