use mathcompile::prelude::*;

#[test]
fn test_independent_math_builders() {
    // Test 1: Independent MathBuilder instances
    let math1 = MathBuilder::new();
    let x1 = math1.var(); // This will be variable index 0 in math1's registry
    let expr1 = &x1 * 2.0; // 2x

    let math2 = MathBuilder::new();
    let x2 = math2.var(); // This will be variable index 0 in math2's registry  
    let expr2 = &x2 + 1.0; // x + 1

    // Evaluate each expression with their own builder
    let result1 = math1.eval(&expr1, &[3.0]);
    let result2 = math2.eval(&expr2, &[3.0]);

    assert_eq!(result1, 6.0); // 2 * 3 = 6
    assert_eq!(result2, 4.0); // 3 + 1 = 4
}

#[test]
fn test_combining_expressions_from_different_builders() {
    let math1 = MathBuilder::new();
    let x1 = math1.var(); // index 0 in math1's registry
    let expr1 = &x1 * 2.0; // 2x

    let math2 = MathBuilder::new();
    let x2 = math2.var(); // index 0 in math2's registry  
    let expr2 = &x2 + 1.0; // x + 1

    // This works because both expressions use the same variable index (0)
    // but they have different registries
    let combined_expr = &expr1 + &expr2; // (2x) + (x + 1) = 3x + 1

    // When we evaluate, we need to be careful about which registry we use
    // Using math1's registry (which only knows about one variable)
    let result = math1.eval(&combined_expr, &[3.0]);
    assert_eq!(result, 10.0); // 2*3 + (3+1) = 6 + 4 = 10
}

#[test]
fn test_multiple_variables_in_each_builder() {
    let math_a = MathBuilder::new();
    let x_a = math_a.var(); // index 0
    let y_a = math_a.var(); // index 1
    let expr_a = &x_a + &y_a; // x + y

    let math_b = MathBuilder::new();
    let x_b = math_b.var(); // index 0 (different registry!)
    let y_b = math_b.var(); // index 1 (different registry!)
    let expr_b = &x_b * &y_b; // x * y

    let result_a = math_a.eval(&expr_a, &[2.0, 3.0]);
    let result_b = math_b.eval(&expr_b, &[2.0, 3.0]);

    assert_eq!(result_a, 5.0); // 2 + 3 = 5
    assert_eq!(result_b, 6.0); // 2 * 3 = 6
}

#[test]
fn test_functions_building_expressions_independently() {
    fn build_quadratic(math: &MathBuilder) -> TypedBuilderExpr<f64> {
        let x = math.var();
        &x * &x + 2.0 * &x + 1.0 // x² + 2x + 1
    }

    fn build_linear(math: &MathBuilder) -> TypedBuilderExpr<f64> {
        let x = math.var();
        3.0 * &x + 2.0 // 3x + 2
    }

    let main_math = MathBuilder::new();

    // Both functions will create their own variables within the same registry
    let quad_expr = build_quadratic(&main_math);
    let linear_expr = build_linear(&main_math);

    // Since both use the same MathBuilder, they share the same registry
    // The quadratic uses variable 0, the linear uses variable 1
    let quad_result = main_math.eval(&quad_expr, &[3.0, 0.0]); // x=3, second var unused
    let linear_result = main_math.eval(&linear_expr, &[0.0, 3.0]); // first var unused, x=3

    assert_eq!(quad_result, 16.0); // 9 + 6 + 1 = 16
    assert_eq!(linear_result, 11.0); // 3*3 + 2 = 11
}

#[test]
fn test_combining_expressions_from_same_builder() {
    fn build_quadratic(math: &MathBuilder) -> TypedBuilderExpr<f64> {
        let x = math.var();
        &x * &x + 2.0 * &x + 1.0 // x² + 2x + 1
    }

    fn build_linear(math: &MathBuilder) -> TypedBuilderExpr<f64> {
        let x = math.var();
        3.0 * &x + 2.0 // 3x + 2
    }

    let main_math = MathBuilder::new();
    let quad_expr = build_quadratic(&main_math);
    let linear_expr = build_linear(&main_math);

    let combined_same_builder = &quad_expr + &linear_expr;
    // This combines variables 0 and 1, so we need values for both
    let combined_result = main_math.eval(&combined_same_builder, &[3.0, 3.0]);
    assert_eq!(combined_result, 27.0); // 16 + 11 = 27
}

#[test]
fn test_variable_indices_are_not_mangled() {
    // Test that variables use simple integer indices, not mangled names
    let math = MathBuilder::new();
    let x = math.var(); // Should be index 0
    let y = math.var(); // Should be index 1
    let z = math.var(); // Should be index 2

    // Create expressions that use these variables
    let expr1 = &x * 2.0; // Uses variable 0
    let expr2 = &y * 3.0; // Uses variable 1  
    let expr3 = &z * 4.0; // Uses variable 2
    let combined = &expr1 + &expr2 + &expr3; // 2x + 3y + 4z

    // Evaluate with specific values for each variable
    let result = math.eval(&combined, &[1.0, 2.0, 3.0]); // x=1, y=2, z=3
    assert_eq!(result, 20.0); // 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
}

#[test]
fn test_registry_isolation() {
    // Test that different MathBuilder instances have isolated registries
    let math1 = MathBuilder::new();
    let math2 = MathBuilder::new();

    // Create variables in each builder
    let x1 = math1.var(); // index 0 in math1
    let y1 = math1.var(); // index 1 in math1

    let x2 = math2.var(); // index 0 in math2 (independent!)
    let y2 = math2.var(); // index 1 in math2 (independent!)

    // Create expressions
    let expr1 = &x1 + &y1; // Uses indices 0,1 in math1's registry
    let expr2 = &x2 * &y2; // Uses indices 0,1 in math2's registry

    // Each can be evaluated independently with their own variable values
    let result1 = math1.eval(&expr1, &[10.0, 20.0]); // x1=10, y1=20
    let result2 = math2.eval(&expr2, &[5.0, 6.0]); // x2=5, y2=6

    assert_eq!(result1, 30.0); // 10 + 20 = 30
    assert_eq!(result2, 30.0); // 5 * 6 = 30

    // The registries are completely independent
    assert_eq!(math1.registry().borrow().len(), 2);
    assert_eq!(math2.registry().borrow().len(), 2);
}
