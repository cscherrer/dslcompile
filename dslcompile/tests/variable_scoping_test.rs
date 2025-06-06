use dslcompile::ast::{DynamicContext, TypedBuilderExpr};

#[test]
fn test_isolated_builders() {
    // Test 1: Independent DynamicContext instances
    let math1 = DynamicContext::new();
    let x1 = math1.var();
    let expr1 = 2.0 * &x1; // Should use variable index 0 in math1's registry

    let math2 = DynamicContext::new();
    let x2 = math2.var();
    let expr2 = 3.0 * &x2; // Should use variable index 0 in math2's registry

    // Each builder has its own registry, so x1 and x2 both get index 0
    let result1 = math1.eval(&expr1, &[5.0]);
    let result2 = math2.eval(&expr2, &[7.0]);

    assert_eq!(result1, 10.0); // 2 * 5
    assert_eq!(result2, 21.0); // 3 * 7
}

#[test]
fn test_shared_registry_confusion() {
    // Test 2: Mixing variables from different builders (this should work but might be confusing)
    let math1 = DynamicContext::new();
    let x = math1.var(); // This gets index 0 in math1's registry

    let math2 = DynamicContext::new();
    let y = math2.var(); // This gets index 0 in math2's registry

    // This creates an expression using x (from math1) but tries to evaluate it using math2
    // Since both variables have the same index (0), this will work but is semantically wrong
    let expr = 2.0 * &x + &y; // x is from math1, y is from math2

    // Note: This "works" because both variables have index 0, but it's conceptually incorrect
    // In the future, we might want to add builder-specific type safety to prevent this
    let result = math2.eval(&expr, &[10.0]); // Only providing one value
    assert_eq!(result, 30.0); // 2 * 10 + 10 (both variables use the same value)
}

#[test]
fn test_proper_shared_context() {
    // Test 3: Proper way to share variables - use the same builder
    let math_a = DynamicContext::new();
    let x = math_a.var(); // x gets index 0
    let y = math_a.var(); // y gets index 1

    let math_b = DynamicContext::new();
    let z = math_b.var(); // z gets index 0 (in different registry)

    // Create expressions using variables from the same builder
    let expr_a = 2.0 * &x + &y; // Both variables from math_a
    let expr_b = 3.0 * &z; // Variable from math_b

    let result_a = math_a.eval(&expr_a, &[5.0, 3.0]); // x=5, y=3
    let result_b = math_b.eval(&expr_b, &[4.0]); // z=4

    assert_eq!(result_a, 13.0); // 2*5 + 3 = 13
    assert_eq!(result_b, 12.0); // 3*4 = 12
}

#[test]
fn test_function_composition_isolation() {
    // Helper functions that create expressions in isolated contexts
    fn build_quadratic(math: &DynamicContext) -> TypedBuilderExpr<f64> {
        let x = math.var(); // Gets next available index in the provided builder
        &x * &x + 2.0 * &x + 1.0 // x² + 2x + 1
    }

    fn build_linear(math: &DynamicContext) -> TypedBuilderExpr<f64> {
        let x = math.var(); // Gets next available index in the provided builder
        3.0 * &x + 5.0 // 3x + 5
    }

    let main_math = DynamicContext::new();

    // Create both expressions using the same builder context
    let quad_expr = build_quadratic(&main_math); // Uses variable index 0
    let linear_expr = build_linear(&main_math); // Uses variable index 1

    // Since both use the same DynamicContext, they share the same registry
    // quad_expr uses index 0, linear_expr uses index 1
    let quad_result = main_math.eval(&quad_expr, &[2.0, 0.0]); // x=2, second var unused
    let linear_result = main_math.eval(&linear_expr, &[0.0, 4.0]); // first var unused, x=4

    assert_eq!(quad_result, 9.0); // 2² + 2*2 + 1 = 9
    assert_eq!(linear_result, 17.0); // 3*4 + 5 = 17
}

#[test]
fn test_function_composition_isolated() {
    // Helper functions that create expressions in isolated contexts
    fn build_quadratic(math: &DynamicContext) -> TypedBuilderExpr<f64> {
        let x = math.var(); // Gets next available index in the provided builder
        &x * &x + 2.0 * &x + 1.0 // x² + 2x + 1
    }

    fn build_linear(math: &DynamicContext) -> TypedBuilderExpr<f64> {
        let x = math.var(); // Gets next available index in the provided builder  
        3.0 * &x + 5.0 // 3x + 5
    }

    let main_math = DynamicContext::new();

    // Create both expressions using the same builder context
    let quad_expr = build_quadratic(&main_math); // Uses variable index 0
    let linear_expr = build_linear(&main_math); // Uses variable index 1

    // Create a composed expression using both
    let composed = &quad_expr + &linear_expr; // (x² + 2x + 1) + (3y + 5) where x is var 0, y is var 1

    // Evaluate with both variables
    let result = main_math.eval(&composed, &[2.0, 4.0]); // x=2, y=4
    assert_eq!(result, 26.0); // (4 + 4 + 1) + (12 + 5) = 9 + 17 = 26
}

#[test]
fn test_variable_ordering() {
    // Test that variables are consistently ordered
    let math = DynamicContext::new();

    let x = math.var(); // Should get index 0
    let y = math.var(); // Should get index 1
    let z = math.var(); // Should get index 2

    let expr = &x + 2.0 * &y + 3.0 * &z; // x + 2y + 3z

    // Values are provided in the order variables were created
    let result = math.eval(&expr, &[1.0, 2.0, 3.0]); // x=1, y=2, z=3
    assert_eq!(result, 14.0); // 1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
}

#[test]
fn test_registry_isolation() {
    // Test that different DynamicContext instances have isolated registries
    let math1 = DynamicContext::new();
    let math2 = DynamicContext::new();

    let x1 = math1.var(); // Index 0 in math1's registry
    let y1 = math1.var(); // Index 1 in math1's registry

    let x2 = math2.var(); // Index 0 in math2's registry (independent)
    let y2 = math2.var(); // Index 1 in math2's registry (independent)

    let expr1 = &x1 + &y1; // Uses indices 0 and 1 in math1's registry
    let expr2 = &x2 * &y2; // Uses indices 0 and 1 in math2's registry

    let result1 = math1.eval(&expr1, &[3.0, 4.0]); // x1=3, y1=4
    let result2 = math2.eval(&expr2, &[5.0, 6.0]); // x2=5, y2=6

    assert_eq!(result1, 7.0); // 3 + 4 = 7
    assert_eq!(result2, 30.0); // 5 * 6 = 30
}
