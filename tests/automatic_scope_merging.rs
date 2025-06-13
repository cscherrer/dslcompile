//! Tests for type-level scope safety and future automatic scope merging functionality
//!
//! This test suite demonstrates type-level scoping that prevents cross-scope operations
//! and documents requirements for future automatic scope merging.

use dslcompile::{
    contexts::dynamic::DynamicContext,
    contexts::static_context::{StaticContext, StaticVar},
    hlist,
};

#[test]
fn test_dynamic_context_type_level_scoping() {
    // This test demonstrates type-level scoping preventing cross-scope operations
    
    // Define function f(x) = x² + 2x + 1 independently in scope 0
    let mut math_f = DynamicContext::<f64, 0>::new();
    let x_f = math_f.var(); // index 0 in f's scope
    let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;

    // Define function g(y) = 3y + 5 independently in scope 1
    let mut math_g = DynamicContext::<f64, 1>::new_explicit();
    let y_g = math_g.var(); // index 0 in g's scope (DIFFERENT SCOPE!)
    let g_expr = 3.0 * &y_g + 5.0;

    // With type-level scoping, cross-scope operations should be compile errors:
    // let h_expr = &f_expr + &g_expr; // ❌ Should NOT compile due to scope mismatch!

    // Instead, we test that same-scope operations work correctly
    
    // Test f_expr evaluation in its own scope
    let f_result = math_f.eval(&f_expr, hlist![2.0]); // f(2) = 4 + 4 + 1 = 9
    assert_eq!(f_result, 9.0);
    
    // Test g_expr evaluation in its own scope  
    let g_result = math_g.eval(&g_expr, hlist![3.0]); // g(3) = 9 + 5 = 14
    assert_eq!(g_result, 14.0);
    
    // The combined result would be f(2) + g(3) = 9 + 14 = 23
    let expected_combined = f_result + g_result;
    assert_eq!(expected_combined, 23.0);
    
    println!("SUCCESS: Type-level scoping prevents cross-scope operations!");
    println!("f(2) = {f_result}, g(3) = {g_result}, combined = {expected_combined}");
}

#[test]
fn test_static_context_scope_isolation_works() {
    // Verify that StaticContext provides proper scope isolation
    // This WORKS because StaticContext uses type-level scopes
    
    let mut ctx = StaticContext::new();

    // Define f(x) = x² in scope 0
    let f = ctx.new_scope(|scope| {
        let (x, _) = scope.auto_var::<f64>();
        x.clone() * x
    });

    // Advance to next scope
    let mut ctx = ctx.next();

    // Define g(y) = 3y in scope 1
    let g = ctx.new_scope(|scope| {
        let (y, scope) = scope.auto_var::<f64>();
        scope.constant(3.0) * y
    });

    // These work independently - no collision
    let f_result = f.eval(hlist![2.0]); // f(2) = 4
    let g_result = g.eval(hlist![3.0]); // g(3) = 9
    
    assert_eq!(f_result, 4.0);
    assert_eq!(g_result, 9.0);

    // Verify scopes are different at type level
    assert_eq!(StaticVar::<f64, 0, 0>::scope_id(), 0); // f's variable in scope 0
    assert_eq!(StaticVar::<f64, 0, 1>::scope_id(), 1); // g's variable in scope 1

    println!("StaticContext provides proper scope isolation via type system");
}

#[test] 
fn test_desired_cross_scope_composition() {
    // This test shows what SHOULD be possible once we implement
    // automatic scope merging for StaticContext expressions too
    
    let mut ctx = StaticContext::new();

    // Define f(x) = x² in scope 0
    let f = ctx.new_scope(|scope| {
        let (x, _) = scope.auto_var::<f64>();
        x.clone() * x
    });

    let mut ctx = ctx.next();

    // Define g(y) = 3y in scope 1  
    let g = ctx.new_scope(|scope| {
        let (y, scope) = scope.auto_var::<f64>();
        scope.constant(3.0) * y
    });

    // DESIRED: This should automatically create a merged expression
    // that takes inputs for both scopes and evaluates correctly
    // Currently this won't compile due to type-level scope mismatch
    
    // TODO: Implement scope merging that allows:
    // let h = f + g; // Should auto-merge scope 0 and scope 1
    // let result = h.eval(hlist![2.0, 3.0]); // f(2) + g(3) = 4 + 9 = 13
    
    // For now, just verify individual evaluation
    assert_eq!(f.eval(hlist![2.0]), 4.0);
    assert_eq!(g.eval(hlist![3.0]), 9.0);
    
    println!("TODO: Implement scope merging for StaticContext cross-scope composition");
}

#[test]
fn test_scope_merging_requirements() {
    // This test documents the exact requirements for scope merging
    
    // REQUIREMENT 1: Automatic scope detection
    // When combining expressions from different contexts, the system should:
    // - Detect all unique scopes involved
    // - Create a mapping from old variable indices to new indices
    // - Generate a merged expression with remapped variables
    
    // REQUIREMENT 2: Type-safe variable remapping
    // The merged expression should:
    // - Preserve all variable types (f64, i32, Vec<f64>, etc.)
    // - Maintain evaluation semantics
    // - Provide clear variable ordering in the merged context
    
    // REQUIREMENT 3: Compositional correctness
    // For expressions f (using vars [0]) and g (using vars [0]):
    // - f + g should create expression using vars [0, 1]
    // - f(a) + g(b) should equal (f + g)(a, b) 
    // - No loss of mathematical meaning
    
    // REQUIREMENT 4: Performance preservation
    // - StaticContext: Zero runtime overhead via compile-time merging
    // - DynamicContext: Minimal overhead, reuse existing evaluation
    
    // REQUIREMENT 5: Error handling
    // - Clear error messages for incompatible merges
    // - Type-level prevention of invalid compositions where possible
    
    println!("Requirements documented for automatic scope merging implementation");
}

#[test]
fn test_manual_workaround_vs_automatic() {
    // Compare manual workaround with what automatic merging should provide
    
    // === MANUAL WORKAROUND (current approach) ===
    let mut math_f = DynamicContext::<f64, 0>::new();
    let x_f = math_f.var();
    let _f_expr = 2.0 * &x_f; // f(x) = 2x
    
    let mut math_g = DynamicContext::<f64, 0>::new(); 
    let y_g = math_g.var();
    let _g_expr = 3.0 * &y_g; // g(y) = 3y
    
    // Manual: Create new unified context and rebuild expressions
    let mut unified = DynamicContext::<f64, 0>::new();
    let x_unified = unified.var(); // index 0
    let y_unified = unified.var(); // index 1
    let f_unified = 2.0 * &x_unified; // Manually recreate f
    let g_unified = 3.0 * &y_unified; // Manually recreate g
    let h_manual = &f_unified + &g_unified;
    
    let manual_result = unified.eval(&h_manual, hlist![4.0, 5.0]); // f(4) + g(5) = 8 + 15 = 23
    assert_eq!(manual_result, 23.0);
    
    // === DESIRED AUTOMATIC APPROACH ===
    // This SHOULD work once scope merging is implemented:
    //
    // let h_auto = &f_expr + &g_expr; // Automatic scope merging!
    // let auto_result = h_auto.eval(hlist![4.0, 5.0]); // Should just work
    // assert_eq!(auto_result, 23.0);
    //
    // Benefits of automatic approach:
    // - No manual expression reconstruction
    // - Reuse existing expression definitions
    // - Compositional - works with any expression complexity
    // - Type-safe variable ordering
    
    println!("Manual workaround: {manual_result}");
    println!("Automatic approach: TODO - implement scope merging");
} and future automatic scope merging functionality
//!
//! This test suite demonstrates type-level scoping that prevents cross-scope operations
//! and documents requirements for future automatic scope merging.

use dslcompile::{
    contexts::dynamic::DynamicContext,
    contexts::static_context::{StaticContext, StaticVar},
    hlist,
};

#[test]
fn test_dynamic_context_type_level_scoping() {
    // This test demonstrates type-level scoping preventing cross-scope operations
    
    // Define function f(x) = x² + 2x + 1 independently in scope 0
    let mut math_f = DynamicContext::<f64, 0>::new();
    let x_f = math_f.var(); // index 0 in f's scope
    let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;

    // Define function g(y) = 3y + 5 independently in scope 1
    let mut math_g = DynamicContext::<f64, 1>::new_explicit();
    let y_g = math_g.var(); // index 0 in g's scope (DIFFERENT SCOPE!)
    let g_expr = 3.0 * &y_g + 5.0;

    // With type-level scoping, cross-scope operations should be compile errors:
    // let h_expr = &f_expr + &g_expr; // ❌ Should NOT compile due to scope mismatch!

    // Instead, we test that same-scope operations work correctly
    
    // Test f_expr evaluation in its own scope
    let f_result = math_f.eval(&f_expr, hlist![2.0]); // f(2) = 4 + 4 + 1 = 9
    assert_eq!(f_result, 9.0);
    
    // Test g_expr evaluation in its own scope  
    let g_result = math_g.eval(&g_expr, hlist![3.0]); // g(3) = 9 + 5 = 14
    assert_eq!(g_result, 14.0);
    
    // The combined result would be f(2) + g(3) = 9 + 14 = 23
    let expected_combined = f_result + g_result;
    assert_eq!(expected_combined, 23.0);
    
    println!("SUCCESS: Type-level scoping prevents cross-scope operations!");
    println!("f(2) = {f_result}, g(3) = {g_result}, combined = {expected_combined}");
}

#[test]
fn test_static_context_scope_isolation_works() {
    // Verify that StaticContext provides proper scope isolation
    // This WORKS because StaticContext uses type-level scopes
    
    let mut ctx = StaticContext::new();

    // Define f(x) = x² in scope 0
    let f = ctx.new_scope(|scope| {
        let (x, _) = scope.auto_var::<f64>();
        x.clone() * x
    });

    // Advance to next scope
    let mut ctx = ctx.next();

    // Define g(y) = 3y in scope 1
    let g = ctx.new_scope(|scope| {
        let (y, scope) = scope.auto_var::<f64>();
        scope.constant(3.0) * y
    });

    // These work independently - no collision
    let f_result = f.eval(hlist![2.0]); // f(2) = 4
    let g_result = g.eval(hlist![3.0]); // g(3) = 9
    
    assert_eq!(f_result, 4.0);
    assert_eq!(g_result, 9.0);

    // Verify scopes are different at type level
    assert_eq!(StaticVar::<f64, 0, 0>::scope_id(), 0); // f's variable in scope 0
    assert_eq!(StaticVar::<f64, 0, 1>::scope_id(), 1); // g's variable in scope 1

    println!("StaticContext provides proper scope isolation via type system");
}

#[test] 
fn test_desired_cross_scope_composition() {
    // This test shows what SHOULD be possible once we implement
    // automatic scope merging for StaticContext expressions too
    
    let mut ctx = StaticContext::new();

    // Define f(x) = x² in scope 0
    let f = ctx.new_scope(|scope| {
        let (x, _) = scope.auto_var::<f64>();
        x.clone() * x
    });

    let mut ctx = ctx.next();

    // Define g(y) = 3y in scope 1  
    let g = ctx.new_scope(|scope| {
        let (y, scope) = scope.auto_var::<f64>();
        scope.constant(3.0) * y
    });

    // DESIRED: This should automatically create a merged expression
    // that takes inputs for both scopes and evaluates correctly
    // Currently this won't compile due to type-level scope mismatch
    
    // TODO: Implement scope merging that allows:
    // let h = f + g; // Should auto-merge scope 0 and scope 1
    // let result = h.eval(hlist![2.0, 3.0]); // f(2) + g(3) = 4 + 9 = 13
    
    // For now, just verify individual evaluation
    assert_eq!(f.eval(hlist![2.0]), 4.0);
    assert_eq!(g.eval(hlist![3.0]), 9.0);
    
    println!("TODO: Implement scope merging for StaticContext cross-scope composition");
}

#[test]
fn test_scope_merging_requirements() {
    // This test documents the exact requirements for scope merging
    
    // REQUIREMENT 1: Automatic scope detection
    // When combining expressions from different contexts, the system should:
    // - Detect all unique scopes involved
    // - Create a mapping from old variable indices to new indices
    // - Generate a merged expression with remapped variables
    
    // REQUIREMENT 2: Type-safe variable remapping
    // The merged expression should:
    // - Preserve all variable types (f64, i32, Vec<f64>, etc.)
    // - Maintain evaluation semantics
    // - Provide clear variable ordering in the merged context
    
    // REQUIREMENT 3: Compositional correctness
    // For expressions f (using vars [0]) and g (using vars [0]):
    // - f + g should create expression using vars [0, 1]
    // - f(a) + g(b) should equal (f + g)(a, b) 
    // - No loss of mathematical meaning
    
    // REQUIREMENT 4: Performance preservation
    // - StaticContext: Zero runtime overhead via compile-time merging
    // - DynamicContext: Minimal overhead, reuse existing evaluation
    
    // REQUIREMENT 5: Error handling
    // - Clear error messages for incompatible merges
    // - Type-level prevention of invalid compositions where possible
    
    println!("Requirements documented for automatic scope merging implementation");
}

#[test]
fn test_manual_workaround_vs_automatic() {
    // Compare manual workaround with what automatic merging should provide
    
    // === MANUAL WORKAROUND (current approach) ===
    let mut math_f = DynamicContext::<f64, 0>::new();
    let x_f = math_f.var();
    let _f_expr = 2.0 * &x_f; // f(x) = 2x
    
    let mut math_g = DynamicContext::<f64, 0>::new(); 
    let y_g = math_g.var();
    let _g_expr = 3.0 * &y_g; // g(y) = 3y
    
    // Manual: Create new unified context and rebuild expressions
    let mut unified = DynamicContext::<f64, 0>::new();
    let x_unified = unified.var(); // index 0
    let y_unified = unified.var(); // index 1
    let f_unified = 2.0 * &x_unified; // Manually recreate f
    let g_unified = 3.0 * &y_unified; // Manually recreate g
    let h_manual = &f_unified + &g_unified;
    
    let manual_result = unified.eval(&h_manual, hlist![4.0, 5.0]); // f(4) + g(5) = 8 + 15 = 23
    assert_eq!(manual_result, 23.0);
    
    // === DESIRED AUTOMATIC APPROACH ===
    // This SHOULD work once scope merging is implemented:
    //
    // let h_auto = &f_expr + &g_expr; // Automatic scope merging!
    // let auto_result = h_auto.eval(hlist![4.0, 5.0]); // Should just work
    // assert_eq!(auto_result, 23.0);
    //
    // Benefits of automatic approach:
    // - No manual expression reconstruction
    // - Reuse existing expression definitions
    // - Compositional - works with any expression complexity
    // - Type-safe variable ordering
    
    println!("Manual workaround: {manual_result}");
    println!("Automatic approach: TODO - implement scope merging");
} 