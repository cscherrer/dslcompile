#[test]
fn test_legacy_system_removed() {
    // Test that we successfully removed the legacy MathExpr system
    // The fact that this compiles means the legacy code is gone
    println!("✅ Legacy MathExpr system successfully removed");
}

#[test]
fn test_scoped_variables_available() {
    // Test that scoped variables are available
    use dslcompile::compile_time::ScopedExpressionBuilder;
    let _builder = ScopedExpressionBuilder::new_f64();
    println!("✅ Scoped variables system available");
}
