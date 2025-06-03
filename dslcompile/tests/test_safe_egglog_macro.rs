use dslcompile_macros::optimize_compile_time;

#[test]
fn test_safe_egglog_basic_optimization() {
    let x = 2.0;
    let _y = 3.0;

    // Test basic identity optimization: x + 0 should become x
    let result1 = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
    assert_eq!(result1, x);

    // Test multiplication identity: x * 1 should become x
    let result2 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [x]);
    assert_eq!(result2, x);

    // Test ln(exp(x)) = x optimization
    let result3 = optimize_compile_time!(var::<0>().exp().ln(), [x]);
    assert!((result3 - x).abs() < 1e-10);

    println!("✅ Safe egglog optimization working!");
    println!("   x + 0 = {result1}");
    println!("   x * 1 = {result2}");
    println!("   ln(exp(x)) = {result3}");
}

#[test]
fn test_safe_egglog_no_infinite_expansion() {
    let x = 1.0;
    let y = 2.0;

    // This should NOT cause infinite expansion (previously problematic)
    let result = optimize_compile_time!(var::<0>().add(var::<1>()), [x, y]);
    assert_eq!(result, x + y);

    println!("✅ No infinite expansion - safe termination!");
}
