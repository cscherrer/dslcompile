use dslcompile_macros::optimize_compile_time;

#[test]
fn test_safe_egglog_works() {
    let x = 2.0;

    // Test that the macro compiles and runs without infinite expansion
    let result = optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);

    // Should optimize x + 0 to x
    assert_eq!(result, x);
    println!("✅ Safe egglog macro works! x + 0 = {result}");
}
