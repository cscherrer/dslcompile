use mathcompile_macros::optimize_compile_time;

#[test]
fn debug_macro_simple() {
    let x = 1.0;

    // Try the simplest possible case
    let result = optimize_compile_time!(constant(1.0), []);

    assert_eq!(result, 1.0);
    println!("Simple constant works: {result}");
}

#[test]
fn debug_macro_var() {
    let x = 2.0;

    // Try a simple variable
    let result = optimize_compile_time!(var::<0>(), [x]);

    assert_eq!(result, x);
    println!("Simple variable works: {result}");
}
