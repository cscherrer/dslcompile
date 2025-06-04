use dslcompile::prelude::*;

#[test]
fn test_compile_time_variable_collision() {
    // This test demonstrates that the compile-time system has the same issue
    // but it's handled differently due to the type system

    use dslcompile::compile_time::*;

    // Define f(x) = 2x using compile-time variable 0
    let f = var::<0>().mul(constant(2.0));

    // Define g(y) = 3y using compile-time variable 0 (collision!)
    let g = var::<0>().mul(constant(3.0));

    // If we naively combine: h = f + g = 2*var[0] + 3*var[0] = 5*var[0]
    let h_wrong = f.add(g);

    // This gives us h(4) = 5*4 = 20, not f(4) + g(7) = 8 + 21 = 29
    let result_wrong = h_wrong.eval(&[4.0]);
    assert_eq!(result_wrong, 20.0);

    // The correct way: use different variable IDs
    let f_correct = var::<0>().mul(constant(2.0)); // f(x) uses var 0
    let g_correct = var::<1>().mul(constant(3.0)); // g(y) uses var 1
    let h_correct = f_correct.add(g_correct);

    // Now h(4,7) = f(4) + g(7) = 2*4 + 3*7 = 8 + 21 = 29
    let result_correct = h_correct.eval(&[4.0, 7.0]);
    assert_eq!(result_correct, 29.0);

    println!("Compile-time collision: {result_wrong} vs correct: {result_correct}");
}
