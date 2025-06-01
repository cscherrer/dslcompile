use mathcompile::compile_time::optimize_compile_time;

fn main() {
    println!("=== Procedural Macro Compile-Time Optimization Demo ===\n");

    // Test 1: Simple optimization - ln(exp(x)) should become x
    println!("Test 1: ln(exp(x)) optimization");
    let x = 2.5_f64;

    // This should generate direct code: x
    let result1 = optimize_compile_time!(var::<0>().exp().ln(), [x]);
    let manual1 = x; // What it should optimize to

    println!("  Input: ln(exp(x)) where x = {x}");
    println!("  Optimized result: {result1}");
    println!("  Manual result: {manual1}");
    println!("  Match: {}\n", (result1 - manual1).abs() < 1e-10);

    // Test 2: Identity optimization - x + 0 should become x
    println!("Test 2: x + 0 optimization");
    let y = 2.71_f64;

    // This should generate direct code: y
    let result2 = optimize_compile_time!(var::<0>().add(constant(0.0)), [y]);
    let manual2 = y;

    println!("  Input: x + 0 where x = {y}");
    println!("  Optimized result: {result2}");
    println!("  Manual result: {manual2}");
    println!("  Match: {}\n", (result2 - manual2).abs() < 1e-10);

    // Test 3: Multiplication identity - x * 1 should become x
    println!("Test 3: x * 1 optimization");
    let z = 1.618_f64;

    // This should generate direct code: z
    let result3 = optimize_compile_time!(var::<0>().mul(constant(1.0)), [z]);
    let manual3 = z;

    println!("  Input: x * 1 where x = {z}");
    println!("  Optimized result: {result3}");
    println!("  Manual result: {manual3}");
    println!("  Match: {}\n", (result3 - manual3).abs() < 1e-10);

    // Test 4: Complex optimization - exp(ln(x) + ln(y)) should become x * y
    println!("Test 4: exp(ln(x) + ln(y)) optimization");
    let a = 2.0_f64;
    let b = 3.0_f64;

    // This should generate direct code: a * b
    let result4 = optimize_compile_time!(var::<0>().ln().add(var::<1>().ln()).exp(), [a, b]);
    let manual4 = a * b;

    println!("  Input: exp(ln(x) + ln(y)) where x = {a}, y = {b}");
    println!("  Optimized result: {result4}");
    println!("  Manual result: {manual4}");
    println!("  Match: {}\n", (result4 - manual4).abs() < 1e-10);

    // Test 5: Mixed operations - ln(exp(x)) + y * 1 + 0 * z should become x + y
    println!("Test 5: Complex mixed optimization");
    let x5 = 1.5_f64;
    let y5 = 2.5_f64;
    let z5 = 999.0_f64; // This should be eliminated

    // This should generate direct code: x5 + y5
    let result5 = optimize_compile_time!(
        var::<0>()
            .exp()
            .ln()
            .add(var::<1>().mul(constant(1.0)))
            .add(constant(0.0).mul(var::<2>())),
        [x5, y5, z5]
    );
    let manual5 = x5 + y5;

    println!("  Input: ln(exp(x)) + y * 1 + 0 * z where x = {x5}, y = {y5}, z = {z5}");
    println!("  Optimized result: {result5}");
    println!("  Manual result: {manual5}");
    println!("  Match: {}\n", (result5 - manual5).abs() < 1e-10);

    println!("=== All tests completed! ===");

    // Performance note: The generated code should be equivalent to direct operations
    // like `x`, `y`, `a * b`, etc. with zero runtime overhead
}
