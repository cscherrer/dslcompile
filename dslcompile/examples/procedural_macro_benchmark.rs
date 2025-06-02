use dslcompile::compile_time::optimize_compile_time;
use std::time::Instant;

fn main() {
    println!("=== Procedural Macro Performance Benchmark ===\n");

    let x = 2.5_f64;
    let y = 1.5_f64;
    let z = 3.0_f64;

    // Warm up
    for _ in 0..1000 {
        let _ = optimize_compile_time!(var::<0>().add(var::<1>()), [x, y]);
        let _ = x + y;
    }

    const ITERATIONS: usize = 1_000_000;

    // Benchmark 1: Simple addition - procedural macro
    let start = Instant::now();
    let mut sum1 = 0.0_f64;
    for _ in 0..ITERATIONS {
        sum1 += optimize_compile_time!(var::<0>().add(var::<1>()), [x, y]);
    }
    let proc_macro_time = start.elapsed();

    // Benchmark 1: Simple addition - manual
    let start = Instant::now();
    let mut sum2 = 0.0_f64;
    for _ in 0..ITERATIONS {
        sum2 += x + y;
    }
    let manual_time = start.elapsed();

    println!("Simple Addition (x + y):");
    println!(
        "  Procedural macro: {:?} ({:.2} ns/op)",
        proc_macro_time,
        proc_macro_time.as_nanos() as f64 / ITERATIONS as f64
    );
    println!(
        "  Manual code:      {:?} ({:.2} ns/op)",
        manual_time,
        manual_time.as_nanos() as f64 / ITERATIONS as f64
    );
    println!(
        "  Overhead:         {:.2}x",
        proc_macro_time.as_nanos() as f64 / manual_time.as_nanos() as f64
    );
    println!("  Results match:    {}", (sum1 - sum2).abs() < 1e-10);
    println!();

    // Benchmark 2: Identity optimization - procedural macro
    let start = Instant::now();
    let mut sum3 = 0.0_f64;
    for _ in 0..ITERATIONS {
        sum3 += optimize_compile_time!(var::<0>().add(constant(0.0)), [x]);
    }
    let proc_macro_opt_time = start.elapsed();

    // Benchmark 2: Identity optimization - manual
    let start = Instant::now();
    let mut sum4 = 0.0_f64;
    for _ in 0..ITERATIONS {
        sum4 += x; // x + 0 optimized to x
    }
    let manual_opt_time = start.elapsed();

    println!("Identity Optimization (x + 0 → x):");
    println!(
        "  Procedural macro: {:?} ({:.2} ns/op)",
        proc_macro_opt_time,
        proc_macro_opt_time.as_nanos() as f64 / ITERATIONS as f64
    );
    println!(
        "  Manual code:      {:?} ({:.2} ns/op)",
        manual_opt_time,
        manual_opt_time.as_nanos() as f64 / ITERATIONS as f64
    );
    println!(
        "  Overhead:         {:.2}x",
        proc_macro_opt_time.as_nanos() as f64 / manual_opt_time.as_nanos() as f64
    );
    println!("  Results match:    {}", (sum3 - sum4).abs() < 1e-10);
    println!();

    // Benchmark 3: Complex optimization - procedural macro
    let start = Instant::now();
    let mut sum5 = 0.0_f64;
    for _ in 0..ITERATIONS {
        sum5 += optimize_compile_time!(
            var::<0>()
                .exp()
                .ln()
                .add(var::<1>().mul(constant(1.0)))
                .add(constant(0.0).mul(var::<2>())),
            [x, y, z]
        );
    }
    let proc_macro_complex_time = start.elapsed();

    // Benchmark 3: Complex optimization - manual
    let start = Instant::now();
    let mut sum6 = 0.0_f64;
    for _ in 0..ITERATIONS {
        sum6 += x + y; // ln(exp(x)) + y * 1 + 0 * z optimized to x + y
    }
    let manual_complex_time = start.elapsed();

    println!("Complex Optimization (ln(exp(x)) + y * 1 + 0 * z → x + y):");
    println!(
        "  Procedural macro: {:?} ({:.2} ns/op)",
        proc_macro_complex_time,
        proc_macro_complex_time.as_nanos() as f64 / ITERATIONS as f64
    );
    println!(
        "  Manual code:      {:?} ({:.2} ns/op)",
        manual_complex_time,
        manual_complex_time.as_nanos() as f64 / ITERATIONS as f64
    );
    println!(
        "  Overhead:         {:.2}x",
        proc_macro_complex_time.as_nanos() as f64 / manual_complex_time.as_nanos() as f64
    );
    println!("  Results match:    {}", (sum5 - sum6).abs() < 1e-10);
    println!();

    println!("=== Summary ===");
    println!("The procedural macro generates direct Rust code at compile time,");
    println!("achieving performance very close to hand-written code while providing");
    println!("automatic mathematical optimization via egglog equality saturation.");
    println!();
    println!("Key benefits:");
    println!("- Zero runtime dispatch (no enums, no function pointers)");
    println!("- Complete mathematical reasoning during compilation");
    println!("- Direct code generation: optimize_compile_time!(expr) → optimized Rust code");
    println!(
        "- Performance within {}x of manual code",
        (proc_macro_time.as_nanos() as f64 / manual_time.as_nanos() as f64)
            .max(proc_macro_opt_time.as_nanos() as f64 / manual_opt_time.as_nanos() as f64)
            .max(proc_macro_complex_time.as_nanos() as f64 / manual_complex_time.as_nanos() as f64)
    );
}
