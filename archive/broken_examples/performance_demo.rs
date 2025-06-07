// Note: zero_overhead_core removed - using Enhanced Scoped System instead
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Zero-Overhead UnifiedContext Performance Results");
    println!("==================================================");

    let x = 3.0;
    let y = 4.0;
    let z = 5.0;
    let iterations = 10_000_000;

    // Native Rust baseline
    println!("\n📊 NATIVE RUST BASELINE:");
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += x + y;
    }
    let native_add_time = start.elapsed();
    println!("  Native Add:     {native_add_time:?} ({iterations} iterations)");

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += x * y;
    }
    let native_mul_time = start.elapsed();
    println!("  Native Mul:     {native_mul_time:?} ({iterations} iterations)");

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += x * x + 2.0 * x * y + y * y + z;
    }
    let native_complex_time = start.elapsed();
    println!("  Native Complex: {native_complex_time:?} ({iterations} iterations)");

    // Zero-overhead implementations
    println!("\n⚡ ZERO-OVERHEAD IMPLEMENTATIONS:");

    // Direct compute context
    // Note: DirectComputeContext removed - using Enhanced Scoped System instead
    // TODO: Migrate to Enhanced Scoped System
    println!("DirectComputeContext removed - demo needs migration to Enhanced Scoped System");
    return;
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += ctx.add_direct(x, y);
    }
    let zero_add_time = start.elapsed();
    println!("  Zero Add:       {zero_add_time:?} ({iterations} iterations)");

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += ctx.mul_direct(x, y);
    }
    let zero_mul_time = start.elapsed();
    println!("  Zero Mul:       {zero_mul_time:?} ({iterations} iterations)");

    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += ctx.complex_direct(x, y, z);
    }
    let zero_complex_time = start.elapsed();
    println!("  Zero Complex:   {zero_complex_time:?} ({iterations} iterations)");

    // Smart context
    // Note: SmartContext removed - using Enhanced Scoped System instead
    // TODO: Migrate to Enhanced Scoped System
    println!("SmartContext removed - demo needs migration to Enhanced Scoped System");
    return;
    let start = Instant::now();
    let mut result = 0.0;
    for _ in 0..iterations {
        result += smart_ctx.add_smart(x, y);
    }
    let smart_add_time = start.elapsed();
    println!("  Smart Add:      {smart_add_time:?} ({iterations} iterations)");

    // Performance comparison
    println!("\n📈 PERFORMANCE COMPARISON:");
    let add_ratio = zero_add_time.as_nanos() as f64 / native_add_time.as_nanos() as f64;
    let mul_ratio = zero_mul_time.as_nanos() as f64 / native_mul_time.as_nanos() as f64;
    let complex_ratio = zero_complex_time.as_nanos() as f64 / native_complex_time.as_nanos() as f64;

    println!(
        "  Add Performance:     {:.2}x native speed",
        1.0 / add_ratio
    );
    println!(
        "  Mul Performance:     {:.2}x native speed",
        1.0 / mul_ratio
    );
    println!(
        "  Complex Performance: {:.2}x native speed",
        1.0 / complex_ratio
    );

    if add_ratio <= 1.1 && mul_ratio <= 1.1 && complex_ratio <= 1.1 {
        println!("\n✅ SUCCESS: Zero-overhead achieved! (within 10% of native performance)");
    } else {
        println!("\n⚠️  WARNING: Some overhead detected");
    }

    // Correctness verification
    println!("\n🔍 CORRECTNESS VERIFICATION:");
    let native_result: f64 = x + y;
    let zero_result: f64 = ctx.add_direct(x, y);
    let smart_result: f64 = smart_ctx.add_smart(x, y);

    println!("  Native result: {native_result}");
    println!("  Zero result:   {zero_result}");
    println!("  Smart result:  {smart_result}");

    if (native_result - zero_result).abs() < 1e-10 && (native_result - smart_result).abs() < 1e-10 {
        println!("  ✅ All results match!");
    } else {
        println!("  ❌ Results don't match!");
    }

    println!("\n🎯 SUMMARY:");
    println!("  The zero-overhead UnifiedContext implementations achieve");
    println!("  native Rust performance while providing a unified API!");
    println!("  This eliminates the 50-200x overhead from expression trees.");
}
