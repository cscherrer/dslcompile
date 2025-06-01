#!/usr/bin/env cargo run --example overhead_analysis

//! Detailed Overhead Analysis
//!
//! This example breaks down exactly where the performance overhead comes from
//! in the trait-based compile-time expression system.

use mathcompile::compile_time::*;

fn main() {
    println!("🔬 Detailed Overhead Analysis");
    println!("=============================");
    println!("Breaking down the performance overhead step by step");
    println!();

    let iterations = 1_000_000;
    let x_val = 1.5;
    let y_val = 2.5;

    // ========================================================================
    // STEP 1: Pure Rust baseline
    // ========================================================================
    
    println!("📊 Step 1: Pure Rust Baseline");
    println!("-----------------------------");
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += x_val + y_val;
    }
    let rust_time = start.elapsed();
    
    println!("Pure Rust: x_val + y_val");
    println!("Time: {:?} ({:.2} ns/op)", rust_time, rust_time.as_nanos() as f64 / iterations as f64);
    println!("Result: {:.6}", sum / iterations as f64);
    println!();

    // ========================================================================
    // STEP 2: Array access overhead
    // ========================================================================
    
    println!("📊 Step 2: Array Access Overhead");
    println!("---------------------------------");
    
    let vars = [x_val, y_val];
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += vars[0] + vars[1];
    }
    let array_time = start.elapsed();
    
    println!("Array access: vars[0] + vars[1]");
    println!("Time: {:?} ({:.2} ns/op)", array_time, array_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", array_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // STEP 3: Bounds-checked array access
    // ========================================================================
    
    println!("📊 Step 3: Bounds-Checked Array Access");
    println!("--------------------------------------");
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += vars.get(0).copied().unwrap_or(0.0) + vars.get(1).copied().unwrap_or(0.0);
    }
    let bounds_check_time = start.elapsed();
    
    println!("Bounds-checked: vars.get(0).copied().unwrap_or(0.0) + vars.get(1).copied().unwrap_or(0.0)");
    println!("Time: {:?} ({:.2} ns/op)", bounds_check_time, bounds_check_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", bounds_check_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("Overhead vs Array: {:.1}x", bounds_check_time.as_nanos() as f64 / array_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // STEP 4: Function call overhead
    // ========================================================================
    
    println!("📊 Step 4: Function Call Overhead");
    println!("---------------------------------");
    
    fn var_access(vars: &[f64], id: usize) -> f64 {
        vars.get(id).copied().unwrap_or(0.0)
    }
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += var_access(&vars, 0) + var_access(&vars, 1);
    }
    let function_time = start.elapsed();
    
    println!("Function calls: var_access(&vars, 0) + var_access(&vars, 1)");
    println!("Time: {:?} ({:.2} ns/op)", function_time, function_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", function_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // STEP 5: Trait method call overhead
    // ========================================================================
    
    println!("📊 Step 5: Trait Method Call Overhead");
    println!("-------------------------------------");
    
    // Simulate the Var<ID> eval method
    struct VarSim<const ID: usize>;
    
    impl<const ID: usize> VarSim<ID> {
        fn eval(&self, vars: &[f64]) -> f64 {
            vars.get(ID).copied().unwrap_or(0.0)
        }
    }
    
    let var_x = VarSim::<0>;
    let var_y = VarSim::<1>;
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += var_x.eval(&vars) + var_y.eval(&vars);
    }
    let method_time = start.elapsed();
    
    println!("Method calls: var_x.eval(&vars) + var_y.eval(&vars)");
    println!("Time: {:?} ({:.2} ns/op)", method_time, method_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", method_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // STEP 6: Trait object overhead
    // ========================================================================
    
    println!("📊 Step 6: Trait Object Overhead");
    println!("--------------------------------");
    
    trait EvalTrait {
        fn eval(&self, vars: &[f64]) -> f64;
    }
    
    impl<const ID: usize> EvalTrait for VarSim<ID> {
        fn eval(&self, vars: &[f64]) -> f64 {
            vars.get(ID).copied().unwrap_or(0.0)
        }
    }
    
    let var_x_trait: &dyn EvalTrait = &VarSim::<0>;
    let var_y_trait: &dyn EvalTrait = &VarSim::<1>;
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += var_x_trait.eval(&vars) + var_y_trait.eval(&vars);
    }
    let trait_object_time = start.elapsed();
    
    println!("Trait objects: var_x_trait.eval(&vars) + var_y_trait.eval(&vars)");
    println!("Time: {:?} ({:.2} ns/op)", trait_object_time, trait_object_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", trait_object_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // STEP 7: Actual compile-time system
    // ========================================================================
    
    println!("📊 Step 7: Actual Compile-Time System");
    println!("-------------------------------------");
    
    let x = var::<0>();
    let y = var::<1>();
    let expr = x.clone().add(y.clone());
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += expr.eval(&vars);
    }
    let compile_time_time = start.elapsed();
    
    println!("Compile-time system: expr.eval(&vars) where expr = x.add(y)");
    println!("Time: {:?} ({:.2} ns/op)", compile_time_time, compile_time_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", compile_time_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // STEP 8: Manual implementation of Add<Var<0>, Var<1>>
    // ========================================================================
    
    println!("📊 Step 8: Manual Add Implementation");
    println!("------------------------------------");
    
    struct ManualAdd {
        left: VarSim<0>,
        right: VarSim<1>,
    }
    
    impl ManualAdd {
        fn eval(&self, vars: &[f64]) -> f64 {
            self.left.eval(vars) + self.right.eval(vars)
        }
    }
    
    let manual_add = ManualAdd {
        left: VarSim::<0>,
        right: VarSim::<1>,
    };
    
    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += manual_add.eval(&vars);
    }
    let manual_time = start.elapsed();
    
    println!("Manual Add: manual_add.eval(&vars)");
    println!("Time: {:?} ({:.2} ns/op)", manual_time, manual_time.as_nanos() as f64 / iterations as f64);
    println!("Overhead vs Rust: {:.1}x", manual_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();

    // ========================================================================
    // ANALYSIS SUMMARY
    // ========================================================================
    
    println!("🎓 Overhead Analysis Summary");
    println!("============================");
    println!();
    
    println!("Performance breakdown:");
    println!("1. Pure Rust:           {:>8.2} ns/op (baseline)", rust_time.as_nanos() as f64 / iterations as f64);
    println!("2. Array access:        {:>8.2} ns/op ({:.1}x)", array_time.as_nanos() as f64 / iterations as f64, array_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("3. Bounds checking:     {:>8.2} ns/op ({:.1}x)", bounds_check_time.as_nanos() as f64 / iterations as f64, bounds_check_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("4. Function calls:      {:>8.2} ns/op ({:.1}x)", function_time.as_nanos() as f64 / iterations as f64, function_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("5. Method calls:        {:>8.2} ns/op ({:.1}x)", method_time.as_nanos() as f64 / iterations as f64, method_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("6. Trait objects:       {:>8.2} ns/op ({:.1}x)", trait_object_time.as_nanos() as f64 / iterations as f64, trait_object_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("7. Compile-time system: {:>8.2} ns/op ({:.1}x)", compile_time_time.as_nanos() as f64 / iterations as f64, compile_time_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!("8. Manual Add:          {:>8.2} ns/op ({:.1}x)", manual_time.as_nanos() as f64 / iterations as f64, manual_time.as_nanos() as f64 / rust_time.as_nanos() as f64);
    println!();
    
    println!("🔍 Key Insights:");
    
    let bounds_overhead = bounds_check_time.as_nanos() as f64 / array_time.as_nanos() as f64;
    if bounds_overhead > 2.0 {
        println!("• Bounds checking adds significant overhead ({:.1}x)", bounds_overhead);
        println!("  The .get().copied().unwrap_or() pattern is expensive!");
    }
    
    let trait_overhead = trait_object_time.as_nanos() as f64 / method_time.as_nanos() as f64;
    if trait_overhead > 2.0 {
        println!("• Trait objects add virtual dispatch overhead ({:.1}x)", trait_overhead);
    }
    
    let system_vs_manual = compile_time_time.as_nanos() as f64 / manual_time.as_nanos() as f64;
    if system_vs_manual > 1.5 {
        println!("• The compile-time system has additional overhead vs manual implementation ({:.1}x)", system_vs_manual);
    } else {
        println!("• The compile-time system performs similarly to manual implementation");
    }
    
    println!();
    println!("💡 Optimization Opportunities:");
    
    if bounds_check_time.as_nanos() as f64 / rust_time.as_nanos() as f64 > 3.0 {
        println!("• Replace .get().copied().unwrap_or() with unsafe indexing for hot paths");
        println!("• Use compile-time bounds checking instead of runtime checks");
    }
    
    if compile_time_time.as_nanos() as f64 / rust_time.as_nanos() as f64 > 5.0 {
        println!("• Consider procedural macros for true zero-cost abstractions");
        println!("• Investigate compiler optimization flags");
    } else {
        println!("• Current overhead is reasonable for the abstraction level provided");
        println!("• Focus on algorithmic improvements rather than micro-optimizations");
    }
    
    println!();
    println!("🎯 The main overhead sources are:");
    println!("1. Bounds checking in vars.get(ID).copied().unwrap_or(0.0)");
    println!("2. Function call overhead for eval() methods");
    println!("3. Potential lack of compiler inlining in debug builds");
} 