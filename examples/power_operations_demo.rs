use mathcompile::backends::cranelift::JITCompiler;
use mathcompile::error::Result;
use mathcompile::final_tagless::{ASTEval, ASTMathExpr};

fn main() -> Result<()> {
    println!("⚡ MathCompile Enhanced Power Operations Demo");
    println!("=======================================");
    println!();

    demo_integer_powers()?;
    demo_fractional_powers()?;
    demo_variable_powers()?;
    demo_negative_powers()?;
    demo_complex_power_expressions()?;

    Ok(())
}

/// Demo 1: Integer power optimizations
fn demo_integer_powers() -> Result<()> {
    println!("🔢 Demo 1: Integer Power Optimizations");
    println!("--------------------------------------");
    println!("Optimized multiplication sequences for x^n where n is integer");
    println!();

    let test_cases = [
        (2, "x²"),
        (3, "x³"),
        (4, "x⁴ = (x²)²"),
        (5, "x⁵ = x⁴ * x"),
        (6, "x⁶ = (x³)²"),
        (8, "x⁸ = ((x²)²)²"),
        (10, "x¹⁰ = (x⁵)²"),
        (16, "x¹⁶ = (x⁸)²"),
    ];

    for (exp, description) in test_cases {
        let expr = ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(f64::from(exp)));

        let compiler = JITCompiler::new()?;
        let jit_func = compiler.compile_single_var(&expr, "x")?;

        let test_value = 2.0;
        let jit_result = jit_func.call_single(test_value);
        let std_result = test_value.powi(exp);
        let error = (jit_result - std_result).abs();

        println!(
            "   {description} = {jit_result:.6} (JIT) vs {std_result:.6} (std), error: {error:.2e}"
        );
        println!(
            "     Operations: {}, Compilation: {} μs",
            jit_func.stats.operation_count, jit_func.stats.compilation_time_us
        );
    }
    println!();

    Ok(())
}

/// Demo 2: Fractional power optimizations
fn demo_fractional_powers() -> Result<()> {
    println!("🔄 Demo 2: Fractional Power Optimizations");
    println!("-----------------------------------------");
    println!("Special handling for common fractional exponents");
    println!();

    let test_cases = [
        (0.5, "x^0.5 = sqrt(x)"),
        (-0.5, "x^-0.5 = 1/sqrt(x)"),
        (1.0 / 3.0, "x^(1/3) = cube root using exp/ln"),
    ];

    for (exp, description) in test_cases {
        let expr = ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(exp));

        let compiler = JITCompiler::new()?;
        let jit_func = compiler.compile_single_var(&expr, "x")?;

        let test_value = 8.0;
        let jit_result = jit_func.call_single(test_value);
        let std_result = test_value.powf(exp);
        let error = (jit_result - std_result).abs();

        println!(
            "   {description} = {jit_result:.8} (JIT) vs {std_result:.8} (std), error: {error:.2e}"
        );
        println!(
            "     Operations: {}, Compilation: {} μs",
            jit_func.stats.operation_count, jit_func.stats.compilation_time_us
        );
    }
    println!();

    Ok(())
}

/// Demo 3: Variable power operations
fn demo_variable_powers() -> Result<()> {
    println!("🔀 Demo 3: Variable Power Operations");
    println!("-----------------------------------");
    println!("Using exp(y * ln(x)) for variable exponents");
    println!();

    // Create expression: x^y
    let expr = ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::var_by_name("y"));

    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_two_vars(&expr, "x", "y")?;

    println!("🔧 Compilation successful!");
    println!("   Variables: {}", jit_func.stats.variable_count);
    println!("   Operations: {}", jit_func.stats.operation_count);
    println!(
        "   Compilation time: {} μs",
        jit_func.stats.compilation_time_us
    );
    println!();

    let test_cases = [
        (2.0, 3.0),  // 2^3 = 8
        (3.0, 2.5),  // 3^2.5 ≈ 15.588
        (4.0, 0.5),  // 4^0.5 = 2
        (10.0, 0.3), // 10^0.3 ≈ 1.995
    ];

    println!("📊 Testing variable powers x^y:");
    for (x_val, y_val) in test_cases {
        let jit_result = jit_func.call_two_vars(x_val, y_val);
        let std_result = x_val.powf(y_val);
        let error = (jit_result - std_result).abs();

        println!(
            "   {x_val:.1}^{y_val:.1} = {jit_result:8.6} (JIT) vs {std_result:8.6} (std), error: {error:.2e}"
        );
    }
    println!();

    Ok(())
}

/// Demo 4: Negative power optimizations
fn demo_negative_powers() -> Result<()> {
    println!("➖ Demo 4: Negative Power Optimizations");
    println!("--------------------------------------");
    println!("Efficient handling of negative integer exponents");
    println!();

    let test_cases = [
        (-1, "x^-1 = 1/x"),
        (-2, "x^-2 = 1/(x²)"),
        (-3, "x^-3 = 1/(x³)"),
        (-4, "x^-4 = 1/(x⁴)"),
    ];

    for (exp, description) in test_cases {
        let expr = ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(f64::from(exp)));

        let compiler = JITCompiler::new()?;
        let jit_func = compiler.compile_single_var(&expr, "x")?;

        let test_value = 3.0;
        let jit_result = jit_func.call_single(test_value);
        let std_result = test_value.powi(exp);
        let error = (jit_result - std_result).abs();

        println!(
            "   {description} = {jit_result:.8} (JIT) vs {std_result:.8} (std), error: {error:.2e}"
        );
    }
    println!();

    Ok(())
}

/// Demo 5: Complex expressions with multiple power operations
fn demo_complex_power_expressions() -> Result<()> {
    println!("🔬 Demo 5: Complex Power Expressions");
    println!("------------------------------------");
    println!("Multiple power operations in a single expression");
    println!();

    // Create expression: x² + y³ + (x*y)^0.5
    let x = ASTEval::var_by_name("x");
    let y = ASTEval::var_by_name("y");
    let expr = ASTEval::add(
        ASTEval::add(
            ASTEval::pow(x.clone(), ASTEval::constant(2.0)),
            ASTEval::pow(y.clone(), ASTEval::constant(3.0)),
        ),
        ASTEval::pow(ASTEval::mul(x, y), ASTEval::constant(0.5)),
    );

    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_two_vars(&expr, "x", "y")?;

    println!("🔧 Expression: x² + y³ + sqrt(x*y)");
    println!("   Variables: {}", jit_func.stats.variable_count);
    println!("   Operations: {}", jit_func.stats.operation_count);
    println!(
        "   Compilation time: {} μs",
        jit_func.stats.compilation_time_us
    );
    println!();

    let test_cases = [
        (2.0, 3.0), // 4 + 27 + sqrt(6) ≈ 33.449
        (4.0, 2.0), // 16 + 8 + sqrt(8) ≈ 26.828
        (1.0, 5.0), // 1 + 125 + sqrt(5) ≈ 128.236
    ];

    println!("📊 Testing complex power expression:");
    for (x_val, y_val) in test_cases {
        let jit_result = jit_func.call_two_vars(x_val, y_val);
        let std_result = x_val.powi(2) + y_val.powi(3) + (x_val * y_val).sqrt();
        let error = (jit_result - std_result).abs();

        println!(
            "   f({x_val:.1}, {y_val:.1}) = {jit_result:8.3} (JIT) vs {std_result:8.3} (std), error: {error:.2e}"
        );
    }
    println!();

    println!("✅ Enhanced power operations working efficiently!");
    println!("🎯 Phase 1 JIT Compilation Foundation - COMPLETED!");

    Ok(())
}
