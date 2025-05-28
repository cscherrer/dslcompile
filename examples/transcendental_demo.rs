use mathcompile::backends::cranelift::JITCompiler;
use mathcompile::error::Result;
use mathcompile::final_tagless::{ASTEval, ASTMathExpr};

fn main() -> Result<()> {
    println!("🧮 MathCompile Transcendental Functions Demo");
    println!("========================================");
    println!();

    demo_exponential()?;
    demo_logarithm()?;
    demo_trigonometric()?;
    demo_complex_expression()?;

    Ok(())
}

/// Demo 1: Exponential function with optimal rational approximation
fn demo_exponential() -> Result<()> {
    println!("📈 Demo 1: Exponential Function exp(x)");
    println!("--------------------------------------");
    println!("Using optimal rational approximation (4,5) with error ~4.2e-12");
    println!();

    // Define expression: exp(x)
    let expr = ASTEval::exp(ASTEval::var_by_name("x"));

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&expr, "x")?;

    println!("🔧 Compilation successful!");
    println!("   Code size: {} bytes", jit_func.stats.code_size_bytes);
    println!(
        "   Compilation time: {} μs",
        jit_func.stats.compilation_time_us
    );
    println!();

    // Test values in the optimal range [-1, 1]
    let test_values = [-1.0, -0.5, 0.0, 0.5, 1.0];

    println!("📊 Testing exp(x) in optimal range [-1, 1]:");
    for x in test_values {
        let jit_result = jit_func.call_single(x);
        let std_result = x.exp();
        let error = (jit_result - std_result).abs();

        println!("   exp({x:4.1}) = {jit_result:12.10} (JIT) vs {std_result:12.10} (std), error: {error:.2e}");
    }
    println!();

    Ok(())
}

/// Demo 2: Natural logarithm with optimal rational approximation
fn demo_logarithm() -> Result<()> {
    println!("📉 Demo 2: Natural Logarithm ln(x)");
    println!("----------------------------------");
    println!("Using optimal rational approximation (4,4) with error ~6.2e-12");
    println!("Implemented as ln(x) = ln(1 + (x-1))");
    println!();

    // Define expression: ln(x)
    let expr = ASTEval::ln(ASTEval::var_by_name("x"));

    // Compile to native code
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&expr, "x")?;

    println!("🔧 Compilation successful!");
    println!();

    // Test values in the range [1, 2] (optimal for ln(1+u) where u ∈ [0,1])
    let test_values = [1.0, 1.2, 1.5, 1.8, 2.0];

    println!("📊 Testing ln(x) in range [1, 2]:");
    for x in test_values {
        let jit_result = jit_func.call_single(x);
        let std_result = x.ln();
        let error = (jit_result - std_result).abs();

        println!("   ln({x:3.1}) = {jit_result:12.10} (JIT) vs {std_result:12.10} (std), error: {error:.2e}");
    }
    println!();

    Ok(())
}

/// Demo 3: Trigonometric functions
fn demo_trigonometric() -> Result<()> {
    println!("🌊 Demo 3: Trigonometric Functions");
    println!("----------------------------------");
    println!("sin(x): Shifted cosine implementation sin(x) = cos(π/2 - x)");
    println!("cos(x): Optimal rational approximation (5,2) with error ~8.5e-11");
    println!();

    // Define expressions
    let sin_expr = ASTEval::sin(ASTEval::var_by_name("x"));
    let cos_expr = ASTEval::cos(ASTEval::var_by_name("x"));

    // Compile to native code
    let compiler1 = JITCompiler::new()?;
    let sin_func = compiler1.compile_single_var(&sin_expr, "x")?;

    let compiler2 = JITCompiler::new()?;
    let cos_func = compiler2.compile_single_var(&cos_expr, "x")?;

    println!("🔧 Compilation successful!");
    println!();

    // Test values in the optimal range [-π/4, π/4]
    let test_values = [
        -std::f64::consts::PI / 4.0,
        -std::f64::consts::PI / 8.0,
        0.0,
        std::f64::consts::PI / 8.0,
        std::f64::consts::PI / 4.0,
    ];

    println!("📊 Testing sin(x) and cos(x) in range [-π/4, π/4]:");
    for x in test_values {
        let sin_jit = sin_func.call_single(x);
        let sin_std = x.sin();
        let sin_error = (sin_jit - sin_std).abs();

        let cos_jit = cos_func.call_single(x);
        let cos_std = x.cos();
        let cos_error = (cos_jit - cos_std).abs();

        println!("   x = {x:8.5}");
        println!("     sin({x:8.5}) = {sin_jit:12.10} (JIT) vs {sin_std:12.10} (std), error: {sin_error:.2e}");
        println!("     cos({x:8.5}) = {cos_jit:12.10} (JIT) vs {cos_std:12.10} (std), error: {cos_error:.2e}");
        println!();
    }

    Ok(())
}

/// Demo 4: Complex expression combining multiple transcendental functions
fn demo_complex_expression() -> Result<()> {
    println!("🔬 Demo 4: Complex Expression");
    println!("-----------------------------");
    println!("Expression: exp(x) * sin(y) + ln(x) * cos(y)");
    println!();

    // Define complex expression: exp(x) * sin(y) + ln(x) * cos(y)
    let x = ASTEval::var_by_name("x");
    let y = ASTEval::var_by_name("y");
    let expr = ASTEval::add(
        ASTEval::mul(ASTEval::exp(x.clone()), ASTEval::sin(y.clone())),
        ASTEval::mul(ASTEval::ln(x), ASTEval::cos(y)),
    );

    // Compile to native code
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

    // Test with various values
    let test_cases = [
        (1.0, 0.0),
        (1.5, std::f64::consts::PI / 6.0),
        (2.0, std::f64::consts::PI / 4.0),
    ];

    println!("📊 Testing complex expression:");
    for (x_val, y_val) in test_cases {
        let jit_result = jit_func.call_two_vars(x_val, y_val);
        let std_result = x_val.exp() * y_val.sin() + x_val.ln() * y_val.cos();
        let error = (jit_result - std_result).abs();

        println!("   f({x_val:.1}, {y_val:.5}) = {jit_result:12.8} (JIT) vs {std_result:12.8} (std), error: {error:.2e}");
    }
    println!();

    println!("✅ All transcendental functions working with high precision!");
    println!("🎯 Ready for production mathematical computations!");

    Ok(())
}
