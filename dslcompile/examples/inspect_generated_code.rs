use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::backends::RustCodeGenerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Inspecting Generated Rust Code");
    println!("=================================");

    // Create a simple expression: x + y
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();
    let expr = &x + &y;
    let ast: ASTRepr<f64> = expr.into();

    // Generate Rust code
    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast, "add_func")?;

    println!("Generated Rust code for 'x + y':");
    println!("{rust_code}");
    println!();

    // Create a more complex expression: x*x + 2*x*y + y*y
    let complex_expr = &x * &x + 2.0 * &x * &y + &y * &y;
    let complex_ast: ASTRepr<f64> = complex_expr.into();

    let complex_code = codegen.generate_function(&complex_ast, "complex_func")?;

    println!("Generated Rust code for 'x² + 2xy + y²':");
    println!("{complex_code}");
    println!();

    // Create an expression with transcendental functions: sin(x) + cos(y)
    let trig_expr = x.sin() + y.cos();
    let trig_ast: ASTRepr<f64> = trig_expr.into();

    let trig_code = codegen.generate_function(&trig_ast, "trig_func")?;

    println!("Generated Rust code for 'sin(x) + cos(y)':");
    println!("{trig_code}");

    Ok(())
}
