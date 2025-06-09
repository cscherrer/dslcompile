use dlopen2::raw::Library;
use dslcompile::ast::{ASTRepr, DynamicContext};
use dslcompile::backends::{RustCodeGenerator, RustCompiler, RustOptLevel};

#[inline(never)]
pub fn native_eval_target(x_val: f64, y_val: f64) -> f64 {
    x_val * y_val + 42.0
}

// This will be set to the compiled function pointer
static mut COMPILED_FUNC_PTR: Option<extern "C" fn(f64, f64) -> f64> = None;

#[inline(never)]
pub fn compiled_eval_target(x_val: f64, y_val: f64) -> f64 {
    unsafe {
        let func = COMPILED_FUNC_PTR.expect("Compiled function not initialized");
        func(x_val, y_val)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build and compile the function once
    let ctx = DynamicContext::new();
    let x = ctx.var();
    let y = ctx.var();
    let expr = &x * &y + 42.0;
    let ast_expr: ASTRepr<f64> = expr.into();

    let codegen = RustCodeGenerator::new();
    let rust_code = codegen.generate_function(&ast_expr, "eval_func")?;

    let compiler = RustCompiler::with_opt_level(RustOptLevel::O3)
        .with_extra_flags(hlist!["-C".to_string(), "target-cpu=native".to_string()]);

    let temp_dir = std::env::temp_dir();
    let source_path = temp_dir.join("profile_test.rs");
    let lib_path = temp_dir.join("libprofile_test.so");

    std::fs::write(&source_path, &rust_code)?;
    compiler.compile_dylib(&rust_code, &source_path, &lib_path)?;

    let library = Library::open(&lib_path)?;
    let func: extern "C" fn(f64, f64) -> f64 = unsafe { library.symbol("eval_func")? };

    // Store the function pointer
    unsafe {
        COMPILED_FUNC_PTR = Some(func);
    }

    let x = 3.5;
    let y = 4.2;

    println!("Compiled result: {}", compiled_eval_target(x, y));
    println!("Native result: {}", native_eval_target(x, y));

    // Keep library alive
    std::mem::forget(library);

    Ok(())
}
