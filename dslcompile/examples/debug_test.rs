use dslcompile::{
    SymbolicOptimizer, DynamicExpr, contexts::DynamicContext, Expr,
};

fn main() {
    println!("🔍 Debug: Testing expression building and code generation");

    // Create the same expression as the failing test: x^2 + 2*x + 1
    let mut math = DynamicContext::new();
    let x: DynamicExpr<f64> = math.var();
    let x_squared = x.clone().pow(math.constant(2.0));
    let expr: Expr<f64> = (&x_squared + 2.0 * &x + 1.0).into();

    println!("Expression AST: {:#?}", expr.as_ast());
    
    // Test the optimizer
    let optimizer = SymbolicOptimizer::new().unwrap();
    
    let rust_code = optimizer
        .generate_rust_source(expr.as_ast(), "debug_func")
        .unwrap();
    
    println!("Generated Rust code:\n{}", rust_code);
    
    // Let's also test a simpler expression
    println!("\n🔍 Testing simpler expression: x + 1");
    let mut math2 = DynamicContext::new();
    let x2: DynamicExpr<f64> = math2.var();
    let simple_expr: Expr<f64> = (&x2 + 1.0).into();
    
    println!("Simple expression AST: {:#?}", simple_expr.as_ast());
    
    let simple_rust_code = optimizer
        .generate_rust_source(simple_expr.as_ast(), "simple_func")
        .unwrap();
    
    println!("Simple generated Rust code:\n{}", simple_rust_code);
    
    // Test trigonometric functions
    println!("\n🔍 Testing trigonometric expression: sin(2*x + cos(y))");
    let mut math3 = DynamicContext::new();
    let x3: DynamicExpr<f64> = math3.var();
    let y3: DynamicExpr<f64> = math3.var();
    let trig_expr: Expr<f64> = (2.0 * &x3 + y3.cos()).sin().into();
    
    println!("Trigonometric expression AST: {:#?}", trig_expr.as_ast());
    
    let trig_rust_code = optimizer
        .generate_rust_source(trig_expr.as_ast(), "trig_func")
        .unwrap();
    
    println!("Trigonometric generated Rust code:\n{}", trig_rust_code);
}