// Detailed explanation of type inference in builder.lambda(|x| f.call(g.call(x)))

use dslcompile::{composition::MathFunction, prelude::*};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Type Inference Explanation ===\n");

    // Step 1: Create some functions
    let square_plus_one = MathFunction::<f64>::from_lambda("square_plus_one", |builder| {
        builder.lambda(|x| x.clone() * x + 1.0)
    });

    let linear =
        MathFunction::<f64>::from_lambda("linear", |builder| builder.lambda(|x| x * 2.0 + 3.0));

    // Step 2: Convert to callable functions
    let f = square_plus_one.as_callable(); // f: CallableFunction<f64>
    let g = linear.as_callable(); // g: CallableFunction<f64>

    println!("Function types:");
    println!("  f: CallableFunction<f64>");
    println!("  g: CallableFunction<f64>");
    println!();

    // Step 3: The key type inference happens here
    println!("=== Type Inference Step by Step ===");
    println!();

    println!("In: builder.lambda(|x| f.call(g.call(x)))");
    println!();

    println!("1. FunctionBuilder.lambda() signature:");
    println!("   pub fn lambda<F>(&mut self, f: F) -> Lambda<T>");
    println!("   where F: FnOnce(LambdaVar<T>) -> LambdaVar<T>");
    println!();

    println!("2. Since we're using MathFunction::<f64>, we have T = f64");
    println!("   So F: FnOnce(LambdaVar<f64>) -> LambdaVar<f64>");
    println!();

    println!("3. The closure |x| f.call(g.call(x)) must match this signature");
    println!("   Therefore: x: LambdaVar<f64>");
    println!();

    println!("4. Let's trace the inner calls:");
    println!("   - g.call(x) takes LambdaVar<f64> → returns LambdaVar<f64>");
    println!("   - f.call(g.call(x)) takes LambdaVar<f64> → returns LambdaVar<f64>");
    println!("   ✓ Types match perfectly!");
    println!();

    // Let's demonstrate this with explicit types to make it crystal clear
    println!("=== Explicit Type Annotations (for clarity) ===");

    let explicit_composed = MathFunction::<f64>::from_lambda("explicit", |builder| {
        builder.lambda(|x: LambdaVar<f64>| -> LambdaVar<f64> {
            let intermediate: LambdaVar<f64> = g.call(x);
            let result: LambdaVar<f64> = f.call(intermediate);
            result
        })
    });

    println!("Explicit version:");
    println!("builder.lambda(|x: LambdaVar<f64>| -> LambdaVar<f64> {{");
    println!("    let intermediate: LambdaVar<f64> = g.call(x);");
    println!("    let result: LambdaVar<f64> = f.call(intermediate);");
    println!("    result");
    println!("}}");
    println!();

    // Test that they produce the same results
    let implicit_composed = MathFunction::<f64>::from_lambda("implicit", |builder| {
        builder.lambda(|x| f.call(g.call(x))) // Types inferred!
    });

    let test_input = 2.0;
    let explicit_result = explicit_composed.eval(hlist![test_input]);
    let implicit_result = implicit_composed.eval(hlist![test_input]);

    println!("=== Verification ===");
    println!("Both versions produce the same result:");
    println!("  Explicit types: {explicit_result}");
    println!("  Inferred types: {implicit_result}");
    println!("  ✓ Match: {}", explicit_result == implicit_result);
    println!();

    // Show what LambdaVar actually represents
    println!("=== What is LambdaVar<f64>? ===");
    println!("LambdaVar<f64> is NOT a concrete f64 value!");
    println!("Instead, it's a symbolic representation of an expression.");
    println!();
    println!("When you write |x| x * x + 1.0:");
    println!("- x is a LambdaVar<f64> representing \"the input variable\"");
    println!("- x * x creates a new LambdaVar<f64> representing \"input squared\"");
    println!("- x * x + 1.0 creates another LambdaVar<f64> representing \"input squared plus 1\"");
    println!();
    println!("Think of LambdaVar<f64> as a \"symbolic f64 expression\" rather than");
    println!("an actual f64 number. It builds an Abstract Syntax Tree (AST) that");
    println!("will later be evaluated with concrete values.");
    println!();

    // Show the AST building process
    println!("=== AST Building Process ===");
    println!("When the lambda executes:");
    println!("1. builder creates: x = LambdaVar(ASTRepr::Variable(0))");
    println!("2. g.call(x) substitutes x into g's body, creating new AST");
    println!("3. f.call(...) substitutes that result into f's body");
    println!("4. Final result is a complex AST representing f(g(x))");
    println!();
    println!("The 'x' parameter is a handle for building expressions,");
    println!("not a runtime value!");

    Ok(())
}
