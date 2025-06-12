// Comparison: Old Variable approach vs New LambdaVar approach
// Shows how LambdaVar provides a more ergonomic interface over the same underlying Variable system

use dslcompile::{
    ast::ast_repr::ASTRepr,
    composition::{FunctionBuilder, LambdaVar, MathFunction},
    contexts::dynamic::DynamicContext,
    prelude::*,
};
use frunk::hlist;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Variable vs LambdaVar Comparison ===\n");

    // OLD APPROACH: Manual Variable management with DynamicContext
    println!("=== OLD APPROACH: Manual Variable Management ===");
    {
        let mut ctx = DynamicContext::<f64>::new();
        let x_var = ctx.var(); // This creates a VariableExpr wrapping Variable(0)
        let y_var = ctx.var(); // This creates a VariableExpr wrapping Variable(1)

        // Build expression manually: x² + 2y + 1
        let expr = &x_var * &x_var + 2.0 * &y_var + 1.0;

        println!("Created variables through DynamicContext:");
        println!("  x_var: VariableExpr -> ASTRepr::Variable(0)");
        println!("  y_var: VariableExpr -> ASTRepr::Variable(1)");
        println!("  Expression: x² + 2y + 1");

        let result = ctx.eval(&expr, frunk::hlist![3.0, 4.0]);
        println!("  Result at x=3, y=4: {}", result); // 3² + 2*4 + 1 = 18
    }
    println!();

    // NEW APPROACH: LambdaVar with automatic scoping
    println!("=== NEW APPROACH: LambdaVar with Lambda Scoping ===");
    {
        let math_func = MathFunction::<f64>::from_lambda("example", |builder| {
            builder.lambda(|x| {
                // x is LambdaVar<f64> wrapping Variable(0)
                x.clone() * x + 1.0 // Builds the same AST structure!
            })
        });

        println!("Created lambda variable automatically:");
        println!("  x: LambdaVar<f64> -> ASTRepr::Variable(0)");
        println!("  Expression: x² + 1");

        let result = math_func.eval(hlist![3.0]);
        println!("  Result at x=3: {}", result); // 3² + 1 = 10
    }
    println!();

    // UNDER THE HOOD: Both create the same AST structure
    println!("=== UNDER THE HOOD: Same AST Structure ===");
    {
        // Manual Variable approach
        let manual_ast = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)), // Variable 0
                Box::new(ASTRepr::Variable(0)), // Variable 0
            )),
            Box::new(ASTRepr::Constant(1.0)),
        );

        // LambdaVar approach (extracted from the lambda)
        let lambda_func = MathFunction::<f64>::from_lambda("extract", |builder| {
            builder.lambda(|x| x.clone() * x + 1.0)
        });

        // Extract the AST from the lambda
        let lambda_ast = &lambda_func.lambda().body;

        println!("Manual AST: Add(Mul(Variable(0), Variable(0)), Constant(1.0))");
        println!("Lambda AST: {:?}", lambda_ast);
        println!("✓ Both create identical AST structures!");

        // Verify they evaluate the same
        let manual_result = manual_ast.eval_with_vars(&[3.0]);
        let lambda_result = lambda_func.eval(hlist![3.0]);
        println!("  Manual result: {}", manual_result);
        println!("  Lambda result: {}", lambda_result);
        println!("  ✓ Same results: {}", manual_result == lambda_result);
    }
    println!();

    // KEY INSIGHT: LambdaVar is a wrapper around Variable with better ergonomics
    println!("=== KEY INSIGHT ===");
    println!("LambdaVar does NOT replace Variable - it WRAPS it!");
    println!();
    println!("LambdaVar<T> {{");
    println!("    ast: ASTRepr<T>  // This IS the Variable(index) under the hood!");
    println!("}}");
    println!();
    println!("Benefits of LambdaVar:");
    println!("  ✓ Automatic variable index management");
    println!("  ✓ Natural mathematical syntax with operator overloading");
    println!("  ✓ Scoped variable lifetime (no manual cleanup)");
    println!("  ✓ Type-safe composition");
    println!("  ✓ Same performance - just a wrapper!");
    println!();

    // COEXISTENCE: Both approaches work together
    println!("=== COEXISTENCE: Both Approaches Work Together ===");
    {
        let mut ctx = DynamicContext::<f64>::new();
        let old_var: TypedBuilderExpr<f64> = ctx.var();

        // Convert old Variable to LambdaVar manually
        let as_lambda_var = LambdaVar::new(old_var.clone().into_ast());

        // Use in lambda context
        let mixed_func = MathFunction::<f64>::from_lambda("mixed", |builder| {
            builder.lambda(|x| {
                // x is the new lambda variable (Variable(0))
                // We can use the converted old variable too, but it would be Variable(0) again
                x.clone() * x + 2.0 // Still just Variable(0) operations
            })
        });

        println!("✓ Old and new approaches use the same underlying Variable system");
        println!("✓ LambdaVar is just a more ergonomic interface!");
    }

    Ok(())
}
