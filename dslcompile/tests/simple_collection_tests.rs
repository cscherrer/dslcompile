//! Collection AST structure tests
//! Focus on verifying that summation AST structures are created correctly

use dslcompile::{
    ast::ast_repr::{ASTRepr, Collection},
    prelude::*,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summation_ast_structure() {
        // Test that summation AST structures are created correctly
        // This is important for ensuring codegen can handle the structures
        let mut ctx = DynamicContext::new();

        // Test 1: Range-based summation AST
        let sum_expr: DynamicExpr<f64, 0> = ctx.sum(1..=3, |i: DynamicExpr<f64, 0>| i);
        let ast = ctx.to_ast(&sum_expr);

        // Verify the AST has the correct structure
        match ast {
            ASTRepr::Sum(collection_box) => match collection_box.as_ref() {
                Collection::Map { lambda, collection } => {
                    assert!(matches!(*lambda.body, ASTRepr::BoundVar(0)));
                    match collection.as_ref() {
                        Collection::Range { start, end } => {
                            assert!(
                                matches!(**start, ASTRepr::Constant(v) if (v - 1.0f64).abs() < 1e-10)
                            );
                            assert!(
                                matches!(**end, ASTRepr::Constant(v) if (v - 3.0f64).abs() < 1e-10)
                            );
                        }
                        _ => panic!("Expected Range collection"),
                    }
                }
                _ => panic!("Expected Map collection"),
            },
            _ => panic!("Expected Sum AST structure"),
        }

        // Test 2: Data-based summation AST
        let data = vec![1.0, 2.0, 3.0];
        let data_sum: DynamicExpr<f64, 0> =
            ctx.sum(data.as_slice(), |x: DynamicExpr<f64, 0>| x.clone());
        let ast2 = ctx.to_ast(&data_sum);

        match ast2 {
            ASTRepr::Sum(collection_box) => match collection_box.as_ref() {
                Collection::Map { lambda, collection } => {
                    assert!(matches!(*lambda.body, ASTRepr::BoundVar(0)));
                    // Verify that data slice collections are embedded as DataArray in the AST
                    assert!(matches!(collection.as_ref(), Collection::DataArray(_)));
                }
                _ => panic!("Expected Map collection for data"),
            },
            _ => panic!("Expected Sum AST structure for data"),
        }

        // This test ensures the AST structure is correct for code generation
        // even though evaluation is not fully implemented yet
    }
}
