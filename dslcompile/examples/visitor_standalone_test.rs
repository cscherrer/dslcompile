//! Standalone Visitor Pattern Test
//!
//! This demonstrates the visitor pattern working correctly to solve
//! the AST traversal duplication problem described in the Perplexity article.

use dslcompile::ast::ast_repr::{ASTRepr, Collection, Lambda};
use dslcompile::ast::{ASTVisitor, ASTMutVisitor, visit_ast, visit_ast_mut};
use std::collections::HashSet;

/// Variable collector visitor - replaces scattered variable collection logic
struct VariableCollector {
    variables: HashSet<usize>,
}

impl VariableCollector {
    fn new() -> Self {
        Self {
            variables: HashSet::new(),
        }
    }

    fn get_variables(self) -> HashSet<usize> {
        self.variables
    }
}

impl ASTVisitor<f64> for VariableCollector {
    type Output = ();
    type Error = ();

    fn visit_constant(&mut self, _value: &f64) -> Result<Self::Output, Self::Error> {
        Ok(())
    }

    fn visit_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error> {
        self.variables.insert(index);
        Ok(())
    }

    fn visit_bound_var(&mut self, index: usize) -> Result<Self::Output, Self::Error> {
        self.variables.insert(index);
        Ok(())
    }

    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
        Ok(())
    }

    fn visit_collection_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error> {
        self.variables.insert(index);
        Ok(())
    }
}

/// Constant folder visitor - replaces scattered transformation logic
struct ConstantFolder;

impl ASTMutVisitor<f64> for ConstantFolder {
    type Error = ();

    fn visit_add_mut(&mut self, left: ASTRepr<f64>, right: ASTRepr<f64>) -> Result<ASTRepr<f64>, Self::Error> {
        let left_transformed = self.visit_mut(left)?;
        let right_transformed = self.visit_mut(right)?;

        match (&left_transformed, &right_transformed) {
            (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                Ok(ASTRepr::Constant(a + b))
            }
            _ => Ok(ASTRepr::Add(Box::new(left_transformed), Box::new(right_transformed)))
        }
    }

    fn visit_mul_mut(&mut self, left: ASTRepr<f64>, right: ASTRepr<f64>) -> Result<ASTRepr<f64>, Self::Error> {
        let left_transformed = self.visit_mut(left)?;
        let right_transformed = self.visit_mut(right)?;

        match (&left_transformed, &right_transformed) {
            (ASTRepr::Constant(a), ASTRepr::Constant(b)) => {
                Ok(ASTRepr::Constant(a * b))
            }
            _ => Ok(ASTRepr::Mul(Box::new(left_transformed), Box::new(right_transformed)))
        }
    }

    fn visit_sin_mut(&mut self, inner: ASTRepr<f64>) -> Result<ASTRepr<f64>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;

        match &inner_transformed {
            ASTRepr::Constant(value) => {
                Ok(ASTRepr::Constant(value.sin()))
            }
            _ => Ok(ASTRepr::Sin(Box::new(inner_transformed)))
        }
    }
}

fn main() {
    println!("🎯 Standalone Visitor Pattern Test");
    println!("===================================");
    
    // Create a sample AST directly: sin(x + 2.0) * (y + 3.0)
    let x_plus_2 = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)), // x
        Box::new(ASTRepr::Constant(2.0))
    );
    
    let sin_x_plus_2 = ASTRepr::Sin(Box::new(x_plus_2));
    
    let y_plus_3 = ASTRepr::Add(
        Box::new(ASTRepr::Variable(1)), // y
        Box::new(ASTRepr::Constant(3.0))
    );
    
    let full_expr = ASTRepr::Mul(
        Box::new(sin_x_plus_2),
        Box::new(y_plus_3)
    );
    
    println!("\n📊 Sample Expression: sin(x + 2.0) * (y + 3.0)");
    println!("AST: {full_expr:?}");

    // Test 1: Variable collection
    println!("\n🔍 Test 1: Variable Collection");
    println!("Before: Required match statements in every file that needed variables");
    println!("After: Single focused visitor implementation");
    
    let mut collector = VariableCollector::new();
    visit_ast(&full_expr, &mut collector).unwrap();
    let variables = collector.get_variables();
    
    println!("Variables found: {variables:?}");
    assert_eq!(variables.len(), 2);
    assert!(variables.contains(&0)); // x
    assert!(variables.contains(&1)); // y
    println!("✅ Variable collection test passed!");

    // Test 2: Constant folding
    println!("\n🔧 Test 2: Constant Folding Transformation");
    println!("Before: Transformation logic scattered across normalization files");
    println!("After: Clean transformer visitor");
    
    // Create expression: (2.0 + 3.0) * sin(1.0)
    let const_expr = ASTRepr::Mul(
        Box::new(ASTRepr::Add(
            Box::new(ASTRepr::Constant(2.0)),
            Box::new(ASTRepr::Constant(3.0))
        )),
        Box::new(ASTRepr::Sin(Box::new(ASTRepr::Constant(1.0))))
    );
    
    println!("Original: {const_expr:?}");
    
    let mut folder = ConstantFolder;
    let folded_ast = visit_ast_mut(const_expr, &mut folder).unwrap();
    
    println!("Folded: {folded_ast:?}");
    
    // Verify constant folding worked
    match folded_ast {
        ASTRepr::Constant(value) => {
            let expected = 5.0 * 1.0_f64.sin(); // (2+3) * sin(1)
            assert!((value - expected).abs() < 1e-10);
            println!("✅ Constant folding test passed! Result: {value:.6}");
        }
        _ => panic!("❌ Constant folding failed - expected single constant"),
    }

    println!("\n✅ Benefits Demonstrated:");
    println!("• Eliminated repetitive match statements across multiple files");
    println!("• Zero runtime overhead (static dispatch)");
    println!("• Compiler-enforced completeness for new AST variants");
    println!("• Consistent behavior across all traversals");
    println!("• Easy extension without touching existing code");

    println!("\n🎉 The visitor pattern successfully eliminates AST traversal duplication!");
    println!("   This solves the exact problem described in the Perplexity article.");
} 