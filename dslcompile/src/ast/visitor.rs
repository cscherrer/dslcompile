use crate::ast::ast_repr::{ASTRepr, Collection, Lambda};
use crate::ast::Scalar;

/// Immutable visitor trait for AST traversal
/// 
/// This trait provides a clean way to traverse AST nodes without modifying them.
/// Each visit method has a default implementation that recursively visits children.
pub trait ASTVisitor<T: Scalar> {
    type Output;
    type Error;

    /// Visit any AST node - dispatches to specific visit methods
    fn visit(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        match expr {
            ASTRepr::Constant(value) => self.visit_constant(value),
            ASTRepr::Variable(index) => self.visit_variable(*index),
            ASTRepr::BoundVar(index) => self.visit_bound_var(*index),
            ASTRepr::Add(left, right) => self.visit_add(left, right),
            ASTRepr::Sub(left, right) => self.visit_sub(left, right),
            ASTRepr::Mul(left, right) => self.visit_mul(left, right),
            ASTRepr::Div(left, right) => self.visit_div(left, right),
            ASTRepr::Pow(base, exp) => self.visit_pow(base, exp),
            ASTRepr::Neg(inner) => self.visit_neg(inner),
            ASTRepr::Sin(inner) => self.visit_sin(inner),
            ASTRepr::Cos(inner) => self.visit_cos(inner),
            ASTRepr::Ln(inner) => self.visit_ln(inner),
            ASTRepr::Exp(inner) => self.visit_exp(inner),
            ASTRepr::Sqrt(inner) => self.visit_sqrt(inner),
            ASTRepr::Sum(collection) => self.visit_sum(collection),
            ASTRepr::Lambda(lambda) => self.visit_lambda(lambda),
            ASTRepr::Let(binding_id, expr, body) => self.visit_let(*binding_id, expr, body),
        }
    }

    // Leaf nodes - override these for custom behavior
    fn visit_constant(&mut self, value: &T) -> Result<Self::Output, Self::Error>;
    fn visit_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error>;
    fn visit_bound_var(&mut self, index: usize) -> Result<Self::Output, Self::Error>;

    // Binary operations - default implementations recursively visit children
    fn visit_add(&mut self, left: &ASTRepr<T>, right: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(left)?;
        self.visit(right)
    }

    fn visit_sub(&mut self, left: &ASTRepr<T>, right: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(left)?;
        self.visit(right)
    }

    fn visit_mul(&mut self, left: &ASTRepr<T>, right: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(left)?;
        self.visit(right)
    }

    fn visit_div(&mut self, left: &ASTRepr<T>, right: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(left)?;
        self.visit(right)
    }

    fn visit_pow(&mut self, base: &ASTRepr<T>, exp: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(base)?;
        self.visit(exp)
    }

    // Unary operations - default implementations recursively visit child
    fn visit_neg(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(inner)
    }

    fn visit_sin(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(inner)
    }

    fn visit_cos(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(inner)
    }

    fn visit_ln(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(inner)
    }

    fn visit_exp(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(inner)
    }

    fn visit_sqrt(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(inner)
    }

    // Complex structures
    fn visit_sum(&mut self, collection: &Collection<T>) -> Result<Self::Output, Self::Error> {
        self.visit_collection(collection)
    }

    fn visit_lambda(&mut self, lambda: &Lambda<T>) -> Result<Self::Output, Self::Error> {
        self.visit(&lambda.body)
    }

    fn visit_let(&mut self, _binding_id: usize, expr: &ASTRepr<T>, body: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        self.visit(expr)?;
        self.visit(body)
    }

    // Collection visitor - override for custom collection handling
    fn visit_collection(&mut self, collection: &Collection<T>) -> Result<Self::Output, Self::Error> {
        match collection {
            Collection::Empty => self.visit_empty_collection(),
            Collection::Singleton(expr) => self.visit(expr),
            Collection::Range { start, end } => {
                self.visit(start)?;
                self.visit(end)
            }
            Collection::Variable(index) => self.visit_collection_variable(*index),
            Collection::Union { left, right } => {
                self.visit_collection(left)?;
                self.visit_collection(right)
            }
            Collection::Intersection { left, right } => {
                self.visit_collection(left)?;
                self.visit_collection(right)
            }
            Collection::Filter { collection, predicate } => {
                self.visit_collection(collection)?;
                self.visit(predicate)
            }
            Collection::Map { lambda, collection } => {
                self.visit(&lambda.body)?;
                self.visit_collection(collection)
            }
        }
    }

    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error>;
    fn visit_collection_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error>;
}

/// Mutable visitor trait for AST transformation
/// 
/// This trait allows modifying AST nodes during traversal.
/// Each visit method returns a potentially modified AST node.
pub trait ASTMutVisitor<T: Scalar + Clone> {
    type Error;

    /// Visit and potentially transform any AST node
    fn visit_mut(&mut self, expr: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        match expr {
            ASTRepr::Constant(value) => self.visit_constant_mut(value),
            ASTRepr::Variable(index) => self.visit_variable_mut(index),
            ASTRepr::BoundVar(index) => self.visit_bound_var_mut(index),
            ASTRepr::Add(left, right) => self.visit_add_mut(*left, *right),
            ASTRepr::Sub(left, right) => self.visit_sub_mut(*left, *right),
            ASTRepr::Mul(left, right) => self.visit_mul_mut(*left, *right),
            ASTRepr::Div(left, right) => self.visit_div_mut(*left, *right),
            ASTRepr::Pow(base, exp) => self.visit_pow_mut(*base, *exp),
            ASTRepr::Neg(inner) => self.visit_neg_mut(*inner),
            ASTRepr::Sin(inner) => self.visit_sin_mut(*inner),
            ASTRepr::Cos(inner) => self.visit_cos_mut(*inner),
            ASTRepr::Ln(inner) => self.visit_ln_mut(*inner),
            ASTRepr::Exp(inner) => self.visit_exp_mut(*inner),
            ASTRepr::Sqrt(inner) => self.visit_sqrt_mut(*inner),
            ASTRepr::Sum(collection) => self.visit_sum_mut(*collection),
            ASTRepr::Lambda(lambda) => self.visit_lambda_mut(*lambda),
            ASTRepr::Let(binding_id, expr, body) => self.visit_let_mut(binding_id, *expr, *body),
        }
    }

    // Leaf nodes - default implementations return unchanged
    fn visit_constant_mut(&mut self, value: T) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Constant(value))
    }

    fn visit_variable_mut(&mut self, index: usize) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Variable(index))
    }

    fn visit_bound_var_mut(&mut self, index: usize) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::BoundVar(index))
    }

    // Binary operations - default implementations recursively transform children
    fn visit_add_mut(&mut self, left: ASTRepr<T>, right: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let left_transformed = self.visit_mut(left)?;
        let right_transformed = self.visit_mut(right)?;
        Ok(ASTRepr::Add(Box::new(left_transformed), Box::new(right_transformed)))
    }

    fn visit_sub_mut(&mut self, left: ASTRepr<T>, right: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let left_transformed = self.visit_mut(left)?;
        let right_transformed = self.visit_mut(right)?;
        Ok(ASTRepr::Sub(Box::new(left_transformed), Box::new(right_transformed)))
    }

    fn visit_mul_mut(&mut self, left: ASTRepr<T>, right: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let left_transformed = self.visit_mut(left)?;
        let right_transformed = self.visit_mut(right)?;
        Ok(ASTRepr::Mul(Box::new(left_transformed), Box::new(right_transformed)))
    }

    fn visit_div_mut(&mut self, left: ASTRepr<T>, right: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let left_transformed = self.visit_mut(left)?;
        let right_transformed = self.visit_mut(right)?;
        Ok(ASTRepr::Div(Box::new(left_transformed), Box::new(right_transformed)))
    }

    fn visit_pow_mut(&mut self, base: ASTRepr<T>, exp: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let base_transformed = self.visit_mut(base)?;
        let exp_transformed = self.visit_mut(exp)?;
        Ok(ASTRepr::Pow(Box::new(base_transformed), Box::new(exp_transformed)))
    }

    // Unary operations - default implementations recursively transform child
    fn visit_neg_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;
        Ok(ASTRepr::Neg(Box::new(inner_transformed)))
    }

    fn visit_sin_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;
        Ok(ASTRepr::Sin(Box::new(inner_transformed)))
    }

    fn visit_cos_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;
        Ok(ASTRepr::Cos(Box::new(inner_transformed)))
    }

    fn visit_ln_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;
        Ok(ASTRepr::Ln(Box::new(inner_transformed)))
    }

    fn visit_exp_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;
        Ok(ASTRepr::Exp(Box::new(inner_transformed)))
    }

    fn visit_sqrt_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let inner_transformed = self.visit_mut(inner)?;
        Ok(ASTRepr::Sqrt(Box::new(inner_transformed)))
    }

    // Complex structures
    fn visit_sum_mut(&mut self, collection: Collection<T>) -> Result<ASTRepr<T>, Self::Error> {
        let transformed_collection = self.visit_collection_mut(collection)?;
        Ok(ASTRepr::Sum(Box::new(transformed_collection)))
    }

    fn visit_lambda_mut(&mut self, lambda: Lambda<T>) -> Result<ASTRepr<T>, Self::Error> {
        let transformed_body = self.visit_mut(*lambda.body)?;
        let transformed_lambda = Lambda {
            var_indices: lambda.var_indices,
            body: Box::new(transformed_body),
        };
        Ok(ASTRepr::Lambda(Box::new(transformed_lambda)))
    }

    fn visit_let_mut(&mut self, binding_id: usize, expr: ASTRepr<T>, body: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        let expr_transformed = self.visit_mut(expr)?;
        let body_transformed = self.visit_mut(body)?;
        Ok(ASTRepr::Let(binding_id, Box::new(expr_transformed), Box::new(body_transformed)))
    }

    // Collection transformation
    fn visit_collection_mut(&mut self, collection: Collection<T>) -> Result<Collection<T>, Self::Error> {
        match collection {
            Collection::Empty => Ok(Collection::Empty),
            Collection::Singleton(expr) => {
                let transformed = self.visit_mut(*expr)?;
                Ok(Collection::Singleton(Box::new(transformed)))
            }
            Collection::Range { start, end } => {
                let start_transformed = self.visit_mut(*start)?;
                let end_transformed = self.visit_mut(*end)?;
                Ok(Collection::Range {
                    start: Box::new(start_transformed),
                    end: Box::new(end_transformed),
                })
            }
            Collection::Variable(index) => Ok(Collection::Variable(index)),
            Collection::Union { left, right } => {
                let left_transformed = self.visit_collection_mut(*left)?;
                let right_transformed = self.visit_collection_mut(*right)?;
                Ok(Collection::Union {
                    left: Box::new(left_transformed),
                    right: Box::new(right_transformed),
                })
            }
            Collection::Intersection { left, right } => {
                let left_transformed = self.visit_collection_mut(*left)?;
                let right_transformed = self.visit_collection_mut(*right)?;
                Ok(Collection::Intersection {
                    left: Box::new(left_transformed),
                    right: Box::new(right_transformed),
                })
            }
            Collection::Filter { collection, predicate } => {
                let collection_transformed = self.visit_collection_mut(*collection)?;
                let predicate_transformed = self.visit_mut(*predicate)?;
                Ok(Collection::Filter {
                    collection: Box::new(collection_transformed),
                    predicate: Box::new(predicate_transformed),
                })
            }
            Collection::Map { lambda, collection } => {
                let body_transformed = self.visit_mut(*lambda.body)?;
                let collection_transformed = self.visit_collection_mut(*collection)?;
                let transformed_lambda = Lambda {
                    var_indices: lambda.var_indices,
                    body: Box::new(body_transformed),
                };
                Ok(Collection::Map {
                    lambda: Box::new(transformed_lambda),
                    collection: Box::new(collection_transformed),
                })
            }
        }
    }
}

/// Convenience function for applying an immutable visitor
pub fn visit_ast<T, V>(expr: &ASTRepr<T>, visitor: &mut V) -> Result<V::Output, V::Error>
where
    T: Scalar,
    V: ASTVisitor<T>,
{
    visitor.visit(expr)
}

/// Convenience function for applying a mutable visitor
pub fn visit_ast_mut<T, V>(expr: ASTRepr<T>, visitor: &mut V) -> Result<ASTRepr<T>, V::Error>
where
    T: Scalar + Clone,
    V: ASTMutVisitor<T>,
{
    visitor.visit_mut(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example visitor that counts nodes
    struct NodeCounter {
        count: usize,
    }

    impl ASTVisitor<f64> for NodeCounter {
        type Output = ();
        type Error = ();

        fn visit_constant(&mut self, _value: &f64) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_collection_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }
    }

    // Example transformer that replaces constants with variables
    struct ConstantToVariable {
        var_index: usize,
    }

    impl ASTMutVisitor<f64> for ConstantToVariable {
        type Error = ();

        fn visit_constant_mut(&mut self, _value: f64) -> Result<ASTRepr<f64>, Self::Error> {
            Ok(ASTRepr::Variable(self.var_index))
        }
    }

    #[test]
    fn test_node_counter() {
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Constant(1.0)),
            Box::new(ASTRepr::Variable(0)),
        );

        let mut counter = NodeCounter { count: 0 };
        visit_ast(&expr, &mut counter).unwrap();
        assert_eq!(counter.count, 2);
    }

    #[test]
    fn test_constant_transformer() {
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Constant(1.0)),
            Box::new(ASTRepr::Constant(2.0)),
        );

        let mut transformer = ConstantToVariable { var_index: 42 };
        let result = visit_ast_mut(expr, &mut transformer).unwrap();

        match result {
            ASTRepr::Add(left, right) => {
                assert!(matches!(*left, ASTRepr::Variable(42)));
                assert!(matches!(*right, ASTRepr::Variable(42)));
            }
            _ => panic!("Expected Add node"),
        }
    }
} 