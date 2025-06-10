//! Mathematical Functions for DSLCompile Expression Types
//!
//! This module provides mathematical function implementations for both VariableExpr
//! and TypedBuilderExpr types, including transcendental functions like sin, cos, ln, exp,
//! sqrt, and power operations.
//!
//! ## Key Components
//!
//! - Transcendental functions: sin, cos, ln, exp, sqrt
//! - Power operations: pow with proper AST construction
//! - Type-safe implementations for Float types
//! - Automatic conversion between VariableExpr and TypedBuilderExpr

use crate::ast::{
    Scalar,
    ast_repr::ASTRepr,
    runtime::expression_builder::{TypedBuilderExpr, VariableExpr, ScalarFloat},
};
use num_traits::FromPrimitive;

// ============================================================================
// MATHEMATICAL FUNCTIONS FOR VariableExpr
// ============================================================================

/// Mathematical functions for VariableExpr with automatic conversion to TypedBuilderExpr
impl<T> VariableExpr<T>
where
    T: ScalarFloat,
{
    /// Sine function
    ///
    /// Converts the VariableExpr to TypedBuilderExpr and applies sine function.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let sin_x = x.sin();
    /// ```
    pub fn sin(self) -> TypedBuilderExpr<T> {
        self.into_expr().sin()
    }

    /// Cosine function
    ///
    /// Converts the VariableExpr to TypedBuilderExpr and applies cosine function.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let cos_x = x.cos();
    /// ```
    pub fn cos(self) -> TypedBuilderExpr<T> {
        self.into_expr().cos()
    }

    /// Natural logarithm
    ///
    /// Converts the VariableExpr to TypedBuilderExpr and applies natural logarithm.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let ln_x = x.ln();
    /// ```
    pub fn ln(self) -> TypedBuilderExpr<T> {
        self.into_expr().ln()
    }

    /// Exponential function
    ///
    /// Converts the VariableExpr to TypedBuilderExpr and applies exponential function.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let exp_x = x.exp();
    /// ```
    pub fn exp(self) -> TypedBuilderExpr<T> {
        self.into_expr().exp()
    }
}

/// Square root function for VariableExpr (requires FromPrimitive for 0.5 conversion)
impl<T> VariableExpr<T>
where
    T: ScalarFloat + FromPrimitive,
{
    /// Square root
    ///
    /// Converts the VariableExpr to TypedBuilderExpr and applies square root function.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let sqrt_x = x.sqrt();
    /// ```
    pub fn sqrt(self) -> TypedBuilderExpr<T> {
        self.into_expr().sqrt()
    }

    /// Power function
    ///
    /// Converts the VariableExpr to TypedBuilderExpr and applies power operation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let y = ctx.var();
    /// let x_pow_y = x.pow(y.into_expr());
    /// ```
    pub fn pow(self, exp: TypedBuilderExpr<T>) -> TypedBuilderExpr<T> {
        self.into_expr().pow(exp)
    }
}

// ============================================================================
// MATHEMATICAL FUNCTIONS FOR TypedBuilderExpr
// ============================================================================

/// Transcendental functions for TypedBuilderExpr with Float types
///
/// These implementations create the appropriate AST nodes for mathematical functions,
/// enabling symbolic computation and code generation.
impl<T: ScalarFloat> TypedBuilderExpr<T> {
    /// Sine function
    ///
    /// Creates a sine AST node for symbolic computation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var().into_expr();
    /// let sin_x = x.sin();
    /// ```
    pub fn sin(self) -> Self {
        Self::new(self.ast.sin(), self.registry)
    }

    /// Cosine function
    ///
    /// Creates a cosine AST node for symbolic computation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var().into_expr();
    /// let cos_x = x.cos();
    /// ```
    pub fn cos(self) -> Self {
        Self::new(self.ast.cos(), self.registry)
    }

    /// Natural logarithm
    ///
    /// Creates a natural logarithm AST node for symbolic computation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var().into_expr();
    /// let ln_x = x.ln();
    /// ```
    pub fn ln(self) -> Self {
        Self::new(self.ast.ln(), self.registry)
    }

    /// Exponential function
    ///
    /// Creates an exponential AST node for symbolic computation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var().into_expr();
    /// let exp_x = x.exp();
    /// ```
    pub fn exp(self) -> Self {
        Self::new(self.ast.exp(), self.registry)
    }
}

/// Square root and power functions for TypedBuilderExpr (requires FromPrimitive for sqrt)
impl<T: ScalarFloat + FromPrimitive> TypedBuilderExpr<T> {
    /// Square root
    ///
    /// Creates a square root AST node for symbolic computation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var().into_expr();
    /// let sqrt_x = x.sqrt();
    /// ```
    pub fn sqrt(self) -> Self {
        Self::new(self.ast.sqrt(), self.registry)
    }

    /// Power function
    ///
    /// Creates a power AST node for symbolic computation.
    ///
    /// # Example
    /// ```
    /// # use dslcompile::ast::runtime::expression_builder::DynamicContext;
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var().into_expr();
    /// let y = ctx.var().into_expr();
    /// let x_pow_y = x.pow(y);
    /// ```
    pub fn pow(self, exp: Self) -> Self {
        Self::new(
            ASTRepr::Pow(Box::new(self.ast), Box::new(exp.ast)),
            self.registry,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::runtime::expression_builder::DynamicContext;

    #[test]
    fn test_variable_expr_math_functions() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();

        // Test that mathematical functions convert VariableExpr to TypedBuilderExpr
        let sin_x = x.clone().sin();
        let cos_x = x.clone().cos();
        let ln_x = x.clone().ln();
        let exp_x = x.clone().exp();
        let sqrt_x = x.clone().sqrt();

        // These should all be TypedBuilderExpr instances
        assert!(matches!(sin_x.as_ast(), ASTRepr::Sin(_)));
        assert!(matches!(cos_x.as_ast(), ASTRepr::Cos(_)));
        assert!(matches!(ln_x.as_ast(), ASTRepr::Ln(_)));
        assert!(matches!(exp_x.as_ast(), ASTRepr::Exp(_)));
        // sqrt is implemented as x^0.5, so it should be a Pow expression
        assert!(matches!(sqrt_x.as_ast(), ASTRepr::Pow(_, _)));
    }

    #[test]
    fn test_typed_builder_expr_math_functions() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var().into_expr();
        let y = ctx.var().into_expr();

        // Test mathematical functions on TypedBuilderExpr
        let sin_x = x.clone().sin();
        let cos_x = x.clone().cos();
        let ln_x = x.clone().ln();
        let exp_x = x.clone().exp();
        let sqrt_x = x.clone().sqrt();
        let x_pow_y = x.pow(y);

        // Verify AST structure
        assert!(matches!(sin_x.as_ast(), ASTRepr::Sin(_)));
        assert!(matches!(cos_x.as_ast(), ASTRepr::Cos(_)));
        assert!(matches!(ln_x.as_ast(), ASTRepr::Ln(_)));
        assert!(matches!(exp_x.as_ast(), ASTRepr::Exp(_)));
        // sqrt is implemented as x^0.5, so it should be a Pow expression
        assert!(matches!(sqrt_x.as_ast(), ASTRepr::Pow(_, _)));
        assert!(matches!(x_pow_y.as_ast(), ASTRepr::Pow(_, _)));
    }

    #[test]
    fn test_power_function_composition() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var().into_expr();
        let two = ctx.constant(2.0);

        // Test x^2
        let x_squared = x.pow(two);

        if let ASTRepr::Pow(base, exp) = x_squared.as_ast() {
            assert!(matches!(&**base, ASTRepr::Variable(_)));
            assert!(matches!(&**exp, ASTRepr::Constant(_)));
        } else {
            panic!("Expected Pow AST node");
        }
    }

    #[test]
    fn test_function_chaining() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();

        // Test chaining: sin(ln(x))
        let result = x.ln().sin();

        if let ASTRepr::Sin(inner) = result.as_ast() {
            assert!(matches!(&**inner, ASTRepr::Ln(_)));
        } else {
            panic!("Expected Sin(Ln(_)) AST structure");
        }
    }
}
