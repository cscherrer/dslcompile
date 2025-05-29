//! Final Tagless Approach for Symbolic Mathematical Expressions
//!
//! This module implements the final tagless approach to solve the expression problem in symbolic
//! mathematics. The final tagless approach uses traits with Generic Associated Types (GATs) to
//! represent mathematical operations, enabling both easy extension of operations and interpreters
//! without modifying existing code.
//!
//! # Technical Motivation
//!
//! Traditional approaches to symbolic mathematics face the expression problem: adding new operations
//! requires modifying existing interpreter code, while adding new interpreters requires modifying
//! existing operation definitions. The final tagless approach solves this by:
//!
//! 1. **Parameterizing representation types**: Operations are defined over abstract representation
//!    types `Repr<T>`, allowing different interpreters to use different concrete representations
//! 2. **Trait-based extensibility**: New operations can be added via trait extension without
//!    modifying existing code
//! 3. **Zero intermediate representation**: Expressions compile directly to target representations
//!    without building intermediate ASTs
//!
//! # Architecture
//!
//! ## Core Traits
//!
//! - **`MathExpr`**: Defines basic mathematical operations (arithmetic, transcendental functions)
//! - **`StatisticalExpr`**: Extends `MathExpr` with statistical functions (logistic, softplus)
//! - **`NumericType`**: Helper trait bundling common numeric type requirements
//!
//! ## Interpreters
//!
//! - **`DirectEval`**: Immediate evaluation using native Rust operations (`type Repr<T> = T`)
//! - **`PrettyPrint`**: String representation generation (`type Repr<T> = String`)
//! - **`ASTEval`**: AST construction for JIT compilation (`type Repr<T> = ASTRepr<T>`)
//!
//! # Usage Patterns
//!
//! ## Polymorphic Expression Definition
//!
//! Define mathematical expressions that work with any interpreter:
//!
//! ```rust
//! use mathcompile::final_tagless::*;
//!
//! // Define a quadratic function: 2x² + 3x + 1
//! fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! where
//!     E::Repr<f64>: Clone,
//! {
//!     let a = E::constant(2.0);
//!     let b = E::constant(3.0);
//!     let c = E::constant(1.0);
//!     
//!     E::add(
//!         E::add(
//!             E::mul(a, E::pow(x.clone(), E::constant(2.0))),
//!             E::mul(b, x)
//!         ),
//!         c
//!     )
//! }
//! ```
//!
//! ## Direct Evaluation
//!
//! Evaluate expressions immediately using native Rust operations:
//!
//! ```rust
//! # use mathcompile::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));
//! assert_eq!(result, 15.0); // 2(4) + 3(2) + 1 = 15
//! ```
//!
//! ## Pretty Printing
//!
//! Generate human-readable mathematical notation:
//!
//! ```rust
//! # use mathcompile::final_tagless::*;
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
//! println!("Expression: {}", pretty);
//! // Output: "((2 * (x ^ 2)) + (3 * x)) + 1"
//! ```
//!
//! # Extension Example
//!
//! Adding new operations requires only trait extension:
//!
//! ```rust
//! use mathcompile::final_tagless::*;
//! use num_traits::Float;
//!
//! // Extend with hyperbolic functions
//! trait HyperbolicExpr: MathExpr {
//!     fn tanh<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T>
//!     where
//!         Self::Repr<T>: Clone,
//!     {
//!         let exp_x = Self::exp(x.clone());
//!         let exp_neg_x = Self::exp(Self::neg(x));
//!         let numerator = Self::sub(exp_x.clone(), exp_neg_x.clone());
//!         let denominator = Self::add(exp_x, exp_neg_x);
//!         Self::div(numerator, denominator)
//!     }
//! }
//!
//! // Automatically works with all existing interpreters
//! impl HyperbolicExpr for DirectEval {}
//! impl HyperbolicExpr for PrettyPrint {}
//! ```

// Core traits and types
pub mod traits;
pub use traits::*;

// AST representation and utilities (now in crate::ast)
pub use crate::ast::ASTRepr;

// Interpreters
pub mod interpreters;
pub use interpreters::{ASTEval, DirectEval, PrettyPrint};

// Polynomial utilities
pub mod polynomial;

// Variable management
pub mod variables;
pub use variables::{
    ExpressionBuilder, VariableRegistry, clear_global_registry, create_variable_map,
    get_variable_index, get_variable_name, register_variable,
};

// Summation infrastructure (placeholder for future expansion)
// These types are referenced in traits but will be fully implemented later
pub use traits::{RangeType, SummandFunction};

/// Simple integer range for summations
///
/// Represents ranges like 1..=n, 0..=100, etc. This is the most common
/// type of range used in mathematical summations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntRange {
    /// Start of the range (inclusive)
    pub start: i64,
    /// End of the range (inclusive)
    pub end: i64,
}

impl IntRange {
    /// Create a new integer range
    #[must_use]
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    /// Create a range from 1 to n (common mathematical convention)
    #[must_use]
    pub fn one_to_n(n: i64) -> Self {
        Self::new(1, n)
    }

    /// Create a range from 0 to n-1 (common programming convention)
    #[must_use]
    pub fn zero_to_n_minus_one(n: i64) -> Self {
        Self::new(0, n - 1)
    }

    /// Iterate over the range values
    pub fn iter(&self) -> impl Iterator<Item = i64> {
        self.start..=self.end
    }
}

impl RangeType for IntRange {
    type IndexType = i64;

    fn start(&self) -> Self::IndexType {
        self.start
    }

    fn end(&self) -> Self::IndexType {
        self.end
    }

    fn contains(&self, value: &Self::IndexType) -> bool {
        *value >= self.start && *value <= self.end
    }

    fn len(&self) -> Self::IndexType {
        if self.end >= self.start {
            self.end - self.start + 1
        } else {
            0
        }
    }

    fn is_empty(&self) -> bool {
        self.end < self.start
    }
}

/// AST-based function for summations
#[derive(Debug, Clone)]
pub struct ASTFunction<T> {
    /// The variable name for the summation index
    pub index_var: String,
    /// The expression representing the function body
    pub body: ASTRepr<T>,
}

impl<T: NumericType> ASTFunction<T> {
    /// Create a new AST-based function
    pub fn new(index_var: &str, body: ASTRepr<T>) -> Self {
        Self {
            index_var: index_var.to_string(),
            body,
        }
    }

    /// Create a constant function: f(i) = c
    pub fn constant_func(index_var: &str, value: T) -> Self {
        Self::new(index_var, ASTRepr::Constant(value))
    }

    /// Create a linear function: f(i) = slope * i + intercept
    pub fn linear(index_var: &str, slope: T, intercept: T) -> Self
    where
        T: Clone,
    {
        let i = ASTRepr::Variable(0); // Assume index variable is at position 0
        let slope_expr = ASTRepr::Constant(slope);
        let intercept_expr = ASTRepr::Constant(intercept);
        let body = ASTRepr::Add(
            Box::new(ASTRepr::Mul(Box::new(slope_expr), Box::new(i))),
            Box::new(intercept_expr),
        );
        Self::new(index_var, body)
    }

    /// Create a power function: f(i) = i^exponent
    pub fn power(index_var: &str, exponent: T) -> Self {
        let i = ASTRepr::Variable(0); // Assume index variable is at position 0
        let exp_expr = ASTRepr::Constant(exponent);
        let body = ASTRepr::Pow(Box::new(i), Box::new(exp_expr));
        Self::new(index_var, body)
    }
}

// Placeholder implementation for SummandFunction
impl<T: NumericType + Copy> SummandFunction<T> for ASTFunction<T> {
    type Body = ASTRepr<T>;

    fn index_var(&self) -> &str {
        &self.index_var
    }

    fn body(&self) -> &Self::Body {
        &self.body
    }

    fn apply(&self, _index: T) -> Self::Body {
        // Placeholder implementation
        self.body.clone()
    }

    fn depends_on_index(&self) -> bool {
        // Placeholder implementation
        true
    }

    fn extract_independent_factors(&self) -> (Vec<Self::Body>, Self::Body) {
        // Placeholder implementation
        (vec![], self.body.clone())
    }
}
