//! # `MathJIT`: High-Performance Symbolic Mathematics
//!
//! `MathJIT` is a high-performance symbolic mathematics library built around the final tagless
//! approach, providing zero-cost abstractions, egglog optimization, and Cranelift JIT compilation.
//!
//! ## Core Design Principles
//!
//! 1. **Final Tagless Architecture**: Zero-cost abstractions using Generic Associated Types (GATs)
//! 2. **Multiple Interpreters**: Same expression definition, multiple evaluation strategies
//! 3. **High Performance**: JIT compilation for native speed execution
//! 4. **Symbolic Optimization**: Egglog-powered expression optimization
//! 5. **Type Safety**: Compile-time guarantees without runtime overhead
//!
//! ## Quick Start
//!
//! ```rust
//! use mathjit::final_tagless::{MathExpr, DirectEval};
//!
//! // Define a polymorphic mathematical expression
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
//!
//! // Evaluate directly
//! let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));
//! assert_eq!(result, 15.0); // 2(4) + 3(2) + 1 = 15
//! ```
//!
//! ## JIT Compilation
//!
//! ```rust
//! # #[cfg(feature = "jit")]
//! # {
//! use mathjit::final_tagless::{MathExpr, DirectEval};
//!
//! # fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
//! # where E::Repr<f64>: Clone,
//! # { E::add(E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::mul(E::constant(3.0), x)), E::constant(1.0)) }
//! // For now, use DirectEval (JIT coming soon)
//! let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));
//! assert_eq!(result, 15.0);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]
#![allow(unstable_name_collisions)]

// Core modules
pub mod error;
pub mod final_tagless;

// JIT compilation module (optional)
#[cfg(feature = "jit")]
pub mod jit;

// Re-export commonly used types
pub use error::{MathJITError, Result};
pub use final_tagless::{
    polynomial, DirectEval, MathExpr, NumericType, PrettyPrint, StatisticalExpr,
};

// JIT support
#[cfg(feature = "jit")]
pub use final_tagless::{JITEval, JITMathExpr, JITRepr};
#[cfg(feature = "jit")]
pub use jit::{CompilationStats, JITCompiler, JITFunction, JITSignature};

// Re-export numeric trait for convenience
pub use num_traits::Float;

/// Convenience module for common mathematical operations
pub mod prelude {
    pub use crate::final_tagless::{
        polynomial, DirectEval, MathExpr, NumericType, PrettyPrint, StatisticalExpr,
    };

    // JIT support
    #[cfg(feature = "jit")]
    pub use crate::final_tagless::{JITEval, JITMathExpr, JITRepr};
    #[cfg(feature = "jit")]
    pub use crate::jit::{CompilationStats, JITCompiler, JITFunction, JITSignature};

    pub use crate::error::{MathJITError, Result};
}

/// Ergonomic wrapper for final tagless expressions with operator overloading
pub mod expr {
    use crate::final_tagless::MathExpr;
    use std::marker::PhantomData;

    /// Wrapper type that enables operator overloading for final tagless expressions
    pub struct Expr<E: MathExpr, T> {
        pub(crate) repr: E::Repr<T>,
        _phantom: PhantomData<E>,
    }

    impl<E: MathExpr, T> Expr<E, T> {
        /// Create a new expression wrapper
        pub fn new(repr: E::Repr<T>) -> Self {
            Self {
                repr,
                _phantom: PhantomData,
            }
        }

        /// Extract the underlying representation
        pub fn into_repr(self) -> E::Repr<T> {
            self.repr
        }

        /// Get a reference to the underlying representation
        pub fn as_repr(&self) -> &E::Repr<T> {
            &self.repr
        }

        /// Create a variable expression
        #[must_use]
        pub fn var(name: &str) -> Self
        where
            E::Repr<T>: Clone,
            T: crate::final_tagless::NumericType,
        {
            Self::new(E::var(name))
        }

        /// Create a constant expression
        pub fn constant(value: T) -> Self
        where
            T: crate::final_tagless::NumericType,
        {
            Self::new(E::constant(value))
        }
    }

    // Operator overloading implementations will be added here
    // This provides ergonomic syntax like: x + y * constant(2.0)
}

#[cfg(test)]
mod tests {

    use crate::final_tagless::{DirectEval, MathExpr};

    #[test]
    fn test_basic_final_tagless() {
        fn simple_expr<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
            E::add(E::mul(E::constant(2.0), x), E::constant(1.0))
        }

        let result = simple_expr::<DirectEval>(DirectEval::var("x", 5.0));
        assert_eq!(result, 11.0); // 2*5 + 1 = 11
    }

    // JIT tests will be added when JIT support is implemented
}
