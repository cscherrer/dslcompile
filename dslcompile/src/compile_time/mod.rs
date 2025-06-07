//! Compile-Time Mathematical Expression System
//!
//! This module provides a compile-time mathematical expression system that leverages
//! Rust's type system to perform optimizations at compile time with perfect composability.
//!
//! ## Enhanced Scoped Variables System (NEW)
//!
//! The enhanced scoped variables system provides:
//! - **Type-safe composition**: Variable scopes prevent collisions at compile time
//! - **Zero runtime overhead**: All scope resolution happens at compile time
//! - **HList integration**: Variadic heterogeneous inputs without MAX_VARS limitations
//! - **Native performance**: Matches native Rust performance
//! - **Perfect composability**: Build mathematical libraries without variable index conflicts
//!
//! ## Example
//!
//! ```rust
//! use dslcompile::prelude::*;
//! use frunk::hlist;
//!
//! let mut ctx = EnhancedContext::new();
//!
//! // Define f(x, y) = x² + 2y in scope 0
//! let f = ctx.new_scope(|scope| {
//!     let (x, scope) = scope.auto_var::<f64>();
//!     let (y, _scope) = scope.auto_var::<f64>();
//!     x.clone() * x + scope.constant(2.0) * y
//! });
//!
//! // Evaluate with HList inputs - zero overhead
//! let result = f.eval_hlist(hlist![3.0, 4.0]); // 3² + 2*4 = 17
//! assert_eq!(result, 17.0);
//! ```
//!
//! ## Legacy Scoped Variables System
//!
//! The original scoped variables system provides:
//! - **Type-safe composition**: Variable scopes prevent collisions at compile time
//! - **Zero runtime overhead**: All scope resolution happens at compile time
//! - **Automatic variable remapping**: Functions compose seamlessly without manual index management
//! - **Perfect composability**: Build mathematical libraries without variable index conflicts
//!
//! ## Example
//!
//! ```rust
//! use dslcompile::prelude::*;
//!
//! let mut builder = Context::new();
//!
//! // Define f(x) = x² in scope 0
//! let f = builder.new_scope(|scope| {
//!     let (x, _scope) = scope.auto_var();
//!     x.clone().mul(x)
//! });
//!
//! // Advance to next scope
//! let mut builder = builder.next();
//!
//! // Define g(y) = 2y in scope 1 (no collision!)
//! let g = builder.new_scope(|scope| {
//!     let (y, scope) = scope.auto_var();
//!     y.mul(scope.constant(2.0))
//! });
//!
//! // Perfect composition with automatic variable remapping
//! let composed = compose(f, g);
//! let combined = composed.add(); // h(x,y) = x² + 2y
//!
//! let result = combined.eval(&[3.0, 4.0]);
//! assert_eq!(result, 17.0); // 3² + 2*4 = 9 + 8 = 17
//! ```

pub mod enhanced_scoped;
pub mod macro_expressions;

// CLEAN ARCHITECTURE: Only one compile-time context needed
// StaticContext = EnhancedContext (renamed for clarity)
pub use enhanced_scoped::{
    EnhancedContext as StaticContext,
    EnhancedScopeBuilder as StaticScopeBuilder,
    EnhancedVar as StaticVar,
    EnhancedConst as StaticConst,
    EnhancedAdd as StaticAdd,
    EnhancedMul as StaticMul,
    EnhancedExpr as StaticExpr,
    HListEval, HListStorage, IntoHListEvaluable,
    enhanced_add as static_add,
    enhanced_mul as static_mul,
};

// LEGACY COMPATIBILITY - for existing code that imports these
// TODO: Remove these in next major version
#[deprecated(note = "Use StaticContext instead - provides automatic scoping + heterogeneous support")]
pub struct Context<T, const SCOPE: usize>(std::marker::PhantomData<T>);

#[deprecated(note = "Use StaticVar instead")]  
pub struct ScopedVar<T, const ID: usize, const SCOPE: usize>(std::marker::PhantomData<T>);

#[deprecated(note = "Use StaticExpr instead")]
pub struct ScopedMathExpr<T, const SCOPE: usize>(std::marker::PhantomData<T>);

#[deprecated(note = "Use Vec<T> instead")]
pub struct ScopedVarArray<T, const SIZE: usize>(std::marker::PhantomData<T>);

#[deprecated(note = "Use StaticContext composition instead")]
pub fn compose<T>(_f: T, _g: T) -> T {
    unimplemented!("Use StaticContext composition instead")
}

// Re-export macro expressions
pub use macro_expressions::*;

// ============================================================================
// LEGACY INTERFACE FOR PROCEDURAL MACRO
// ============================================================================

/// Simple variable type for procedural macro interface
#[derive(Debug, Clone)]
pub struct CompileTimeVar<const ID: usize>;

/// Simple constant type for procedural macro interface  
#[derive(Debug, Clone)]
pub struct CompileTimeConst {
    value: f64,
}

impl<const ID: usize> CompileTimeVar<ID> {
    /// Add operation
    pub fn add<T>(self, other: T) -> CompileTimeAdd<Self, T> {
        CompileTimeAdd {
            left: self,
            right: other,
        }
    }

    /// Subtract operation
    pub fn sub<T>(self, other: T) -> CompileTimeSub<Self, T> {
        CompileTimeSub {
            left: self,
            right: other,
        }
    }

    /// Multiply operation
    pub fn mul<T>(self, other: T) -> CompileTimeMul<Self, T> {
        CompileTimeMul {
            left: self,
            right: other,
        }
    }

    /// Divide operation
    pub fn div<T>(self, other: T) -> CompileTimeDiv<Self, T> {
        CompileTimeDiv {
            left: self,
            right: other,
        }
    }

    /// Power operation
    pub fn pow<T>(self, other: T) -> CompileTimePow<Self, T> {
        CompileTimePow {
            base: self,
            exp: other,
        }
    }

    /// Sine operation
    #[must_use]
    pub fn sin(self) -> CompileTimeSin<Self> {
        CompileTimeSin { inner: self }
    }

    /// Cosine operation
    #[must_use]
    pub fn cos(self) -> CompileTimeCos<Self> {
        CompileTimeCos { inner: self }
    }

    /// Exponential operation
    #[must_use]
    pub fn exp(self) -> CompileTimeExp<Self> {
        CompileTimeExp { inner: self }
    }

    /// Natural logarithm operation
    #[must_use]
    pub fn ln(self) -> CompileTimeLn<Self> {
        CompileTimeLn { inner: self }
    }
}

impl CompileTimeConst {
    /// Add operation
    pub fn add<T>(self, other: T) -> CompileTimeAdd<Self, T> {
        CompileTimeAdd {
            left: self,
            right: other,
        }
    }

    /// Subtract operation
    pub fn sub<T>(self, other: T) -> CompileTimeSub<Self, T> {
        CompileTimeSub {
            left: self,
            right: other,
        }
    }

    /// Multiply operation
    pub fn mul<T>(self, other: T) -> CompileTimeMul<Self, T> {
        CompileTimeMul {
            left: self,
            right: other,
        }
    }

    /// Divide operation
    pub fn div<T>(self, other: T) -> CompileTimeDiv<Self, T> {
        CompileTimeDiv {
            left: self,
            right: other,
        }
    }

    /// Power operation
    pub fn pow<T>(self, other: T) -> CompileTimePow<Self, T> {
        CompileTimePow {
            base: self,
            exp: other,
        }
    }

    /// Sine operation
    #[must_use]
    pub fn sin(self) -> CompileTimeSin<Self> {
        CompileTimeSin { inner: self }
    }

    /// Cosine operation
    #[must_use]
    pub fn cos(self) -> CompileTimeCos<Self> {
        CompileTimeCos { inner: self }
    }

    /// Exponential operation
    #[must_use]
    pub fn exp(self) -> CompileTimeExp<Self> {
        CompileTimeExp { inner: self }
    }

    /// Natural logarithm operation
    #[must_use]
    pub fn ln(self) -> CompileTimeLn<Self> {
        CompileTimeLn { inner: self }
    }
}

// Operation types for procedural macro
#[derive(Debug, Clone)]
pub struct CompileTimeAdd<L, R> {
    left: L,
    right: R,
}

#[derive(Debug, Clone)]
pub struct CompileTimeSub<L, R> {
    left: L,
    right: R,
}

#[derive(Debug, Clone)]
pub struct CompileTimeMul<L, R> {
    left: L,
    right: R,
}

#[derive(Debug, Clone)]
pub struct CompileTimeDiv<L, R> {
    left: L,
    right: R,
}

#[derive(Debug, Clone)]
pub struct CompileTimePow<B, E> {
    base: B,
    exp: E,
}

#[derive(Debug, Clone)]
pub struct CompileTimeSin<T> {
    inner: T,
}

#[derive(Debug, Clone)]
pub struct CompileTimeCos<T> {
    inner: T,
}

#[derive(Debug, Clone)]
pub struct CompileTimeExp<T> {
    inner: T,
}

#[derive(Debug, Clone)]
pub struct CompileTimeLn<T> {
    inner: T,
}

// Implement operations for all operation types
macro_rules! impl_operations {
    ($type:ident) => {
        impl<L, R> $type<L, R> {
            pub fn add<T>(self, other: T) -> CompileTimeAdd<Self, T> {
                CompileTimeAdd {
                    left: self,
                    right: other,
                }
            }

            pub fn sub<T>(self, other: T) -> CompileTimeSub<Self, T> {
                CompileTimeSub {
                    left: self,
                    right: other,
                }
            }

            pub fn mul<T>(self, other: T) -> CompileTimeMul<Self, T> {
                CompileTimeMul {
                    left: self,
                    right: other,
                }
            }

            pub fn div<T>(self, other: T) -> CompileTimeDiv<Self, T> {
                CompileTimeDiv {
                    left: self,
                    right: other,
                }
            }

            pub fn pow<T>(self, other: T) -> CompileTimePow<Self, T> {
                CompileTimePow {
                    base: self,
                    exp: other,
                }
            }

            pub fn sin(self) -> CompileTimeSin<Self> {
                CompileTimeSin { inner: self }
            }

            pub fn cos(self) -> CompileTimeCos<Self> {
                CompileTimeCos { inner: self }
            }

            pub fn exp(self) -> CompileTimeExp<Self> {
                CompileTimeExp { inner: self }
            }

            pub fn ln(self) -> CompileTimeLn<Self> {
                CompileTimeLn { inner: self }
            }
        }
    };
}

impl_operations!(CompileTimeAdd);
impl_operations!(CompileTimeSub);
impl_operations!(CompileTimeMul);
impl_operations!(CompileTimeDiv);

macro_rules! impl_unary_operations {
    ($type:ident) => {
        impl<T> $type<T> {
            pub fn add<U>(self, other: U) -> CompileTimeAdd<Self, U> {
                CompileTimeAdd {
                    left: self,
                    right: other,
                }
            }

            pub fn sub<U>(self, other: U) -> CompileTimeSub<Self, U> {
                CompileTimeSub {
                    left: self,
                    right: other,
                }
            }

            pub fn mul<U>(self, other: U) -> CompileTimeMul<Self, U> {
                CompileTimeMul {
                    left: self,
                    right: other,
                }
            }

            pub fn div<U>(self, other: U) -> CompileTimeDiv<Self, U> {
                CompileTimeDiv {
                    left: self,
                    right: other,
                }
            }

            pub fn pow<U>(self, other: U) -> CompileTimePow<Self, U> {
                CompileTimePow {
                    base: self,
                    exp: other,
                }
            }

            pub fn sin(self) -> CompileTimeSin<Self> {
                CompileTimeSin { inner: self }
            }

            pub fn cos(self) -> CompileTimeCos<Self> {
                CompileTimeCos { inner: self }
            }

            pub fn exp(self) -> CompileTimeExp<Self> {
                CompileTimeExp { inner: self }
            }

            pub fn ln(self) -> CompileTimeLn<Self> {
                CompileTimeLn { inner: self }
            }
        }
    };
}

impl_unary_operations!(CompileTimeSin);
impl_unary_operations!(CompileTimeCos);
impl_unary_operations!(CompileTimeExp);
impl_unary_operations!(CompileTimeLn);

// CompileTimePow needs special handling since it has 2 generic parameters
impl<B, E> CompileTimePow<B, E> {
    pub fn add<T>(self, other: T) -> CompileTimeAdd<Self, T> {
        CompileTimeAdd {
            left: self,
            right: other,
        }
    }

    pub fn sub<T>(self, other: T) -> CompileTimeSub<Self, T> {
        CompileTimeSub {
            left: self,
            right: other,
        }
    }

    pub fn mul<T>(self, other: T) -> CompileTimeMul<Self, T> {
        CompileTimeMul {
            left: self,
            right: other,
        }
    }

    pub fn div<T>(self, other: T) -> CompileTimeDiv<Self, T> {
        CompileTimeDiv {
            left: self,
            right: other,
        }
    }

    pub fn pow<T>(self, other: T) -> CompileTimePow<Self, T> {
        CompileTimePow {
            base: self,
            exp: other,
        }
    }

    pub fn sin(self) -> CompileTimeSin<Self> {
        CompileTimeSin { inner: self }
    }

    pub fn cos(self) -> CompileTimeCos<Self> {
        CompileTimeCos { inner: self }
    }

    pub fn exp(self) -> CompileTimeExp<Self> {
        CompileTimeExp { inner: self }
    }

    pub fn ln(self) -> CompileTimeLn<Self> {
        CompileTimeLn { inner: self }
    }
}

/// Create a variable for procedural macro interface
#[must_use]
pub fn var<const ID: usize>() -> CompileTimeVar<ID> {
    CompileTimeVar
}

/// Create a constant for procedural macro interface
#[must_use]
pub fn constant(value: f64) -> CompileTimeConst {
    CompileTimeConst { value }
}

// Re-export for procedural macro
pub use dslcompile_macros::optimize_compile_time;

// REMOVED: Legacy type aliases - use StaticContext instead
// All functionality consolidated into StaticContext with automatic scoping + HList support

/// Trait for compile-time expression evaluation
pub trait CompileTimeEval<T> {
    /// Evaluate the expression with the given variable values
    fn eval(&self, vars: &[T]) -> T;

    /// Convert to AST representation for analysis
    fn to_ast(&self) -> crate::ast::ASTRepr<T>;
}
