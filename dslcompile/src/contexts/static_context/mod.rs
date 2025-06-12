//! Compile-Time Mathematical Expression System
//!
//! This module provides a compile-time mathematical expression system that leverages
//! Rust's type system to perform optimizations at compile time with perfect composability.
//!
//! ## Static Scoped Variables System (NEW)
//!
//! The static scoped variables system provides:
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
//! let mut ctx = StaticContext::new();
//!
//! // Define f(x, y) = x² + 2y in scope 0
//! let f = ctx.new_scope(|scope| {
//!     let (x, scope) = scope.auto_var::<f64>();
//!     let (y, scope) = scope.auto_var::<f64>();
//!     x.clone() * x + scope.constant(2.0) * y
//! });
//!
//! // Evaluate with HList inputs - zero overhead
//! let result = f.eval(hlist![3.0, 4.0]); // 3² + 2*4 = 17
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
//! use frunk::hlist;
//!
//! let mut ctx = StaticContext::new();
//!
//! // Define f(x, y) = x² + 2y with automatic scope management
//! let f = ctx.new_scope(|scope| {
//!     let (x, scope) = scope.auto_var::<f64>();
//!     let (y, scope) = scope.auto_var::<f64>();
//!     x.clone() * x + scope.constant(2.0) * y
//! });
//!
//! // Evaluate with zero overhead
//! let result = f.eval(hlist![3.0, 4.0]);
//! assert_eq!(result, 17.0); // 3² + 2*4 = 9 + 8 = 17
//! ```

pub mod macro_expressions;
pub mod static_scoped;

// CLEAN ARCHITECTURE: Only one compile-time context needed
pub use static_scoped::{
    HListEval, HListStorage, IntoHListEvaluable, StaticAdd, StaticConst, StaticContext, StaticExpr,
    StaticMul, StaticScopeBuilder, StaticVar, static_add, static_mul,
};

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

/// Trait for compile-time expression evaluation
pub trait CompileTimeEval<T> {
    /// Evaluate the expression with the given variable values
    fn eval(&self, vars: &[T]) -> T;

    /// Convert to AST representation for analysis
    fn to_ast(&self) -> crate::ast::ASTRepr<T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;

    #[test]
    fn test_compile_time_var_creation() {
        let var0: CompileTimeVar<0> = var::<0>();
        let var1: CompileTimeVar<1> = var::<1>();
        
        // Variables should be different types
        assert_ne!(std::any::TypeId::of::<CompileTimeVar<0>>(), std::any::TypeId::of::<CompileTimeVar<1>>());
    }

    #[test]
    fn test_compile_time_const_creation() {
        let const_pi = constant(3.14159);
        let const_e = constant(2.71828);
        
        assert_eq!(const_pi.value, 3.14159);
        assert_eq!(const_e.value, 2.71828);
    }

    #[test]
    fn test_compile_time_var_arithmetic() {
        let x = var::<0>();
        let y = var::<1>();
        let const_2 = constant(2.0);

        // Test addition
        let _add_expr = x.clone().add(y.clone());
        let _add_const = x.clone().add(const_2.clone());

        // Test subtraction
        let _sub_expr = x.clone().sub(y.clone());
        let _sub_const = x.clone().sub(const_2.clone());

        // Test multiplication
        let _mul_expr = x.clone().mul(y.clone());
        let _mul_const = x.clone().mul(const_2.clone());

        // Test division
        let _div_expr = x.clone().div(y.clone());
        let _div_const = x.clone().div(const_2.clone());

        // Test power
        let _pow_expr = x.clone().pow(y.clone());
        let _pow_const = x.clone().pow(const_2);
    }

    #[test]
    fn test_compile_time_var_transcendental() {
        let x = var::<0>();

        // Test trigonometric functions
        let _sin_x = x.clone().sin();
        let _cos_x = x.clone().cos();

        // Test exponential and logarithmic functions
        let _exp_x = x.clone().exp();
        let _ln_x = x.ln();
    }

    #[test]
    fn test_compile_time_const_arithmetic() {
        let const_3 = constant(3.0);
        let const_4 = constant(4.0);
        let x = var::<0>();

        // Test addition
        let _add_expr = const_3.clone().add(const_4.clone());
        let _add_var = const_3.clone().add(x.clone());

        // Test subtraction
        let _sub_expr = const_3.clone().sub(const_4.clone());
        let _sub_var = const_3.clone().sub(x.clone());

        // Test multiplication
        let _mul_expr = const_3.clone().mul(const_4.clone());
        let _mul_var = const_3.clone().mul(x.clone());

        // Test division
        let _div_expr = const_3.clone().div(const_4.clone());
        let _div_var = const_3.clone().div(x.clone());

        // Test power
        let _pow_expr = const_3.clone().pow(const_4);
        let _pow_var = const_3.clone().pow(x);
    }

    #[test]
    fn test_compile_time_const_transcendental() {
        let const_pi = constant(3.14159);

        // Test trigonometric functions
        let _sin_pi = const_pi.clone().sin();
        let _cos_pi = const_pi.clone().cos();

        // Test exponential and logarithmic functions
        let _exp_pi = const_pi.clone().exp();
        let _ln_pi = const_pi.ln();
    }

    #[test]
    fn test_compile_time_expression_chaining() {
        let x = var::<0>();
        let y = var::<1>();
        let const_2 = constant(2.0);

        // Test complex expression: (x + y) * 2
        let _complex_expr = x.clone().add(y.clone()).mul(const_2.clone());

        // Test nested functions: sin(x + 1)
        let const_1 = constant(1.0);
        let _nested_expr = x.clone().add(const_1).sin();

        // Test power chaining: x^2 + y^2
        let _pythagorean = x.clone().pow(const_2.clone()).add(y.pow(const_2));
    }

    #[test]
    fn test_compile_time_add_operations() {
        let x = var::<0>();
        let const_5 = constant(5.0);
        let add_expr = x.add(const_5);

        // Test that add expressions can be further operated on
        let const_3 = constant(3.0);
        let _chained = add_expr.add(const_3);
    }

    #[test]
    fn test_compile_time_sub_operations() {
        let x = var::<0>();
        let const_5 = constant(5.0);
        let sub_expr = x.sub(const_5);

        // Test that sub expressions can be further operated on
        let const_3 = constant(3.0);
        let _chained = sub_expr.sub(const_3);
    }

    #[test]
    fn test_compile_time_mul_operations() {
        let x = var::<0>();
        let const_5 = constant(5.0);
        let mul_expr = x.mul(const_5);

        // Test that mul expressions can be further operated on
        let const_3 = constant(3.0);
        let _chained = mul_expr.mul(const_3);
    }

    #[test]
    fn test_compile_time_div_operations() {
        let x = var::<0>();
        let const_5 = constant(5.0);
        let div_expr = x.div(const_5);

        // Test that div expressions can be further operated on
        let const_3 = constant(3.0);
        let _chained = div_expr.div(const_3);
    }

    #[test]
    fn test_compile_time_pow_operations() {
        let x = var::<0>();
        let const_2 = constant(2.0);
        let pow_expr = x.pow(const_2);

        // Test that pow expressions can be further operated on
        let const_3 = constant(3.0);
        let _chained = pow_expr.add(const_3);
        
        // Test power chaining
        let y = var::<1>();
        let x2 = var::<0>();
        let const_2_clone = constant(2.0);
        let pow_expr2 = x2.pow(const_2_clone);
        let _power_chain = pow_expr2.pow(y);
    }

    #[test]
    fn test_compile_time_sin_operations() {
        let x = var::<0>();
        let sin_expr = x.sin();

        // Test that sin expressions can be further operated on
        let const_2 = constant(2.0);
        let _chained = sin_expr.mul(const_2);
    }

    #[test]
    fn test_compile_time_cos_operations() {
        let x = var::<0>();
        let cos_expr = x.cos();

        // Test that cos expressions can be further operated on
        let const_2 = constant(2.0);
        let _chained = cos_expr.mul(const_2);
    }

    #[test]
    fn test_compile_time_exp_operations() {
        let x = var::<0>();
        let exp_expr = x.exp();

        // Test that exp expressions can be further operated on
        let const_2 = constant(2.0);
        let _chained = exp_expr.add(const_2);
    }

    #[test]
    fn test_compile_time_ln_operations() {
        let x = var::<0>();
        let ln_expr = x.ln();

        // Test that ln expressions can be further operated on
        let const_2 = constant(2.0);
        let _chained = ln_expr.sub(const_2);
    }

    #[test]
    fn test_mixed_type_operations() {
        let x = var::<0>();
        let y = var::<1>();
        let const_pi = constant(3.14159);

        // Test mixing variables and constants
        let _mixed1 = x.clone().add(const_pi.clone());
        let _mixed2 = const_pi.clone().mul(y.clone());
        
        // Test complex mixed expression
        let _complex = x.clone().sin().add(y.cos()).mul(const_pi);
    }

    #[test]
    fn test_expression_type_safety() {
        // Test that different variable indices create different types
        let x0 = var::<0>();
        let x1 = var::<1>();
        let x2 = var::<2>();

        // These should all be different types at compile time
        let _expr1 = x0.clone().add(x1.clone());
        let _expr2 = x1.clone().add(x2.clone());
        let _expr3 = x0.mul(x2);
    }

    #[test]
    fn test_deeply_nested_expressions() {
        let x = var::<0>();
        let y = var::<1>();
        let const_1 = constant(1.0);
        let const_2 = constant(2.0);

        // Test deeply nested expression: sin(exp(x + 1)) * cos(y^2)
        let _nested = x.clone()
            .add(const_1)
            .exp()
            .sin()
            .mul(y.pow(const_2).cos());
    }

    #[test]
    fn test_mathematical_identities_structure() {
        let x = var::<0>();
        let const_0 = constant(0.0);
        let const_1 = constant(1.0);

        // Test identity structures (not evaluation, just construction)
        let _identity1 = x.clone().add(const_0); // x + 0
        let _identity2 = x.clone().mul(const_1); // x * 1
        let _identity3 = x.exp().ln(); // ln(exp(x))
    }

    #[test]
    fn test_compile_time_structures() {
        // Test that all the compile-time structures can be created
        let x = var::<0>();
        let y = var::<1>();
        let c = constant(42.0);

        // Create various expression types
        let add_expr: CompileTimeAdd<_, _> = x.clone().add(y.clone());
        let sub_expr: CompileTimeSub<_, _> = x.clone().sub(y.clone());
        let mul_expr: CompileTimeMul<_, _> = x.clone().mul(y.clone());
        let div_expr: CompileTimeDiv<_, _> = x.clone().div(y.clone());
        let pow_expr: CompileTimePow<_, _> = x.clone().pow(y.clone());
        
        let sin_expr: CompileTimeSin<_> = x.clone().sin();
        let cos_expr: CompileTimeCos<_> = x.clone().cos();
        let exp_expr: CompileTimeExp<_> = x.clone().exp();
        let ln_expr: CompileTimeLn<_> = x.ln();

        // Test that these can be further composed
        let _complex = add_expr.mul(c);
        let _trig_combo = sin_expr.add(cos_expr);
        let _exp_log = exp_expr.sub(ln_expr);
    }

    #[test]
    fn test_constant_value_access() {
        let const_pi = constant(3.14159);
        let const_e = constant(2.71828);
        let const_zero = constant(0.0);

        // Test that we can access the values
        assert_eq!(const_pi.value, 3.14159);
        assert_eq!(const_e.value, 2.71828);
        assert_eq!(const_zero.value, 0.0);
    }

    #[test]
    fn test_variable_index_uniqueness() {
        // Test that variables with different indices are distinct
        let _x0 = var::<0>();
        let _x1 = var::<1>();
        let _x2 = var::<2>();
        let _x10 = var::<10>();
        let _x100 = var::<100>();

        // Each should have a unique type
        assert_ne!(
            std::any::TypeId::of::<CompileTimeVar<0>>(),
            std::any::TypeId::of::<CompileTimeVar<1>>()
        );
        assert_ne!(
            std::any::TypeId::of::<CompileTimeVar<1>>(),
            std::any::TypeId::of::<CompileTimeVar<2>>()
        );
    }
}
