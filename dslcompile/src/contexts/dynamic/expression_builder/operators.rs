//! Operator Overloading for `DSLCompile` Expression Types
//!
//! This module contains all arithmetic operator implementations for `VariableExpr` and `DynamicExpr`,
//! including operations between same types, cross-type operations, reference operations, and scalar operations.
//!
//! ## Key Components
//!
//! - **`VariableExpr` Operations**: Automatic conversion to `DynamicExpr`
//! - **`DynamicExpr` Operations**: Direct AST manipulation
//! - **Reference Operations**: Efficient operations with borrowed values
//! - **Scalar Operations**: Operations between expressions and scalar values
//! - **Cross-Type Operations**: Type-safe operations between different numeric types

// Mathematical operations - only available for Scalar types
mod math_operators {
    use crate::{
        ast::{ExpressionType, Scalar, ast_repr::ASTRepr},
        contexts::dynamic::expression_builder::{DynamicBoundVar, DynamicExpr, VariableExpr},
    };
    use std::ops::{Add, Div, Mul, Neg, Sub};

    // ============================================================================
    // OPERATOR OVERLOADING FOR VariableExpr - AUTOMATIC CONVERSION
    // ============================================================================

    // Arithmetic operations for VariableExpr - automatically convert to DynamicExpr
    impl<T> Add for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn add(self, rhs: Self) -> Self::Output {
            self.into_expr::<0>() + rhs.into_expr::<0>()
        }
    }

    impl<T> Add<&VariableExpr<T>> for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
            self.clone().into_expr::<0>() + rhs.clone().into_expr::<0>()
        }
    }

    impl<T> Add<VariableExpr<T>> for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn add(self, rhs: VariableExpr<T>) -> Self::Output {
            self.clone().into_expr::<0>() + rhs.into_expr::<0>()
        }
    }

    impl<T> Add<&VariableExpr<T>> for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
            self.into_expr::<0>() + rhs.clone().into_expr::<0>()
        }
    }

    impl<T> Mul for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn mul(self, rhs: Self) -> Self::Output {
            self.into_expr::<0>() * rhs.into_expr::<0>()
        }
    }

    impl<T> Mul<&VariableExpr<T>> for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
            self.clone().into_expr::<0>() * rhs.clone().into_expr::<0>()
        }
    }

    impl<T> Mul<VariableExpr<T>> for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn mul(self, rhs: VariableExpr<T>) -> Self::Output {
            self.clone().into_expr::<0>() * rhs.into_expr::<0>()
        }
    }

    impl<T> Mul<&VariableExpr<T>> for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
            self.into_expr::<0>() * rhs.clone().into_expr::<0>()
        }
    }

    impl<T> Sub for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.into_expr::<0>() - rhs.into_expr::<0>()
        }
    }

    impl<T> Sub<&VariableExpr<T>> for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
            self.clone().into_expr::<0>() - rhs.clone().into_expr::<0>()
        }
    }

    impl<T> Sub<VariableExpr<T>> for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn sub(self, rhs: VariableExpr<T>) -> Self::Output {
            self.clone().into_expr::<0>() - rhs.into_expr::<0>()
        }
    }

    impl<T> Sub<&VariableExpr<T>> for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
            self.into_expr::<0>() - rhs.clone().into_expr::<0>()
        }
    }

    impl<T> Neg for VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Neg<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn neg(self) -> Self::Output {
            -self.into_expr::<0>()
        }
    }

    impl<T> Neg for &VariableExpr<T>
    where
        T: Scalar + ExpressionType + PartialEq + Neg<Output = T>,
    {
        type Output = DynamicExpr<T, 0>;

        fn neg(self) -> Self::Output {
            -self.clone().into_expr::<0>()
        }
    }

    // ============================================================================
    // SCALAR OPERATIONS FOR VariableExpr - SPECIFIC IMPLEMENTATIONS
    // ============================================================================

    impl Add<f64> for VariableExpr<f64> {
        type Output = DynamicExpr<f64, 0>;

        fn add(self, rhs: f64) -> Self::Output {
            self.into_expr::<0>() + rhs
        }
    }

    impl Add<VariableExpr<f64>> for f64 {
        type Output = DynamicExpr<f64, 0>;

        fn add(self, rhs: VariableExpr<f64>) -> Self::Output {
            self + rhs.into_expr::<0>()
        }
    }

    impl Mul<f64> for VariableExpr<f64> {
        type Output = DynamicExpr<f64, 0>;

        fn mul(self, rhs: f64) -> Self::Output {
            self.into_expr::<0>() * rhs
        }
    }

    impl Mul<VariableExpr<f64>> for f64 {
        type Output = DynamicExpr<f64, 0>;

        fn mul(self, rhs: VariableExpr<f64>) -> Self::Output {
            self * rhs.into_expr::<0>()
        }
    }

    impl Sub<f64> for VariableExpr<f64> {
        type Output = DynamicExpr<f64, 0>;

        fn sub(self, rhs: f64) -> Self::Output {
            self.into_expr::<0>() - rhs
        }
    }

    impl Sub<VariableExpr<f64>> for f64 {
        type Output = DynamicExpr<f64, 0>;

        fn sub(self, rhs: VariableExpr<f64>) -> Self::Output {
            self - rhs.into_expr::<0>()
        }
    }

    impl Div<f64> for VariableExpr<f64> {
        type Output = DynamicExpr<f64, 0>;

        fn div(self, rhs: f64) -> Self::Output {
            self.into_expr::<0>() / rhs
        }
    }

    impl Div<VariableExpr<f64>> for f64 {
        type Output = DynamicExpr<f64, 0>;

        fn div(self, rhs: VariableExpr<f64>) -> Self::Output {
            self / rhs.into_expr::<0>()
        }
    }

    // ============================================================================
    // CROSS-TYPE OPERATIONS FOR VariableExpr
    // ============================================================================
    // Note: Cross-type operations removed by design - use explicit conversions instead
    // This follows Rust's philosophy of explicit type conversions
    // Example: x_f64 + y_f32.into() or x_f64 + DynamicExpr::<f64>::from(y_f32)

    // ============================================================================
    // SAME-TYPE ARITHMETIC OPERATIONS FOR DynamicExpr
    // ============================================================================

    impl<T, const SCOPE: usize> Add for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: Self) -> Self::Output {
            DynamicExpr::new(ASTRepr::add_binary(self.ast, rhs.ast), self.registry)
        }
    }

    impl<T, const SCOPE: usize> Mul for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: Self) -> Self::Output {
            DynamicExpr::new(ASTRepr::mul_binary(self.ast, rhs.ast), self.registry)
        }
    }

    impl<T, const SCOPE: usize> Sub for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn sub(self, rhs: Self) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(self.ast), Box::new(rhs.ast)),
                self.registry,
            )
        }
    }

    impl<T, const SCOPE: usize> Div for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Div<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn div(self, rhs: Self) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(self.ast), Box::new(rhs.ast)),
                self.registry,
            )
        }
    }

    impl<T, const SCOPE: usize> Neg for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Neg<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn neg(self) -> Self::Output {
            DynamicExpr::new(-self.ast, self.registry)
        }
    }

    // ============================================================================
    // REFERENCE OPERATIONS FOR DynamicExpr - SCOPE-AWARE ONLY
    // ============================================================================

    // Reference operations - SCOPE-AWARE ONLY
    impl<T, const SCOPE: usize> Add<&DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            use crate::contexts::ScopeMerger;
            if ScopeMerger::needs_merging(self, rhs) {
                // Different registries - need scope merging
                let merged =
                    ScopeMerger::merge_and_combine(self, rhs, |l, r| ASTRepr::add_binary(l, r));
                // Convert back to scoped type - this is safe because the operation preserves scope semantics
                DynamicExpr::new(merged.ast, merged.registry)
            } else {
                // Same registry - use direct AST combination
                DynamicExpr::new(
                    ASTRepr::add_binary(self.ast.clone(), rhs.ast.clone()),
                    self.registry.clone(),
                )
            }
        }
    }

    impl<T, const SCOPE: usize> Add<DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(self.ast.clone(), rhs.ast),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Add<&DynamicExpr<T, SCOPE>> for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(self.ast, rhs.ast.clone()),
                self.registry,
            )
        }
    }

    impl<T, const SCOPE: usize> Mul<&DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(self.ast.clone(), rhs.ast.clone()),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Mul<DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(self.ast.clone(), rhs.ast),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Mul<&DynamicExpr<T, SCOPE>> for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(self.ast, rhs.ast.clone()),
                self.registry,
            )
        }
    }

    impl<T, const SCOPE: usize> Sub<&DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn sub(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(self.ast.clone()), Box::new(rhs.ast.clone())),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Sub<DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn sub(self, rhs: DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(self.ast.clone()), Box::new(rhs.ast)),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Sub<&DynamicExpr<T, SCOPE>> for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn sub(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(self.ast), Box::new(rhs.ast.clone())),
                self.registry,
            )
        }
    }

    impl<T, const SCOPE: usize> Div<&DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Div<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn div(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(self.ast.clone()), Box::new(rhs.ast.clone())),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Div<DynamicExpr<T, SCOPE>> for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Div<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn div(self, rhs: DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(self.ast.clone()), Box::new(rhs.ast)),
                self.registry.clone(),
            )
        }
    }

    impl<T, const SCOPE: usize> Div<&DynamicExpr<T, SCOPE>> for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Div<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn div(self, rhs: &DynamicExpr<T, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(self.ast), Box::new(rhs.ast.clone())),
                self.registry,
            )
        }
    }

    // Negation for references - SCOPE-AWARE ONLY
    impl<T, const SCOPE: usize> Neg for &DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Neg<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn neg(self) -> Self::Output {
            DynamicExpr::new(-&self.ast, self.registry.clone())
        }
    }

    // ============================================================================
    // SCALAR OPERATIONS - SCOPE-AWARE
    // ============================================================================

    // Scalar operations for DynamicExpr - maintain scope
    impl<const SCOPE: usize> Add<f64> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn add(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(self.ast, ASTRepr::Constant(rhs)),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Add<DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn add(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(ASTRepr::Constant(self), rhs.ast),
                rhs.registry,
            )
        }
    }

    impl<const SCOPE: usize> Mul<f64> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn mul(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(self.ast, ASTRepr::Constant(rhs)),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Mul<DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn mul(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(ASTRepr::Constant(self), rhs.ast),
                rhs.registry,
            )
        }
    }

    impl<const SCOPE: usize> Sub<f64> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn sub(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs))),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Sub<DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn sub(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
                rhs.registry,
            )
        }
    }

    impl<const SCOPE: usize> Div<f64> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn div(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs))),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Div<DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn div(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
                rhs.registry,
            )
        }
    }

    // ============================================================================
    // SCALAR OPERATIONS FOR REFERENCES - SCOPE-AWARE
    // ============================================================================

    // Reference scalar operations for DynamicExpr - maintain scope
    impl<const SCOPE: usize> Add<f64> for &DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn add(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(self.ast.clone(), ASTRepr::Constant(rhs)),
                self.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Add<&DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn add(self, rhs: &DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(ASTRepr::Constant(self), rhs.ast.clone()),
                rhs.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Mul<f64> for &DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn mul(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(self.ast.clone(), ASTRepr::Constant(rhs)),
                self.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Mul<&DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn mul(self, rhs: &DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(ASTRepr::Constant(self), rhs.ast.clone()),
                rhs.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Sub<f64> for &DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn sub(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(self.ast.clone()), Box::new(ASTRepr::Constant(rhs))),
                self.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Sub<&DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn sub(self, rhs: &DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
                rhs.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Div<f64> for &DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn div(self, rhs: f64) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(self.ast.clone()), Box::new(ASTRepr::Constant(rhs))),
                self.registry.clone(),
            )
        }
    }

    impl<const SCOPE: usize> Div<&DynamicExpr<f64, SCOPE>> for f64 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn div(self, rhs: &DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
                rhs.registry.clone(),
            )
        }
    }

    // ============================================================================
    // INTEGER LITERAL OPERATIONS - SCOPE-AWARE
    // ============================================================================

    // Integer literal operations for DynamicExpr<f64> - automatically convert to f64
    impl<const SCOPE: usize> Mul<i32> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn mul(self, rhs: i32) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(self.ast, ASTRepr::Constant(f64::from(rhs))),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Mul<DynamicExpr<f64, SCOPE>> for i32 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn mul(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::mul_binary(ASTRepr::Constant(f64::from(self)), rhs.ast),
                rhs.registry,
            )
        }
    }

    impl<const SCOPE: usize> Add<i32> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn add(self, rhs: i32) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(self.ast, ASTRepr::Constant(f64::from(rhs))),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Add<DynamicExpr<f64, SCOPE>> for i32 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn add(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::add_binary(ASTRepr::Constant(f64::from(self)), rhs.ast),
                rhs.registry,
            )
        }
    }

    impl<const SCOPE: usize> Sub<i32> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn sub(self, rhs: i32) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(
                    Box::new(self.ast),
                    Box::new(ASTRepr::Constant(f64::from(rhs))),
                ),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Sub<DynamicExpr<f64, SCOPE>> for i32 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn sub(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Sub(
                    Box::new(ASTRepr::Constant(f64::from(self))),
                    Box::new(rhs.ast),
                ),
                rhs.registry,
            )
        }
    }

    impl<const SCOPE: usize> Div<i32> for DynamicExpr<f64, SCOPE> {
        type Output = DynamicExpr<f64, SCOPE>;

        fn div(self, rhs: i32) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(
                    Box::new(self.ast),
                    Box::new(ASTRepr::Constant(f64::from(rhs))),
                ),
                self.registry,
            )
        }
    }

    impl<const SCOPE: usize> Div<DynamicExpr<f64, SCOPE>> for i32 {
        type Output = DynamicExpr<f64, SCOPE>;

        fn div(self, rhs: DynamicExpr<f64, SCOPE>) -> Self::Output {
            DynamicExpr::new(
                ASTRepr::Div(
                    Box::new(ASTRepr::Constant(f64::from(self))),
                    Box::new(rhs.ast),
                ),
                rhs.registry,
            )
        }
    }

    // ============================================================================
    // RUNTIME SCOPE MERGING INTEGRATION
    // ============================================================================

    // Rather than complex cross-scope operator implementations that conflict with
    // same-scope operators, we integrate scope merging into the existing operators.
    // The same-scope operators can detect at runtime if scope merging is needed.

    // ============================================================================
    // TESTS
    // ============================================================================

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::contexts::dynamic::expression_builder::DynamicContext;

        #[test]
        fn test_variable_expr_arithmetic() {
            let mut ctx = DynamicContext::new();
            let x = ctx.var();
            let y = ctx.var();

            // Test basic arithmetic operations
            let sum: DynamicExpr<f64> = x.clone() + y.clone();
            let product: DynamicExpr<f64> = x.clone() * y.clone();
            let difference: DynamicExpr<f64> = x.clone() - y.clone();
            let negation: DynamicExpr<f64> = -x.clone();

            // These should all be DynamicExpr instances
            assert!(matches!(sum.as_ast(), ASTRepr::Add(_)));
            assert!(matches!(product.as_ast(), ASTRepr::Mul(_)));
            assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
            assert!(matches!(negation.as_ast(), ASTRepr::Neg(_)));
        }

        #[test]
        fn test_variable_expr_scalar_operations() {
            let mut ctx = DynamicContext::new();
            let x = ctx.var();

            // Test scalar operations
            let sum: DynamicExpr<f64> = x.clone() + 5.0;
            let product: DynamicExpr<f64> = x.clone() * 2.0;
            let difference: DynamicExpr<f64> = x.clone() - 1.0;
            let quotient: DynamicExpr<f64> = x.clone() / 3.0;

            // Verify AST structure
            assert!(matches!(sum.as_ast(), ASTRepr::Add(_)));
            assert!(matches!(product.as_ast(), ASTRepr::Mul(_)));
            assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
            assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
        }

        #[test]
        fn test_typed_builder_expr_arithmetic() {
            let mut ctx = DynamicContext::new();
            let x: DynamicExpr<f64> = ctx.var().into_expr();
            let y: DynamicExpr<f64> = ctx.var().into_expr();

            // Test arithmetic operations
            let sum: DynamicExpr<f64> = x.clone() + y.clone();
            let product: DynamicExpr<f64> = x.clone() * y.clone();
            let difference: DynamicExpr<f64> = x.clone() - y.clone();
            let quotient: DynamicExpr<f64> = x.clone() / y.clone();
            let negation: DynamicExpr<f64> = -x.clone();

            // Verify AST structure
            assert!(matches!(sum.as_ast(), ASTRepr::Add(_)));
            assert!(matches!(product.as_ast(), ASTRepr::Mul(_)));
            assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
            assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
            assert!(matches!(negation.as_ast(), ASTRepr::Neg(_)));
        }

        #[test]
        fn test_reference_operations() {
            let mut ctx = DynamicContext::new();
            let x: DynamicExpr<f64> = ctx.var().into_expr();
            let y: DynamicExpr<f64> = ctx.var().into_expr();

            // Test reference operations
            let sum = &x + &y;
            let difference = &x - &y;
            let quotient = &x / &y;

            // Verify AST structure
            assert!(matches!(sum.as_ast(), ASTRepr::Add(_)));
            assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
            assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
        }

        #[test]
        fn test_scalar_commutative_operations() {
            let mut ctx = DynamicContext::new();
            let x: DynamicExpr<f64> = ctx.var().into_expr();

            // Test commutative operations
            let sum1: DynamicExpr<f64> = x.clone() + 5.0;
            let sum2: DynamicExpr<f64> = 5.0 + x.clone();
            let product1: DynamicExpr<f64> = x.clone() * 2.0;
            let product2: DynamicExpr<f64> = 2.0 * x.clone();

            // Both should create valid AST structures
            assert!(matches!(sum1.as_ast(), ASTRepr::Add(_)));
            assert!(matches!(sum2.as_ast(), ASTRepr::Add(_)));
            assert!(matches!(product1.as_ast(), ASTRepr::Mul(_)));
            assert!(matches!(product2.as_ast(), ASTRepr::Mul(_)));
        }
    }

    // ============================================================================
    // OPERATOR OVERLOADING FOR DynamicBoundVar - CLOSURE-BASED CSE SUPPORT
    // ============================================================================

    // Convert DynamicBoundVar to DynamicExpr and then use existing operators
    impl<T, const SCOPE: usize> Add for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: Self) -> Self::Output {
            self.to_expr() + rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Add<&DynamicBoundVar<T, SCOPE>> for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: &DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.clone().to_expr() + rhs.clone().to_expr()
        }
    }

    impl<T, const SCOPE: usize> Add<DynamicBoundVar<T, SCOPE>> for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.clone().to_expr() + rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Add<&DynamicBoundVar<T, SCOPE>> for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: &DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.to_expr() + rhs.clone().to_expr()
        }
    }

    // Multiplication operations for DynamicBoundVar
    impl<T, const SCOPE: usize> Mul for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: Self) -> Self::Output {
            self.to_expr() * rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Mul<&DynamicBoundVar<T, SCOPE>> for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: &DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.clone().to_expr() * rhs.clone().to_expr()
        }
    }

    impl<T, const SCOPE: usize> Mul<DynamicBoundVar<T, SCOPE>> for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.clone().to_expr() * rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Mul<&DynamicBoundVar<T, SCOPE>> for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: &DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.to_expr() * rhs.clone().to_expr()
        }
    }

    // Subtraction operations for DynamicBoundVar
    impl<T, const SCOPE: usize> Sub for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.to_expr() - rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Sub<&DynamicBoundVar<T, SCOPE>> for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Sub<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn sub(self, rhs: &DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.clone().to_expr() - rhs.clone().to_expr()
        }
    }

    // Division operations for DynamicBoundVar
    impl<T, const SCOPE: usize> Div for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Div<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn div(self, rhs: Self) -> Self::Output {
            self.to_expr() / rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Div<&DynamicBoundVar<T, SCOPE>> for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Div<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn div(self, rhs: &DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self.clone().to_expr() / rhs.clone().to_expr()
        }
    }

    // Cross-operations: DynamicBoundVar with DynamicExpr
    impl<T, const SCOPE: usize> Add<DynamicExpr<T, SCOPE>> for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: DynamicExpr<T, SCOPE>) -> Self::Output {
            self.to_expr() + rhs
        }
    }

    impl<T, const SCOPE: usize> Add<DynamicBoundVar<T, SCOPE>> for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Add<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn add(self, rhs: DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self + rhs.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Mul<DynamicExpr<T, SCOPE>> for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: DynamicExpr<T, SCOPE>) -> Self::Output {
            self.to_expr() * rhs
        }
    }

    impl<T, const SCOPE: usize> Mul<DynamicBoundVar<T, SCOPE>> for DynamicExpr<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Mul<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn mul(self, rhs: DynamicBoundVar<T, SCOPE>) -> Self::Output {
            self * rhs.to_expr()
        }
    }

    // Negation for DynamicBoundVar
    impl<T, const SCOPE: usize> Neg for DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Neg<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn neg(self) -> Self::Output {
            -self.to_expr()
        }
    }

    impl<T, const SCOPE: usize> Neg for &DynamicBoundVar<T, SCOPE>
    where
        T: Scalar + ExpressionType + PartialEq + Neg<Output = T>,
    {
        type Output = DynamicExpr<T, SCOPE>;

        fn neg(self) -> Self::Output {
            -self.clone().to_expr()
        }
    }
} // end math_operators module

// Re-export the operators module
