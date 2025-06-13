//! Operator Overloading for DSLCompile Expression Types
//!
//! This module contains all arithmetic operator implementations for VariableExpr and TypedBuilderExpr,
//! including operations between same types, cross-type operations, reference operations, and scalar operations.
//!
//! ## Key Components
//!
//! - **VariableExpr Operations**: Automatic conversion to TypedBuilderExpr
//! - **TypedBuilderExpr Operations**: Direct AST manipulation
//! - **Reference Operations**: Efficient operations with borrowed values
//! - **Scalar Operations**: Operations between expressions and scalar values
//! - **Cross-Type Operations**: Type-safe operations between different numeric types

use crate::{
    ast::{Scalar, ast_repr::ASTRepr},
    contexts::dynamic::expression_builder::{TypedBuilderExpr, VariableExpr},
};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// OPERATOR OVERLOADING FOR VariableExpr - AUTOMATIC CONVERSION
// ============================================================================

// Arithmetic operations for VariableExpr - automatically convert to TypedBuilderExpr
impl<T> Add for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn add(self, rhs: Self) -> Self::Output {
        self.into_expr::<0>() + rhs.into_expr::<0>()
    }
}

impl<T> Add<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr::<0>() + rhs.clone().into_expr::<0>()
    }
}

impl<T> Add<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn add(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr::<0>() + rhs.into_expr::<0>()
    }
}

impl<T> Add<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn add(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr::<0>() + rhs.clone().into_expr::<0>()
    }
}

impl<T> Mul for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.into_expr::<0>() * rhs.into_expr::<0>()
    }
}

impl<T> Mul<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr::<0>() * rhs.clone().into_expr::<0>()
    }
}

impl<T> Mul<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn mul(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr::<0>() * rhs.into_expr::<0>()
    }
}

impl<T> Mul<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn mul(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr::<0>() * rhs.clone().into_expr::<0>()
    }
}

impl<T> Sub for VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.into_expr::<0>() - rhs.into_expr::<0>()
    }
}

impl<T> Sub<&VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.clone().into_expr::<0>() - rhs.clone().into_expr::<0>()
    }
}

impl<T> Sub<VariableExpr<T>> for &VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn sub(self, rhs: VariableExpr<T>) -> Self::Output {
        self.clone().into_expr::<0>() - rhs.into_expr::<0>()
    }
}

impl<T> Sub<&VariableExpr<T>> for VariableExpr<T>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn sub(self, rhs: &VariableExpr<T>) -> Self::Output {
        self.into_expr::<0>() - rhs.clone().into_expr::<0>()
    }
}

impl<T> Neg for VariableExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn neg(self) -> Self::Output {
        -self.into_expr::<0>()
    }
}

impl<T> Neg for &VariableExpr<T>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T, 0>;

    fn neg(self) -> Self::Output {
        -self.clone().into_expr::<0>()
    }
}

// ============================================================================
// SCALAR OPERATIONS FOR VariableExpr - SPECIFIC IMPLEMENTATIONS
// ============================================================================

impl Add<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64, 0>;

    fn add(self, rhs: f64) -> Self::Output {
        self.into_expr::<0>() + rhs
    }
}

impl Add<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64, 0>;

    fn add(self, rhs: VariableExpr<f64>) -> Self::Output {
        self + rhs.into_expr::<0>()
    }
}

impl Mul<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64, 0>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.into_expr::<0>() * rhs
    }
}

impl Mul<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64, 0>;

    fn mul(self, rhs: VariableExpr<f64>) -> Self::Output {
        self * rhs.into_expr::<0>()
    }
}

impl Sub<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64, 0>;

    fn sub(self, rhs: f64) -> Self::Output {
        self.into_expr::<0>() - rhs
    }
}

impl Sub<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64, 0>;

    fn sub(self, rhs: VariableExpr<f64>) -> Self::Output {
        self - rhs.into_expr::<0>()
    }
}

impl Div<f64> for VariableExpr<f64> {
    type Output = TypedBuilderExpr<f64, 0>;

    fn div(self, rhs: f64) -> Self::Output {
        self.into_expr::<0>() / rhs
    }
}

impl Div<VariableExpr<f64>> for f64 {
    type Output = TypedBuilderExpr<f64, 0>;

    fn div(self, rhs: VariableExpr<f64>) -> Self::Output {
        self / rhs.into_expr::<0>()
    }
}

// ============================================================================
// CROSS-TYPE OPERATIONS FOR VariableExpr
// ============================================================================
// Note: Cross-type operations removed by design - use explicit conversions instead
// This follows Rust's philosophy of explicit type conversions
// Example: x_f64 + y_f32.into() or x_f64 + TypedBuilderExpr::<f64>::from(y_f32)

// ============================================================================
// SAME-TYPE ARITHMETIC OPERATIONS FOR TypedBuilderExpr
// ============================================================================

impl<T, const SCOPE: usize> Add for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn add(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(self.ast), Box::new(rhs.ast)),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Mul for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn mul(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast), Box::new(rhs.ast)),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Sub for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn sub(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast), Box::new(rhs.ast)),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Div for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn div(self, rhs: Self) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast), Box::new(rhs.ast)),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Neg for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-self.ast, self.registry)
    }
}

// ============================================================================
// REFERENCE OPERATIONS FOR TypedBuilderExpr - SCOPE-AWARE ONLY
// ============================================================================

// Reference operations - SCOPE-AWARE ONLY
impl<T, const SCOPE: usize> Add<&TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn add(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        use crate::contexts::ScopeMerger;
        if ScopeMerger::needs_merging(self, rhs) {
            // Different registries - need scope merging
            let merged = ScopeMerger::merge_and_combine(self, rhs, |l, r| {
                ASTRepr::Add(Box::new(l), Box::new(r))
            });
            // Convert back to scoped type - this is safe because the operation preserves scope semantics
            TypedBuilderExpr::new(merged.ast, merged.registry)
        } else {
            // Same registry - use direct AST combination
            TypedBuilderExpr::new(
                ASTRepr::Add(Box::new(self.ast.clone()), Box::new(rhs.ast.clone())),
                self.registry.clone(),
            )
        }
    }
}

impl<T, const SCOPE: usize> Add<TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn add(self, rhs: TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(self.ast.clone()), Box::new(rhs.ast)),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Add<&TypedBuilderExpr<T, SCOPE>> for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Add<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn add(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(self.ast), Box::new(rhs.ast.clone())),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Mul<&TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn mul(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast.clone()), Box::new(rhs.ast.clone())),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Mul<TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn mul(self, rhs: TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast.clone()), Box::new(rhs.ast)),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Mul<&TypedBuilderExpr<T, SCOPE>> for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Mul<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn mul(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast), Box::new(rhs.ast.clone())),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Sub<&TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn sub(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast.clone()), Box::new(rhs.ast.clone())),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Sub<TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn sub(self, rhs: TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast.clone()), Box::new(rhs.ast)),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Sub<&TypedBuilderExpr<T, SCOPE>> for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Sub<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn sub(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast), Box::new(rhs.ast.clone())),
            self.registry,
        )
    }
}

impl<T, const SCOPE: usize> Div<&TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn div(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast.clone()), Box::new(rhs.ast.clone())),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Div<TypedBuilderExpr<T, SCOPE>> for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn div(self, rhs: TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast.clone()), Box::new(rhs.ast)),
            self.registry.clone(),
        )
    }
}

impl<T, const SCOPE: usize> Div<&TypedBuilderExpr<T, SCOPE>> for TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Div<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn div(self, rhs: &TypedBuilderExpr<T, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast), Box::new(rhs.ast.clone())),
            self.registry,
        )
    }
}

// Negation for references - SCOPE-AWARE ONLY
impl<T, const SCOPE: usize> Neg for &TypedBuilderExpr<T, SCOPE>
where
    T: Scalar + Neg<Output = T>,
{
    type Output = TypedBuilderExpr<T, SCOPE>;

    fn neg(self) -> Self::Output {
        TypedBuilderExpr::new(-&self.ast, self.registry.clone())
    }
}

// ============================================================================
// SCALAR OPERATIONS - SCOPE-AWARE
// ============================================================================

// Scalar operations for TypedBuilderExpr - maintain scope
impl<const SCOPE: usize> Add<f64> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn add(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Add<TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn add(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

impl<const SCOPE: usize> Mul<f64> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn mul(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Mul<TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn mul(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

impl<const SCOPE: usize> Sub<f64> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn sub(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Sub<TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn sub(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

impl<const SCOPE: usize> Div<f64> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn div(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Div<TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn div(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

// ============================================================================
// SCALAR OPERATIONS FOR REFERENCES - SCOPE-AWARE
// ============================================================================

// Reference scalar operations for TypedBuilderExpr - maintain scope
impl<const SCOPE: usize> Add<f64> for &TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn add(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(self.ast.clone()), Box::new(ASTRepr::Constant(rhs))),
            self.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Add<&TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn add(self, rhs: &TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
            rhs.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Mul<f64> for &TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn mul(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast.clone()), Box::new(ASTRepr::Constant(rhs))),
            self.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Mul<&TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn mul(self, rhs: &TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
            rhs.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Sub<f64> for &TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn sub(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast.clone()), Box::new(ASTRepr::Constant(rhs))),
            self.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Sub<&TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn sub(self, rhs: &TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
            rhs.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Div<f64> for &TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn div(self, rhs: f64) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast.clone()), Box::new(ASTRepr::Constant(rhs))),
            self.registry.clone(),
        )
    }
}

impl<const SCOPE: usize> Div<&TypedBuilderExpr<f64, SCOPE>> for f64 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn div(self, rhs: &TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(ASTRepr::Constant(self)), Box::new(rhs.ast.clone())),
            rhs.registry.clone(),
        )
    }
}

// ============================================================================
// INTEGER LITERAL OPERATIONS - SCOPE-AWARE
// ============================================================================

// Integer literal operations for TypedBuilderExpr<f64> - automatically convert to f64
impl<const SCOPE: usize> Mul<i32> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn mul(self, rhs: i32) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs as f64))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Mul<TypedBuilderExpr<f64, SCOPE>> for i32 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn mul(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Mul(Box::new(ASTRepr::Constant(self as f64)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

impl<const SCOPE: usize> Add<i32> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn add(self, rhs: i32) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs as f64))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Add<TypedBuilderExpr<f64, SCOPE>> for i32 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn add(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Add(Box::new(ASTRepr::Constant(self as f64)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

impl<const SCOPE: usize> Sub<i32> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn sub(self, rhs: i32) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs as f64))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Sub<TypedBuilderExpr<f64, SCOPE>> for i32 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn sub(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Sub(Box::new(ASTRepr::Constant(self as f64)), Box::new(rhs.ast)),
            rhs.registry,
        )
    }
}

impl<const SCOPE: usize> Div<i32> for TypedBuilderExpr<f64, SCOPE> {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn div(self, rhs: i32) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(self.ast), Box::new(ASTRepr::Constant(rhs as f64))),
            self.registry,
        )
    }
}

impl<const SCOPE: usize> Div<TypedBuilderExpr<f64, SCOPE>> for i32 {
    type Output = TypedBuilderExpr<f64, SCOPE>;

    fn div(self, rhs: TypedBuilderExpr<f64, SCOPE>) -> Self::Output {
        TypedBuilderExpr::new(
            ASTRepr::Div(Box::new(ASTRepr::Constant(self as f64)), Box::new(rhs.ast)),
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
        let sum: TypedBuilderExpr<f64> = x.clone() + y.clone();
        let product: TypedBuilderExpr<f64> = x.clone() * y.clone();
        let difference: TypedBuilderExpr<f64> = x.clone() - y.clone();
        let negation: TypedBuilderExpr<f64> = -x.clone();

        // These should all be TypedBuilderExpr instances
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(negation.as_ast(), ASTRepr::Neg(_)));
    }

    #[test]
    fn test_variable_expr_scalar_operations() {
        let mut ctx = DynamicContext::new();
        let x = ctx.var();

        // Test scalar operations
        let sum: TypedBuilderExpr<f64> = x.clone() + 5.0;
        let product: TypedBuilderExpr<f64> = x.clone() * 2.0;
        let difference: TypedBuilderExpr<f64> = x.clone() - 1.0;
        let quotient: TypedBuilderExpr<f64> = x.clone() / 3.0;

        // Verify AST structure
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
    }

    #[test]
    fn test_typed_builder_expr_arithmetic() {
        let mut ctx = DynamicContext::new();
        let x: TypedBuilderExpr<f64> = ctx.var().into_expr();
        let y: TypedBuilderExpr<f64> = ctx.var().into_expr();

        // Test arithmetic operations
        let sum: TypedBuilderExpr<f64> = x.clone() + y.clone();
        let product: TypedBuilderExpr<f64> = x.clone() * y.clone();
        let difference: TypedBuilderExpr<f64> = x.clone() - y.clone();
        let quotient: TypedBuilderExpr<f64> = x.clone() / y.clone();
        let negation: TypedBuilderExpr<f64> = -x.clone();

        // Verify AST structure
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
        assert!(matches!(negation.as_ast(), ASTRepr::Neg(_)));
    }

    #[test]
    fn test_reference_operations() {
        let mut ctx = DynamicContext::new();
        let x: TypedBuilderExpr<f64> = ctx.var().into_expr();
        let y: TypedBuilderExpr<f64> = ctx.var().into_expr();

        // Test reference operations
        let sum = &x + &y;
        let difference = &x - &y;
        let quotient = &x / &y;

        // Verify AST structure
        assert!(matches!(sum.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(difference.as_ast(), ASTRepr::Sub(_, _)));
        assert!(matches!(quotient.as_ast(), ASTRepr::Div(_, _)));
    }

    #[test]
    fn test_scalar_commutative_operations() {
        let mut ctx = DynamicContext::new();
        let x: TypedBuilderExpr<f64> = ctx.var().into_expr();

        // Test commutative operations
        let sum1: TypedBuilderExpr<f64> = x.clone() + 5.0;
        let sum2: TypedBuilderExpr<f64> = 5.0 + x.clone();
        let product1: TypedBuilderExpr<f64> = x.clone() * 2.0;
        let product2: TypedBuilderExpr<f64> = 2.0 * x.clone();

        // Both should create valid AST structures
        assert!(matches!(sum1.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(sum2.as_ast(), ASTRepr::Add(_, _)));
        assert!(matches!(product1.as_ast(), ASTRepr::Mul(_, _)));
        assert!(matches!(product2.as_ast(), ASTRepr::Mul(_, _)));
    }
}
