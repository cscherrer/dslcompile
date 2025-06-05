//! Unified Variadic System using Frunk `HLists`
//!
//! This module provides a unified approach to heterogeneous variadic functions
//! that works identically for both Static and Dynamic contexts, eliminating
//! the need for separate Context/HeteroContext types.

use crate::ast::ASTRepr;
use frunk::hlist::HList;
use frunk::{HCons, HNil};
use std::marker::PhantomData;

// ============================================================================
// CORE UNIFIED CONTEXT TRAIT
// ============================================================================

/// Unified context trait that both Static and Dynamic contexts implement
pub trait UnifiedContext {
    type Expr: Clone;

    /// Create a variable from any supported type
    fn var<T: IntoContextValue>(&mut self, value: T) -> Self::Expr;

    /// Create a constant
    fn constant(&mut self, value: f64) -> Self::Expr;

    /// Add expressions
    fn add(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr;

    /// Multiply expressions  
    fn mul(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr;

    /// Subtract expressions
    fn sub(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr;

    /// Divide expressions
    fn div(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr;

    /// Power expressions
    fn pow(&mut self, base: Self::Expr, exp: Self::Expr) -> Self::Expr;

    /// Sine function
    fn sin(&mut self, expr: Self::Expr) -> Self::Expr;

    /// Cosine function
    fn cos(&mut self, expr: Self::Expr) -> Self::Expr;

    /// Natural log
    fn ln(&mut self, expr: Self::Expr) -> Self::Expr;

    /// Exponential
    fn exp(&mut self, expr: Self::Expr) -> Self::Expr;

    /// Square root
    fn sqrt(&mut self, expr: Self::Expr) -> Self::Expr;

    /// Sum multiple heterogeneous arguments
    fn sum<Args>(&mut self, args: Args) -> Self::Expr
    where
        Args: HList,
        Self: Sized,
        Args: Summable<Self>;

    /// Multiply multiple heterogeneous arguments  
    fn multiply<Args>(&mut self, args: Args) -> Self::Expr
    where
        Args: HList,
        Self: Sized,
        Args: Multipliable<Self>;
}

// ============================================================================
// VALUE CONVERSION TRAIT
// ============================================================================

/// Trait for types that can be converted to context values
pub trait IntoContextValue {
    fn into_value(self) -> ContextValue;
}

/// Unified value type that both contexts can handle
#[derive(Clone, Debug)]
pub enum ContextValue {
    Scalar(f64),
    Vector(Vec<f64>),
    Index(usize),
    Boolean(bool),
}

impl IntoContextValue for f64 {
    fn into_value(self) -> ContextValue {
        ContextValue::Scalar(self)
    }
}

impl IntoContextValue for Vec<f64> {
    fn into_value(self) -> ContextValue {
        ContextValue::Vector(self)
    }
}

impl IntoContextValue for &[f64] {
    fn into_value(self) -> ContextValue {
        ContextValue::Vector(self.to_vec())
    }
}

impl IntoContextValue for usize {
    fn into_value(self) -> ContextValue {
        ContextValue::Index(self)
    }
}

impl IntoContextValue for bool {
    fn into_value(self) -> ContextValue {
        ContextValue::Boolean(self)
    }
}

// ============================================================================
// FRUNK-BASED VARIADIC OPERATIONS
// ============================================================================

/// Trait for `HLists` that can be summed in a context
pub trait Summable<Ctx: UnifiedContext> {
    fn sum_with_context(self, ctx: &mut Ctx) -> Ctx::Expr;
}

/// Trait for `HLists` that can be used as function arguments
pub trait FunctionArgs<Ctx: UnifiedContext> {
    type Output;
    fn apply_to_function<F>(self, ctx: &mut Ctx, func: F) -> Self::Output
    where
        F: FnOnce(&mut Ctx, Self) -> Self::Output;
}

/// Trait for `HLists` that can be converted to expressions
pub trait IntoExpression<Ctx: UnifiedContext> {
    fn into_expression(self, ctx: &mut Ctx) -> Ctx::Expr;
}

// ============================================================================
// IMPLEMENTATIONS FOR COMMON HLIST PATTERNS
// ============================================================================

// Base case: Single element (HNil tail)
impl<Ctx: UnifiedContext, T: IntoContextValue> Summable<Ctx> for HCons<T, HNil> {
    fn sum_with_context(self, ctx: &mut Ctx) -> Ctx::Expr {
        ctx.var(self.head)
    }
}

// Recursive case: Multiple elements (non-HNil tail)
impl<Ctx: UnifiedContext, Head, TailHead, TailTail> Summable<Ctx>
    for HCons<Head, HCons<TailHead, TailTail>>
where
    Head: IntoContextValue,
    TailHead: IntoContextValue,
    TailTail: HList,
    HCons<TailHead, TailTail>: Summable<Ctx>,
{
    fn sum_with_context(self, ctx: &mut Ctx) -> Ctx::Expr {
        let head_expr = ctx.var(self.head);
        let tail_expr = self.tail.sum_with_context(ctx);
        ctx.add(head_expr, tail_expr)
    }
}

// Single element conversion
impl<Ctx: UnifiedContext, T: IntoContextValue> IntoExpression<Ctx> for HCons<T, HNil> {
    fn into_expression(self, ctx: &mut Ctx) -> Ctx::Expr {
        ctx.var(self.head)
    }
}

// Two element addition
impl<Ctx: UnifiedContext, T1, T2> IntoExpression<Ctx> for HCons<T1, HCons<T2, HNil>>
where
    T1: IntoContextValue,
    T2: IntoContextValue,
{
    fn into_expression(self, ctx: &mut Ctx) -> Ctx::Expr {
        let HCons {
            head: first,
            tail: HCons { head: second, .. },
        } = self;
        let expr1 = ctx.var(first);
        let expr2 = ctx.var(second);
        ctx.add(expr1, expr2)
    }
}

// ============================================================================
// UNIFIED FUNCTION INTERFACE
// ============================================================================

/// Universal sum function that works with any `HList` and any context
pub fn sum<Args, Ctx>(ctx: &mut Ctx, args: Args) -> Ctx::Expr
where
    Args: HList + Summable<Ctx>,
    Ctx: UnifiedContext,
{
    args.sum_with_context(ctx)
}

/// Universal multiply function
pub fn multiply<Args, Ctx>(ctx: &mut Ctx, args: Args) -> Ctx::Expr
where
    Args: HList + Multipliable<Ctx>,
    Ctx: UnifiedContext,
{
    args.multiply_with_context(ctx)
}

/// Universal function application
pub fn apply<Args, Ctx, F, Output>(ctx: &mut Ctx, args: Args, func: F) -> Output
where
    Args: HList + FunctionArgs<Ctx, Output = Output>,
    Ctx: UnifiedContext,
    F: FnOnce(&mut Ctx, Args) -> Output,
{
    args.apply_to_function(ctx, func)
}

// ============================================================================
// TRAIT FOR MULTIPLICATION (similar to summation)
// ============================================================================

pub trait Multipliable<Ctx: UnifiedContext> {
    fn multiply_with_context(self, ctx: &mut Ctx) -> Ctx::Expr;
}

// Base case for multiplication
impl<Ctx: UnifiedContext, T: IntoContextValue> Multipliable<Ctx> for HCons<T, HNil> {
    fn multiply_with_context(self, ctx: &mut Ctx) -> Ctx::Expr {
        ctx.var(self.head)
    }
}

// Recursive case for multiplication
impl<Ctx: UnifiedContext, Head, TailHead, TailTail> Multipliable<Ctx>
    for HCons<Head, HCons<TailHead, TailTail>>
where
    Head: IntoContextValue,
    TailHead: IntoContextValue,
    TailTail: HList,
    HCons<TailHead, TailTail>: Multipliable<Ctx>,
{
    fn multiply_with_context(self, ctx: &mut Ctx) -> Ctx::Expr {
        let head_expr = ctx.var(self.head);
        let tail_expr = self.tail.multiply_with_context(ctx);
        ctx.mul(head_expr, tail_expr)
    }
}

// ============================================================================
// ADVANCED OPERATIONS WITH MIXED TYPES
// ============================================================================

/// Trait for array operations with heterogeneous arguments
pub trait ArrayOperation<Ctx: UnifiedContext> {
    fn index_with_bias(self, ctx: &mut Ctx) -> Ctx::Expr;
}

// Array[index] + bias pattern
impl<Ctx: UnifiedContext> ArrayOperation<Ctx> for HCons<Vec<f64>, HCons<usize, HCons<f64, HNil>>> {
    fn index_with_bias(self, ctx: &mut Ctx) -> Ctx::Expr {
        let HCons {
            head: array,
            tail:
                HCons {
                    head: index,
                    tail: HCons { head: bias, .. },
                },
        } = self;

        // This would need array indexing support in the context
        let array_val = if index < array.len() {
            array[index]
        } else {
            0.0
        };
        let array_expr = ctx.var(array_val);
        let bias_expr = ctx.var(bias);
        ctx.add(array_expr, bias_expr)
    }
}

/// Trait for conditional operations
pub trait ConditionalOperation<Ctx: UnifiedContext> {
    fn if_then_else(self, ctx: &mut Ctx) -> Ctx::Expr;
}

// Condition ? true_val : false_val pattern
impl<Ctx: UnifiedContext, T1, T2> ConditionalOperation<Ctx>
    for HCons<bool, HCons<T1, HCons<T2, HNil>>>
where
    T1: IntoContextValue,
    T2: IntoContextValue,
{
    fn if_then_else(self, ctx: &mut Ctx) -> Ctx::Expr {
        let HCons {
            head: condition,
            tail:
                HCons {
                    head: true_val,
                    tail: HCons {
                        head: false_val, ..
                    },
                },
        } = self;

        if condition {
            ctx.var(true_val)
        } else {
            ctx.var(false_val)
        }
    }
}

// ============================================================================
// UNIFIED CONTEXT IMPLEMENTATIONS (PROTOTYPES)
// ============================================================================

/// Static context with compile-time optimization
pub struct StaticContext<T: Clone = f64> {
    variable_count: usize,
    _phantom: PhantomData<T>,
}

/// Dynamic context with runtime flexibility
pub struct DynamicContext {
    variable_count: usize,
    ast_nodes: Vec<ASTRepr<f64>>,
}

/// Unified static expression type
#[derive(Clone)]
pub struct StaticExpr<T: Clone = f64> {
    // This would contain compile-time type information
    node_type: StaticNodeType<T>,
    variable_id: Option<usize>,
}

/// Dynamic expression type (feature parity with static)
#[derive(Clone)]
pub struct DynamicExpr {
    ast: ASTRepr<f64>,
    variable_id: Option<usize>,
}

#[derive(Clone)]
enum StaticNodeType<T: Clone> {
    Variable(usize),
    Constant(T),
    Add(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    Mul(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    Sub(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    Div(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    Pow(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    Sin(Box<StaticExpr<T>>),
    Cos(Box<StaticExpr<T>>),
    Ln(Box<StaticExpr<T>>),
    Exp(Box<StaticExpr<T>>),
    Sqrt(Box<StaticExpr<T>>),
}

impl<T: Clone> Default for StaticContext<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> StaticContext<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            variable_count: 0,
            _phantom: PhantomData,
        }
    }
}

impl Default for DynamicContext {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            variable_count: 0,
            ast_nodes: Vec::new(),
        }
    }
}

// Implement UnifiedContext for both
impl UnifiedContext for StaticContext<f64> {
    type Expr = StaticExpr<f64>;

    fn var<T: IntoContextValue>(&mut self, value: T) -> Self::Expr {
        let var_id = self.variable_count;
        self.variable_count += 1;

        // For static context, we might store the value type information
        StaticExpr {
            node_type: StaticNodeType::Variable(var_id),
            variable_id: Some(var_id),
        }
    }

    fn constant(&mut self, value: f64) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Constant(value),
            variable_id: None,
        }
    }

    fn add(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Add(Box::new(left), Box::new(right)),
            variable_id: None,
        }
    }

    fn mul(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Mul(Box::new(left), Box::new(right)),
            variable_id: None,
        }
    }

    // ... implement other operations
    fn sub(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Sub(Box::new(left), Box::new(right)),
            variable_id: None,
        }
    }
    fn div(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Div(Box::new(left), Box::new(right)),
            variable_id: None,
        }
    }
    fn pow(&mut self, base: Self::Expr, exp: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Pow(Box::new(base), Box::new(exp)),
            variable_id: None,
        }
    }
    fn sin(&mut self, expr: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Sin(Box::new(expr)),
            variable_id: None,
        }
    }
    fn cos(&mut self, expr: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Cos(Box::new(expr)),
            variable_id: None,
        }
    }
    fn ln(&mut self, expr: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Ln(Box::new(expr)),
            variable_id: None,
        }
    }
    fn exp(&mut self, expr: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Exp(Box::new(expr)),
            variable_id: None,
        }
    }
    fn sqrt(&mut self, expr: Self::Expr) -> Self::Expr {
        StaticExpr {
            node_type: StaticNodeType::Sqrt(Box::new(expr)),
            variable_id: None,
        }
    }

    fn sum<Args>(&mut self, args: Args) -> Self::Expr
    where
        Args: HList + Summable<Self>,
    {
        args.sum_with_context(self)
    }

    fn multiply<Args>(&mut self, args: Args) -> Self::Expr
    where
        Args: HList + Multipliable<Self>,
    {
        args.multiply_with_context(self)
    }
}

impl UnifiedContext for DynamicContext {
    type Expr = DynamicExpr;

    fn var<T: IntoContextValue>(&mut self, value: T) -> Self::Expr {
        let var_id = self.variable_count;
        self.variable_count += 1;

        // For dynamic context, we create AST nodes
        let ast = ASTRepr::Variable(var_id);
        DynamicExpr {
            ast,
            variable_id: Some(var_id),
        }
    }

    fn constant(&mut self, value: f64) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Constant(value),
            variable_id: None,
        }
    }

    fn add(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Add(Box::new(left.ast), Box::new(right.ast)),
            variable_id: None,
        }
    }

    fn mul(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Mul(Box::new(left.ast), Box::new(right.ast)),
            variable_id: None,
        }
    }

    // ... implement other operations
    fn sub(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Sub(Box::new(left.ast), Box::new(right.ast)),
            variable_id: None,
        }
    }
    fn div(&mut self, left: Self::Expr, right: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Div(Box::new(left.ast), Box::new(right.ast)),
            variable_id: None,
        }
    }
    fn pow(&mut self, base: Self::Expr, exp: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Pow(Box::new(base.ast), Box::new(exp.ast)),
            variable_id: None,
        }
    }
    fn sin(&mut self, expr: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Sin(Box::new(expr.ast)),
            variable_id: None,
        }
    }
    fn cos(&mut self, expr: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Cos(Box::new(expr.ast)),
            variable_id: None,
        }
    }
    fn ln(&mut self, expr: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Ln(Box::new(expr.ast)),
            variable_id: None,
        }
    }
    fn exp(&mut self, expr: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Exp(Box::new(expr.ast)),
            variable_id: None,
        }
    }
    fn sqrt(&mut self, expr: Self::Expr) -> Self::Expr {
        DynamicExpr {
            ast: ASTRepr::Sqrt(Box::new(expr.ast)),
            variable_id: None,
        }
    }

    fn sum<Args>(&mut self, args: Args) -> Self::Expr
    where
        Args: HList + Summable<Self>,
    {
        args.sum_with_context(self)
    }

    fn multiply<Args>(&mut self, args: Args) -> Self::Expr
    where
        Args: HList + Multipliable<Self>,
    {
        args.multiply_with_context(self)
    }
}

// ============================================================================
// UNIFIED API EXAMPLES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use frunk::hlist;

    #[test]
    fn test_unified_sum_static() {
        let mut ctx = StaticContext::new();

        // New method-based API!
        let _sum1 = ctx.sum(hlist![3.0]);
        let _sum2 = ctx.sum(hlist![3.0, 4.0]);
        let _sum3 = ctx.sum(hlist![3.0, 4.0, 5.0]);
        let _sum_mixed = ctx.sum(hlist![3.0, vec![1.0, 2.0], 42usize]);
    }

    #[test]
    fn test_unified_sum_dynamic() {
        let mut ctx = DynamicContext::new();

        // Identical method-based API!
        let _sum1 = ctx.sum(hlist![3.0]);
        let _sum2 = ctx.sum(hlist![3.0, 4.0]);
        let _sum3 = ctx.sum(hlist![3.0, 4.0, 5.0]);
        let _sum_mixed = ctx.sum(hlist![3.0, vec![1.0, 2.0], 42usize]);
    }

    #[test]
    fn test_unified_multiply() {
        let mut static_ctx = StaticContext::new();
        let mut dynamic_ctx = DynamicContext::new();

        // Method-based API for both contexts
        let _static_mul = static_ctx.multiply(hlist![2.0, 3.0, 4.0]);
        let _dynamic_mul = dynamic_ctx.multiply(hlist![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_api_comparison() {
        let mut ctx = DynamicContext::new();

        // Both APIs work, but method-based is more ergonomic
        let _old_way = sum(&mut ctx, hlist![1.0, 2.0]);
        let _new_way = ctx.sum(hlist![1.0, 2.0]);
    }

    #[test]
    fn test_all_mathematical_operations() {
        let mut static_ctx = StaticContext::new();
        let mut dynamic_ctx = DynamicContext::new();

        // Test basic operations
        let x_static = static_ctx.var(5.0);
        let y_static = static_ctx.constant(3.0);

        let x_dynamic = dynamic_ctx.var(5.0);
        let y_dynamic = dynamic_ctx.constant(3.0);

        // Test all binary operations
        let _add_static = static_ctx.add(x_static.clone(), y_static.clone());
        let _sub_static = static_ctx.sub(x_static.clone(), y_static.clone());
        let _mul_static = static_ctx.mul(x_static.clone(), y_static.clone());
        let _div_static = static_ctx.div(x_static.clone(), y_static.clone());
        let _pow_static = static_ctx.pow(x_static.clone(), y_static.clone());

        let _add_dynamic = dynamic_ctx.add(x_dynamic.clone(), y_dynamic.clone());
        let _sub_dynamic = dynamic_ctx.sub(x_dynamic.clone(), y_dynamic.clone());
        let _mul_dynamic = dynamic_ctx.mul(x_dynamic.clone(), y_dynamic.clone());
        let _div_dynamic = dynamic_ctx.div(x_dynamic.clone(), y_dynamic.clone());
        let _pow_dynamic = dynamic_ctx.pow(x_dynamic.clone(), y_dynamic.clone());

        // Test all unary operations
        let _sin_static = static_ctx.sin(x_static.clone());
        let _cos_static = static_ctx.cos(x_static.clone());
        let _ln_static = static_ctx.ln(x_static.clone());
        let _exp_static = static_ctx.exp(x_static.clone());
        let _sqrt_static = static_ctx.sqrt(x_static);

        let _sin_dynamic = dynamic_ctx.sin(x_dynamic.clone());
        let _cos_dynamic = dynamic_ctx.cos(x_dynamic.clone());
        let _ln_dynamic = dynamic_ctx.ln(x_dynamic.clone());
        let _exp_dynamic = dynamic_ctx.exp(x_dynamic.clone());
        let _sqrt_dynamic = dynamic_ctx.sqrt(x_dynamic);
    }

    #[test]
    fn test_method_chaining() {
        let mut ctx = DynamicContext::new();

        // Test ergonomic method chaining
        let x = ctx.var(2.0);
        let y = ctx.var(3.0);

        // Build sub-expressions first to avoid multiple borrows
        let xy_mul = ctx.mul(x.clone(), y.clone());
        let xy_add = ctx.add(x, y);
        let sin_part = ctx.sin(xy_add);
        let complex_expr = ctx.add(xy_mul, sin_part);

        // Verify it compiles and has the right structure
        assert!(complex_expr.variable_id.is_none()); // It's not a simple variable
    }

    #[test]
    fn test_heterogeneous_operations() {
        let mut static_ctx = StaticContext::new();
        let mut dynamic_ctx = DynamicContext::new();

        // Test the sum operation with various heterogeneous arguments
        let _sum_scalars = static_ctx.sum(hlist![1.0, 2.0, 3.0, 4.0]);
        let _sum_mixed = static_ctx.sum(hlist![5.0, vec![1.0, 2.0], 42usize, true]);

        let _dyn_sum_scalars = dynamic_ctx.sum(hlist![1.0, 2.0, 3.0, 4.0]);
        let _dyn_sum_mixed = dynamic_ctx.sum(hlist![5.0, vec![1.0, 2.0], 42usize, true]);

        // Test the multiply operation
        let _mul_scalars = static_ctx.multiply(hlist![2.0, 3.0, 4.0]);
        let _dyn_mul_scalars = dynamic_ctx.multiply(hlist![2.0, 3.0, 4.0]);
    }
}
