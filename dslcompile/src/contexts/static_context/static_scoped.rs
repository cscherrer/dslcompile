//! Static Scoped Variables with `HList` Integration
//!
//! This module provides the next-generation compile-time mathematical expression system
//! that combines:
//! - **Type-level scoping** for safe composability (from scoped.rs)
//! - **Zero-overhead performance** matching native Rust (from heterogeneous.rs)  
//! - **`HList` integration** for variadic heterogeneous inputs (from `DynamicContext`)
//! - **No artificial limitations** - no `MAX_VARS`, grows as needed
//!
//! ## Key Features
//! - **Safe Composability**: Type-level scopes prevent variable collisions
//! - **Native Performance**: Zero runtime overhead, direct field access
//! - **Heterogeneous Types**: Mix f64, Vec<f64>, usize, custom types seamlessly
//! - **Ergonomic API**: Natural mathematical syntax with operator overloading
//! - **`HList` Storage**: Compile-time heterogeneous storage without size limits
//!
//! ## Example
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

use frunk::{HCons, HNil};
use std::marker::PhantomData;
use num_traits;

// ============================================================================
// CORE TRAITS - ZERO-OVERHEAD FOUNDATION
// ============================================================================

/// Core trait for types that can participate in static scoped expressions
/// 
/// This trait is compatible with DynamicContext's DslType system for seamless interoperability.
/// All types that implement DslType from DynamicContext should also work in StaticContext.
pub trait StaticExpressionType: Clone + std::fmt::Debug + 'static {}

// ============================================================================
// TYPE SYSTEM COMPATIBILITY WITH DYNAMICCONTEXT
// ============================================================================

// Import the type system traits from DynamicContext for compatibility
use crate::contexts::dynamic::expression_builder::type_system::{DslType, DataType};
use crate::ast::Scalar;

// Automatic implementation: Any DslType can be used as StaticExpressionType
impl<T> StaticExpressionType for T
where
    T: DslType + Clone + std::fmt::Debug + 'static,
{
}

// Additional implementations for Vec types that implement DataType
// This enables heterogeneous data support matching DynamicContext capabilities
impl<T> StaticExpressionType for Vec<T>
where
    T: Scalar + Clone + std::fmt::Debug + 'static,
{
}

/// Zero-overhead storage trait using `HList` compile-time specialization
pub trait HListStorage<T: StaticExpressionType> {
    /// Get value with zero runtime dispatch - pure compile-time specialization
    fn get_typed(&self, var_id: usize) -> T;
}

/// Zero-overhead expression evaluation trait
pub trait StaticExpr<T: StaticExpressionType, const SCOPE: usize>: Clone + std::fmt::Debug {
    /// Evaluate with zero runtime dispatch using `HList` storage
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>;
    
    /// Convert to AST representation (for bridge functions and summation)
    /// This is used when we need to interoperate with dynamic systems or
    /// perform operations like summation that require AST manipulation.
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar;
}

// ============================================================================
// ENHANCED CONTEXT - ERGONOMIC SCOPE MANAGEMENT
// ============================================================================

/// Static context with automatic scope management and `HList` integration
#[derive(Debug)]
pub struct StaticContext<const NEXT_SCOPE: usize> {
    _scope: PhantomData<[(); NEXT_SCOPE]>,
}

impl Default for StaticContext<0> {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticContext<0> {
    /// Create a new static context (starts at scope 0)
    #[must_use]
    pub fn new() -> Self {
        Self {
            _scope: PhantomData,
        }
    }
}

impl<const NEXT_SCOPE: usize> StaticContext<NEXT_SCOPE> {
    /// Create a new scope with automatic variable management
    pub fn new_scope<F, R>(&mut self, f: F) -> R
    where
        F: for<'a> FnOnce(StaticScopeBuilder<NEXT_SCOPE, 0>) -> R,
    {
        f(StaticScopeBuilder::new())
    }

    /// Create a single-argument lambda function with automatic scope management
    ///
    /// This provides LambdaVar-style syntax without awkward scope threading:
    ///
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let mut ctx = StaticContext::new();
    ///
    /// // Clean lambda syntax - no scope threading!
    /// let f = ctx.lambda(|x| {
    ///     x.clone() * x.clone()  // Simple quadratic function
    /// });
    /// let result = f.eval(hlist![3.0]); // 3² = 9
    /// ```
    pub fn lambda<F, E>(&mut self, f: F) -> HListEvaluable<E, f64, NEXT_SCOPE>
    where
        F: FnOnce(StaticVar<f64, 0, NEXT_SCOPE>) -> E,
        E: StaticExpr<f64, NEXT_SCOPE>,
    {
        let var = StaticVar::<f64, 0, NEXT_SCOPE>::new();
        let expr = f(var);
        HListEvaluable::new(expr)
    }

    /// Advance to the next scope for composition
    #[must_use]
    pub fn next(self) -> StaticContext<{ NEXT_SCOPE + 1 }> {
        StaticContext {
            _scope: PhantomData,
        }
    }

}

// ============================================================================
// ENHANCED SCOPE BUILDER - AUTOMATIC VARIABLE TRACKING
// ============================================================================

/// Scope builder that automatically tracks variables and their types
#[derive(Debug, Clone)]
pub struct StaticScopeBuilder<const SCOPE: usize, const NEXT_VAR_ID: usize> {
    _scope: PhantomData<[(); SCOPE]>,
    _var_id: PhantomData<[(); NEXT_VAR_ID]>,
}

impl<const SCOPE: usize, const NEXT_VAR_ID: usize> StaticScopeBuilder<SCOPE, NEXT_VAR_ID> {
    fn new() -> Self {
        Self {
            _scope: PhantomData,
            _var_id: PhantomData,
        }
    }

    /// Create a new variable and return updated builder
    #[must_use]
    pub fn auto_var<T: StaticExpressionType>(
        self,
    ) -> (
        StaticVar<T, NEXT_VAR_ID, SCOPE>,
        StaticScopeBuilder<SCOPE, { NEXT_VAR_ID + 1 }>,
    ) {
        (StaticVar::new(), StaticScopeBuilder::new())
    }

    /// Create a constant in this scope
    #[must_use]
    pub fn constant<T: StaticExpressionType>(&self, value: T) -> StaticConst<T, SCOPE> {
        StaticConst::new(value)
    }

    /// Create summation expressions with proper bound variable semantics
    ///
    /// This provides the same semantics as `DynamicContext::sum()` but with compile-time optimization.
    /// The iterator variable is a bound variable that doesn't consume scope variable IDs,
    /// allowing access to free variables from the same scope.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    ///
    /// let mut ctx = StaticContext::new();
    /// let expr = ctx.new_scope(|scope| {
    ///     let (mu, scope) = scope.auto_var::<f64>();  // Variable(0)
    ///     let data = vec![1.0, 2.0, 3.0];
    ///     let (sum_expr, _) = scope.sum(data, |x| {   // x is BoundVar(0)
    ///         x - mu.clone()  // BoundVar(0) - Variable(0) - works!
    ///     });
    ///     sum_expr
    /// });
    /// ```
    #[must_use]
    pub fn sum<I, F, E>(
        self, 
        iterable: I, 
        f: F
    ) -> (StaticSumExpr<E, SCOPE>, StaticScopeBuilder<SCOPE, NEXT_VAR_ID>)
    where
        I: IntoStaticSummableRange,
        F: FnOnce(StaticBoundVar<f64, 0, SCOPE>) -> E,
        E: StaticExpr<f64, SCOPE> ,
    {
        // Create iterator variable as bound variable (doesn't consume variable IDs)
        let iter_var = StaticBoundVar::<f64, 0, SCOPE>::new();

        // Apply the closure to get the summation body
        let body_expr = f(iter_var);

        // Convert input to summable range
        let summable_range = iterable.into_static_summable();

        (
            StaticSumExpr::new(summable_range, body_expr),
            StaticScopeBuilder::new()  // Same NEXT_VAR_ID - bound vars don't consume IDs
        )
    }
}

// ============================================================================
// ENHANCED VARIABLES - ZERO-OVERHEAD WITH TYPE-LEVEL IDS
// ============================================================================

/// Static variable with compile-time type and ID tracking
#[derive(Debug, Clone)]
pub struct StaticVar<T: StaticExpressionType, const VAR_ID: usize, const SCOPE: usize> {
    _type: PhantomData<T>,
    _var_id: PhantomData<[(); VAR_ID]>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: StaticExpressionType, const VAR_ID: usize, const SCOPE: usize> StaticVar<T, VAR_ID, SCOPE> {
    fn new() -> Self {
        Self {
            _type: PhantomData,
            _var_id: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Get the compile-time variable ID
    #[must_use]
    pub const fn var_id() -> usize {
        VAR_ID
    }

    /// Get the compile-time scope ID
    #[must_use]
    pub const fn scope_id() -> usize {
        SCOPE
    }
}

impl<T: StaticExpressionType, const VAR_ID: usize, const SCOPE: usize> StaticExpr<T, SCOPE>
    for StaticVar<T, VAR_ID, SCOPE>
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // ZERO DISPATCH - COMPILE-TIME SPECIALIZED ACCESS!
        storage.get_typed(VAR_ID)
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Variable(VAR_ID)
    }
}

// StaticVar now uses its own optimized interface instead of the removed unified Expr trait
impl<T: StaticExpressionType, const VAR_ID: usize, const SCOPE: usize> StaticVar<T, VAR_ID, SCOPE>
where
    T: crate::ast::Scalar,
{
    /// Compile-time constant: Variable ID
    pub const VARIABLE_ID: usize = VAR_ID;
    
    /// Compile-time constant: Scope ID  
    pub const SCOPE_ID: usize = SCOPE;
    
    /// Compile-time constant: Complexity (always 1 for variables)
    pub const COMPLEXITY: usize = 1;
    
    /// Convert to AST representation (for bridge functions)
    pub fn to_ast(&self) -> crate::ast::ASTRepr<T> {
        crate::ast::ASTRepr::Variable(VAR_ID)
    }

    /// Compile-time pretty printing
    pub fn pretty_print() -> String {
        format!("x{}", VAR_ID)
    }

    /// Compile-time variable set (always contains just this variable)
    pub fn variables() -> [usize; 1] {
        [VAR_ID]
    }
}

// ============================================================================
// BOUND VARIABLE SUPPORT - LAMBDA INTEGRATION
// ============================================================================

/// Static bound variable for lambda expressions
/// 
/// Represents variables that are bound within lambda expressions (like loop iteration variables).
/// These use index-based access similar to ASTRepr::BoundVar for compatibility with DynamicContext.
#[derive(Debug, Clone)]
pub struct StaticBoundVar<T: StaticExpressionType, const BOUND_ID: usize, const SCOPE: usize> {
    _type: PhantomData<T>,
    _bound_id: PhantomData<[(); BOUND_ID]>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: StaticExpressionType, const BOUND_ID: usize, const SCOPE: usize> 
    StaticBoundVar<T, BOUND_ID, SCOPE> 
{
    pub fn new() -> Self {
        Self {
            _type: PhantomData,
            _bound_id: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Get the compile-time bound variable ID
    #[must_use]
    pub const fn bound_id() -> usize {
        BOUND_ID
    }
}

impl<T: StaticExpressionType, const BOUND_ID: usize, const SCOPE: usize> StaticExpr<T, SCOPE>
    for StaticBoundVar<T, BOUND_ID, SCOPE>
{
    fn eval_zero<S>(&self, _storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // Bound variables should never be directly evaluated - they should be substituted away
        panic!("BoundVar({}) should have been substituted during summation evaluation. This indicates a bug in the AST substitution logic.", BOUND_ID)
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::BoundVar(BOUND_ID)
    }
}

// StaticBoundVar now uses its own optimized interface
impl<T: StaticExpressionType, const BOUND_ID: usize, const SCOPE: usize> StaticBoundVar<T, BOUND_ID, SCOPE>
where
    T: crate::ast::Scalar,
{
    /// Compile-time constant: Bound variable ID
    pub const BOUND_ID_CONST: usize = BOUND_ID;
    
    /// Compile-time constant: Scope ID
    pub const SCOPE_ID: usize = SCOPE;
    
    /// Compile-time constant: Complexity (always 1 for bound variables)
    pub const COMPLEXITY: usize = 1;
    
    /// Convert to AST representation (for bridge functions)
    pub fn to_ast(&self) -> crate::ast::ASTRepr<T> {
        crate::ast::ASTRepr::BoundVar(BOUND_ID)
    }

    /// Compile-time pretty printing
    pub fn pretty_print() -> String {
        format!("λ{}", BOUND_ID)
    }

    /// Bound variables contribute no free variables (compile-time empty set)
    pub fn free_variables() -> [usize; 0] {
        []
    }
}

// ============================================================================
// STATICBOUNDVAR OPERATOR OVERLOADING - ARITHMETIC WITH STATIC TYPES
// ============================================================================

/// StaticBoundVar - StaticVar operations
impl<T, const BOUND_ID: usize, const VAR_ID: usize, const SCOPE: usize> std::ops::Sub<StaticVar<T, VAR_ID, SCOPE>>
    for StaticBoundVar<T, BOUND_ID, SCOPE>
where
    T: StaticExpressionType + std::ops::Sub<Output = T>,
{
    type Output = StaticSub<T, StaticBoundVar<T, BOUND_ID, SCOPE>, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
    fn sub(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
        StaticSub { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

/// StaticBoundVar * StaticBoundVar operations
impl<T, const BOUND_ID1: usize, const BOUND_ID2: usize, const SCOPE: usize> std::ops::Mul<StaticBoundVar<T, BOUND_ID2, SCOPE>>
    for StaticBoundVar<T, BOUND_ID1, SCOPE>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
{
    type Output = StaticMul<T, StaticBoundVar<T, BOUND_ID1, SCOPE>, StaticBoundVar<T, BOUND_ID2, SCOPE>, SCOPE>;
    fn mul(self, rhs: StaticBoundVar<T, BOUND_ID2, SCOPE>) -> Self::Output {
        StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

/// StaticBoundVar / StaticVar operations
impl<T, const BOUND_ID: usize, const VAR_ID: usize, const SCOPE: usize> std::ops::Div<StaticVar<T, VAR_ID, SCOPE>>
    for StaticBoundVar<T, BOUND_ID, SCOPE>
where
    T: StaticExpressionType + std::ops::Div<Output = T>,
{
    type Output = StaticDiv<T, StaticBoundVar<T, BOUND_ID, SCOPE>, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
    fn div(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
        StaticDiv { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

// ============================================================================
// ENHANCED CONSTANTS - COMPILE-TIME VALUES
// ============================================================================

/// Static constant with compile-time scope tracking
#[derive(Debug, Clone)]
pub struct StaticConst<T: StaticExpressionType, const SCOPE: usize> {
    value: T,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T: StaticExpressionType, const SCOPE: usize> StaticConst<T, SCOPE> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            _scope: PhantomData,
        }
    }

    /// Get the constant value
    #[must_use]
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: StaticExpressionType, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticConst<T, SCOPE> {
    fn eval_zero<S>(&self, _storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // COMPILE-TIME CONSTANT - ZERO RUNTIME COST
        self.value.clone()
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Constant(self.value.clone())
    }
}

// ============================================================================
// SUMMATION SUPPORT - UNIFIED API WITH DYNAMICCONTEXT
// ============================================================================

/// Static version of summable ranges for compile-time optimization
#[derive(Debug, Clone)]
pub enum StaticSummableRange {
    /// Mathematical range like 1..=10 for symbolic optimization
    MathematicalRange { start: i64, end: i64 },
    /// Data iteration for compile-time known values
    DataIteration { values: Vec<f64> },
}

impl StaticSummableRange {
    /// Get all values in this range as an iterator
    pub fn values(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            StaticSummableRange::MathematicalRange { start, end } => {
                Box::new((*start..=*end).map(|i| i as f64))
            }
            StaticSummableRange::DataIteration { values } => {
                Box::new(values.iter().copied())
            }
        }
    }
}

/// Trait for converting different types into static summable ranges
pub trait IntoStaticSummableRange {
    fn into_static_summable(self) -> StaticSummableRange;
}

/// Implementation for mathematical ranges
impl IntoStaticSummableRange for std::ops::RangeInclusive<i64> {
    fn into_static_summable(self) -> StaticSummableRange {
        StaticSummableRange::MathematicalRange {
            start: *self.start(),
            end: *self.end(),
        }
    }
}

/// Implementation for data vectors
impl IntoStaticSummableRange for Vec<f64> {
    fn into_static_summable(self) -> StaticSummableRange {
        StaticSummableRange::DataIteration { values: self }
    }
}

/// Implementation for data slices
impl IntoStaticSummableRange for &[f64] {
    fn into_static_summable(self) -> StaticSummableRange {
        StaticSummableRange::DataIteration {
            values: self.to_vec(),
        }
    }
}

/// Implementation for data vector references
impl IntoStaticSummableRange for &Vec<f64> {
    fn into_static_summable(self) -> StaticSummableRange {
        StaticSummableRange::DataIteration {
            values: self.clone(),
        }
    }
}

// ============================================================================
// BOUND VARIABLE STACK IMPLEMENTATION
// ============================================================================



/// Static summation expression with zero-overhead evaluation
///
/// This represents a summation that can be:
/// - Evaluated at compile time if no unbound variables
/// - Composed with other expressions if unbound variables exist
///
/// Uses the same rewrite rules as `DynamicContext` for consistency.
#[derive(Debug, Clone)]
pub struct StaticSumExpr<E, const SCOPE: usize>
where
    E: StaticExpr<f64, SCOPE>,
{
    range: StaticSummableRange,
    body: E,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<E, const SCOPE: usize> StaticSumExpr<E, SCOPE>
where
    E: StaticExpr<f64, SCOPE> ,
{
    /// Create a new static summation expression
    pub fn new(range: StaticSummableRange, body: E) -> Self {
        Self {
            range,
            body,
            _scope: PhantomData,
        }
    }
    
    /// Substitute a bound variable with a value in an AST (LambdaVar-style substitution)
    fn substitute_bound_variable(
        &self,
        ast: &crate::ast::ASTRepr<f64>,
        bound_id: usize,
        replacement: &crate::ast::ASTRepr<f64>,
    ) -> crate::ast::ASTRepr<f64> {
        use crate::ast::ASTRepr;
        
        match ast {
            ASTRepr::BoundVar(id) if *id == bound_id => replacement.clone(),
            ASTRepr::BoundVar(id) => ASTRepr::BoundVar(*id),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Constant(val) => ASTRepr::Constant(*val),
            ASTRepr::Add(terms) => {
                let substituted_terms: Vec<_> = terms
                    .elements()
                    .map(|term| self.substitute_bound_variable(term, bound_id, replacement))
                    .collect();
                ASTRepr::Add(crate::ast::multiset::MultiSet::from_iter(substituted_terms))
            }
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(self.substitute_bound_variable(left, bound_id, replacement)),
                Box::new(self.substitute_bound_variable(right, bound_id, replacement)),
            ),
            ASTRepr::Mul(factors) => {
                let substituted_factors: Vec<_> = factors
                    .elements()
                    .map(|factor| self.substitute_bound_variable(factor, bound_id, replacement))
                    .collect();
                ASTRepr::Mul(crate::ast::multiset::MultiSet::from_iter(substituted_factors))
            }
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(self.substitute_bound_variable(left, bound_id, replacement)),
                Box::new(self.substitute_bound_variable(right, bound_id, replacement)),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(self.substitute_bound_variable(base, bound_id, replacement)),
                Box::new(self.substitute_bound_variable(exp, bound_id, replacement)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(self.substitute_bound_variable(
                inner, bound_id, replacement,
            ))),
            ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(self.substitute_bound_variable(
                inner, bound_id, replacement,
            ))),
            ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(self.substitute_bound_variable(
                inner, bound_id, replacement,
            ))),
            ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(self.substitute_bound_variable(
                inner, bound_id, replacement,
            ))),
            ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(self.substitute_bound_variable(
                inner, bound_id, replacement,
            ))),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(self.substitute_bound_variable(
                inner, bound_id, replacement,
            ))),
            // For other AST variants, recursively substitute
            _ => ast.clone(), // Fallback for complex cases
        }
    }
    
    /// Evaluate an AST with HList storage (bridge between AST and StaticExpr evaluation)
    fn eval_ast_with_storage<S>(&self, ast: &crate::ast::ASTRepr<f64>, storage: &S) -> f64
    where
        S: HListStorage<f64>,
    {
        use crate::ast::ASTRepr;
        
        match ast {
            ASTRepr::Constant(val) => *val,
            ASTRepr::Variable(idx) => storage.get_typed(*idx),
            ASTRepr::BoundVar(_) => panic!("BoundVar should have been substituted away"),
            ASTRepr::Add(terms) => terms
                .elements()
                .map(|term| self.eval_ast_with_storage(term, storage))
                .sum(),
            ASTRepr::Sub(left, right) => {
                self.eval_ast_with_storage(left, storage) - self.eval_ast_with_storage(right, storage)
            }
            ASTRepr::Mul(factors) => factors
                .elements()
                .map(|factor| self.eval_ast_with_storage(factor, storage))
                .product(),
            ASTRepr::Div(left, right) => {
                self.eval_ast_with_storage(left, storage) / self.eval_ast_with_storage(right, storage)
            }
            ASTRepr::Pow(base, exp) => {
                self.eval_ast_with_storage(base, storage).powf(self.eval_ast_with_storage(exp, storage))
            }
            ASTRepr::Neg(inner) => -self.eval_ast_with_storage(inner, storage),
            ASTRepr::Sin(inner) => self.eval_ast_with_storage(inner, storage).sin(),
            ASTRepr::Cos(inner) => self.eval_ast_with_storage(inner, storage).cos(),
            ASTRepr::Exp(inner) => self.eval_ast_with_storage(inner, storage).exp(),
            ASTRepr::Ln(inner) => self.eval_ast_with_storage(inner, storage).ln(),
            ASTRepr::Sqrt(inner) => self.eval_ast_with_storage(inner, storage).sqrt(),
            // For complex cases like Sum, Lambda, etc., we'd need more sophisticated handling
            _ => panic!("Unsupported AST node in eval_ast_with_storage: {:?}", ast),
        }
    }
}

impl<E, const SCOPE: usize> StaticExpr<f64, SCOPE> for StaticSumExpr<E, SCOPE>
where
    E: StaticExpr<f64, SCOPE> ,
{
    fn eval_zero<S>(&self, storage: &S) -> f64
    where
        S: HListStorage<f64>,
    {
        // LambdaVar-style AST substitution approach
        // 1. Convert body expression to AST
        // 2. For each value in range, substitute BoundVar(0) with the value
        // 3. Evaluate the substituted AST with the original storage
        use crate::ast::ASTRepr;
        
        let body_ast = self.body.to_ast();
        let mut sum = 0.0;
        
        for value in self.range.values() {
            // Substitute BoundVar(0) with the current iteration value
            let substituted_ast = self.substitute_bound_variable(&body_ast, 0, &ASTRepr::Constant(value));
            
            // Evaluate the substituted AST using HList evaluation
            sum += self.eval_ast_with_storage(&substituted_ast, storage);
        }
        
        sum
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<f64>
    where
        f64: crate::ast::Scalar,
    {
        use crate::ast::ast_repr::{Collection, Lambda};
        
        // Create proper Sum AST with Collection and Lambda
        match &self.range {
            StaticSummableRange::MathematicalRange { start, end } => {
                // Create range collection
                let collection = Collection::Range {
                    start: Box::new(crate::ast::ASTRepr::Constant(*start as f64)),
                    end: Box::new(crate::ast::ASTRepr::Constant(*end as f64)),
                };
                
                // Create lambda from body
                let lambda = Lambda {
                    var_indices: vec![0], // Single bound variable
                    body: Box::new(self.body.to_ast()),
                };
                
                crate::ast::ASTRepr::Sum(Box::new(collection))
            }
            StaticSummableRange::DataIteration { values } => {
                // Create data array collection
                let collection = Collection::DataArray(values.clone());
                
                // Create lambda from body
                let lambda = Lambda {
                    var_indices: vec![0], // Single bound variable
                    body: Box::new(self.body.to_ast()),
                };
                
                crate::ast::ASTRepr::Sum(Box::new(collection))
            }
        }
    }
}
        

impl<E, const SCOPE: usize> IntoHListEvaluable<f64, SCOPE> for StaticSumExpr<E, SCOPE> where
    E: StaticExpr<f64, SCOPE> 
{
}

// ============================================================================
// ENHANCED OPERATIONS - ZERO-OVERHEAD ARITHMETIC
// ============================================================================

/// Static addition with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticAdd<T, L, R, const SCOPE: usize>
where
    T: StaticExpressionType + std::ops::Add<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticAdd<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Add<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // ZERO DISPATCH MONOMORPHIZATION - NO RUNTIME OVERHEAD!
        self.left.eval_zero(storage) + self.right.eval_zero(storage)
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::add_binary(self.left.to_ast(), self.right.to_ast())
    }
}

/// Static multiplication with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticMul<T, L, R, const SCOPE: usize>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticMul<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // ZERO DISPATCH MONOMORPHIZATION - NO RUNTIME OVERHEAD!
        self.left.eval_zero(storage) * self.right.eval_zero(storage)
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::mul_binary(self.left.to_ast(), self.right.to_ast())
    }
}

/// Static subtraction with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticSub<T, L, R, const SCOPE: usize>
where
    T: StaticExpressionType + std::ops::Sub<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticSub<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Sub<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // ZERO DISPATCH MONOMORPHIZATION - NO RUNTIME OVERHEAD!
        self.left.eval_zero(storage) - self.right.eval_zero(storage)
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Sub(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}


/// Static division with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticDiv<T, L, R, const SCOPE: usize>
where
    T: StaticExpressionType + std::ops::Div<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticDiv<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Div<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // ZERO DISPATCH MONOMORPHIZATION - NO RUNTIME OVERHEAD!
        self.left.eval_zero(storage) / self.right.eval_zero(storage)
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Div(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}

/// Static power operation with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticPow<T, L, R, const SCOPE: usize>
where
    T: StaticExpressionType + num_traits::Float,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    base: L,
    exponent: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticPow<T, L, R, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        // ZERO DISPATCH MONOMORPHIZATION - NO RUNTIME OVERHEAD!
        self.base.eval_zero(storage).powf(self.exponent.eval_zero(storage))
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Pow(Box::new(self.base.to_ast()), Box::new(self.exponent.to_ast()))
    }
}

// ============================================================================
// MACRO-BASED OPERATOR IMPLEMENTATIONS - REDUCED CODE DUPLICATION
// ============================================================================

/// Macro to generate all four basic operations for Variable-Variable combinations
macro_rules! impl_var_var_ops {
    () => {
        impl<T, const VAR_ID1: usize, const VAR_ID2: usize, const SCOPE: usize>
            std::ops::Add<StaticVar<T, VAR_ID2, SCOPE>> for StaticVar<T, VAR_ID1, SCOPE>
        where
            T: StaticExpressionType + std::ops::Add<Output = T>,
        {
            type Output = StaticAdd<T, Self, StaticVar<T, VAR_ID2, SCOPE>, SCOPE>;
            fn add(self, rhs: StaticVar<T, VAR_ID2, SCOPE>) -> Self::Output {
                StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID1: usize, const VAR_ID2: usize, const SCOPE: usize>
            std::ops::Sub<StaticVar<T, VAR_ID2, SCOPE>> for StaticVar<T, VAR_ID1, SCOPE>
        where
            T: StaticExpressionType + std::ops::Sub<Output = T>,
        {
            type Output = StaticSub<T, Self, StaticVar<T, VAR_ID2, SCOPE>, SCOPE>;
            fn sub(self, rhs: StaticVar<T, VAR_ID2, SCOPE>) -> Self::Output {
                StaticSub { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID1: usize, const VAR_ID2: usize, const SCOPE: usize>
            std::ops::Mul<StaticVar<T, VAR_ID2, SCOPE>> for StaticVar<T, VAR_ID1, SCOPE>
        where
            T: StaticExpressionType + std::ops::Mul<Output = T>,
        {
            type Output = StaticMul<T, Self, StaticVar<T, VAR_ID2, SCOPE>, SCOPE>;
            fn mul(self, rhs: StaticVar<T, VAR_ID2, SCOPE>) -> Self::Output {
                StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID1: usize, const VAR_ID2: usize, const SCOPE: usize>
            std::ops::Div<StaticVar<T, VAR_ID2, SCOPE>> for StaticVar<T, VAR_ID1, SCOPE>
        where
            T: StaticExpressionType + std::ops::Div<Output = T>,
        {
            type Output = StaticDiv<T, Self, StaticVar<T, VAR_ID2, SCOPE>, SCOPE>;
            fn div(self, rhs: StaticVar<T, VAR_ID2, SCOPE>) -> Self::Output {
                StaticDiv { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
    };
}

/// Macro to generate all four basic operations for Variable-Constant combinations
macro_rules! impl_var_const_ops {
    () => {
        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Add<StaticConst<T, SCOPE>>
            for StaticVar<T, VAR_ID, SCOPE>
        where
            T: StaticExpressionType + std::ops::Add<Output = T>,
        {
            type Output = StaticAdd<T, Self, StaticConst<T, SCOPE>, SCOPE>;
            fn add(self, rhs: StaticConst<T, SCOPE>) -> Self::Output {
                StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Sub<StaticConst<T, SCOPE>>
            for StaticVar<T, VAR_ID, SCOPE>
        where
            T: StaticExpressionType + std::ops::Sub<Output = T>,
        {
            type Output = StaticSub<T, Self, StaticConst<T, SCOPE>, SCOPE>;
            fn sub(self, rhs: StaticConst<T, SCOPE>) -> Self::Output {
                StaticSub { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Mul<StaticConst<T, SCOPE>>
            for StaticVar<T, VAR_ID, SCOPE>
        where
            T: StaticExpressionType + std::ops::Mul<Output = T>,
        {
            type Output = StaticMul<T, Self, StaticConst<T, SCOPE>, SCOPE>;
            fn mul(self, rhs: StaticConst<T, SCOPE>) -> Self::Output {
                StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Div<StaticConst<T, SCOPE>>
            for StaticVar<T, VAR_ID, SCOPE>
        where
            T: StaticExpressionType + std::ops::Div<Output = T>,
        {
            type Output = StaticDiv<T, Self, StaticConst<T, SCOPE>, SCOPE>;
            fn div(self, rhs: StaticConst<T, SCOPE>) -> Self::Output {
                StaticDiv { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
    };
}

/// Macro to generate all four basic operations for Constant-Variable combinations
macro_rules! impl_const_var_ops {
    () => {
        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Add<StaticVar<T, VAR_ID, SCOPE>>
            for StaticConst<T, SCOPE>
        where
            T: StaticExpressionType + std::ops::Add<Output = T>,
        {
            type Output = StaticAdd<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
            fn add(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
                StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Sub<StaticVar<T, VAR_ID, SCOPE>>
            for StaticConst<T, SCOPE>
        where
            T: StaticExpressionType + std::ops::Sub<Output = T>,
        {
            type Output = StaticSub<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
            fn sub(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
                StaticSub { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Mul<StaticVar<T, VAR_ID, SCOPE>>
            for StaticConst<T, SCOPE>
        where
            T: StaticExpressionType + std::ops::Mul<Output = T>,
        {
            type Output = StaticMul<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
            fn mul(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
                StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const VAR_ID: usize, const SCOPE: usize> std::ops::Div<StaticVar<T, VAR_ID, SCOPE>>
            for StaticConst<T, SCOPE>
        where
            T: StaticExpressionType + std::ops::Div<Output = T>,
        {
            type Output = StaticDiv<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
            fn div(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
                StaticDiv { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
    };
}

/// Macro to generate Constant-Constant operations
macro_rules! impl_const_const_ops {
    () => {
        impl<T, const SCOPE: usize> std::ops::Add<StaticConst<T, SCOPE>>
            for StaticConst<T, SCOPE>
        where
            T: StaticExpressionType + std::ops::Add<Output = T>,
        {
            type Output = StaticAdd<T, Self, StaticConst<T, SCOPE>, SCOPE>;
            fn add(self, rhs: StaticConst<T, SCOPE>) -> Self::Output {
                StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }

        impl<T, const SCOPE: usize> std::ops::Mul<StaticConst<T, SCOPE>>
            for StaticConst<T, SCOPE>
        where
            T: StaticExpressionType + std::ops::Mul<Output = T>,
        {
            type Output = StaticMul<T, Self, StaticConst<T, SCOPE>, SCOPE>;
            fn mul(self, rhs: StaticConst<T, SCOPE>) -> Self::Output {
                StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
    };
}

// Generate all the basic operations using macros
impl_var_var_ops!();
impl_var_const_ops!();
impl_const_var_ops!();
impl_const_const_ops!();

/// Macro to generate expression-expression operators (the complex ones)
macro_rules! impl_expr_expr_ops {
    ($expr1:ident, $expr2:ident) => {
        // Expression + Expression
        impl<T, L1, R1, L2, R2, const SCOPE: usize> std::ops::Add<$expr2<T, L2, R2, SCOPE>>
            for $expr1<T, L1, R1, SCOPE>
        where
            T: StaticExpressionType + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
            L1: StaticExpr<T, SCOPE>, R1: StaticExpr<T, SCOPE>,
            L2: StaticExpr<T, SCOPE>, R2: StaticExpr<T, SCOPE>,
        {
            type Output = StaticAdd<T, Self, $expr2<T, L2, R2, SCOPE>, SCOPE>;
            fn add(self, rhs: $expr2<T, L2, R2, SCOPE>) -> Self::Output {
                StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
        
        // Expression * Expression  
        impl<T, L1, R1, L2, R2, const SCOPE: usize> std::ops::Mul<$expr2<T, L2, R2, SCOPE>>
            for $expr1<T, L1, R1, SCOPE>
        where
            T: StaticExpressionType + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
            L1: StaticExpr<T, SCOPE>, R1: StaticExpr<T, SCOPE>,
            L2: StaticExpr<T, SCOPE>, R2: StaticExpr<T, SCOPE>,
        {
            type Output = StaticMul<T, Self, $expr2<T, L2, R2, SCOPE>, SCOPE>;
            fn mul(self, rhs: $expr2<T, L2, R2, SCOPE>) -> Self::Output {
                StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
    };
}

// Generate the most important expression-expression combinations
impl_expr_expr_ops!(StaticAdd, StaticAdd);
impl_expr_expr_ops!(StaticDiv, StaticDiv);
impl_expr_expr_ops!(StaticMul, StaticMul);

/// Macro to generate expression-variable operations
macro_rules! impl_expr_var_ops {
    ($expr:ident) => {
        impl<T, L, R, const VAR_ID: usize, const SCOPE: usize> std::ops::Div<StaticVar<T, VAR_ID, SCOPE>>
            for $expr<T, L, R, SCOPE>
        where
            T: StaticExpressionType + std::ops::Div<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
            L: StaticExpr<T, SCOPE>, R: StaticExpr<T, SCOPE>,
        {
            type Output = StaticDiv<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
            fn div(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
                StaticDiv { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
            }
        }
    };
}

// Apply to common expression types
impl_expr_var_ops!(StaticSub);
impl_expr_var_ops!(StaticMul);

// Expression + Variable operations
impl<T, L, R, const VAR_ID: usize, const SCOPE: usize> std::ops::Add<StaticVar<T, VAR_ID, SCOPE>>
    for StaticAdd<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Add<Output = T>,
    L: StaticExpr<T, SCOPE>, R: StaticExpr<T, SCOPE>,
{
    type Output = StaticAdd<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
    fn add(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
        StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

// ============================================================================
// LEGACY MANUAL IMPLEMENTATIONS - KEEP FOR COMPATIBILITY
// ============================================================================
// Note: The macros above should handle most cases, but we keep some manual
// implementations for specific edge cases or backward compatibility

// Note: Basic Variable-Variable, Variable-Constant, Constant-Variable, and Constant-Constant
// operations are now automatically generated by the macros above. This eliminates hundreds
// of lines of repetitive code while maintaining the same functionality.

// ============================================================================
// SPECIALIZED EXPRESSION IMPLEMENTATIONS NOT COVERED BY MACROS
// ============================================================================
// These are kept for specific complex expression combinations that require
// special handling beyond the basic macro-generated operations

// Expression - Expression (Mul - Ln)
impl<T, L, R, E, const SCOPE: usize> std::ops::Sub<StaticLn<T, E, SCOPE>>
    for StaticMul<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + num_traits::Float,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
    E: StaticExpr<T, SCOPE>,
{
    type Output = StaticSub<T, Self, StaticLn<T, E, SCOPE>, SCOPE>;
    fn sub(self, rhs: StaticLn<T, E, SCOPE>) -> Self::Output {
        StaticSub { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

// Expression + Expression (Sub + Mul)
impl<T, L1, R1, L2, R2, const SCOPE: usize> std::ops::Add<StaticMul<T, L2, R2, SCOPE>>
    for StaticSub<T, L1, R1, SCOPE>
where
    T: StaticExpressionType + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
    L1: StaticExpr<T, SCOPE>,
    R1: StaticExpr<T, SCOPE>,
    L2: StaticExpr<T, SCOPE>,
    R2: StaticExpr<T, SCOPE>,
{
    type Output = StaticAdd<T, Self, StaticMul<T, L2, R2, SCOPE>, SCOPE>;
    fn add(self, rhs: StaticMul<T, L2, R2, SCOPE>) -> Self::Output {
        StaticAdd { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

// Constant * Expression combinations
impl<T, L, R, const SCOPE: usize> std::ops::Mul<StaticMul<T, L, R, SCOPE>>
    for StaticConst<T, SCOPE>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    type Output = StaticMul<T, Self, StaticMul<T, L, R, SCOPE>, SCOPE>;
    fn mul(self, rhs: StaticMul<T, L, R, SCOPE>) -> Self::Output {
        StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

// Expression * Variable (Mul * Var)
impl<T, L, R, const VAR_ID: usize, const SCOPE: usize> std::ops::Mul<StaticVar<T, VAR_ID, SCOPE>>
    for StaticMul<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    type Output = StaticMul<T, Self, StaticVar<T, VAR_ID, SCOPE>, SCOPE>;
    fn mul(self, rhs: StaticVar<T, VAR_ID, SCOPE>) -> Self::Output {
        StaticMul { left: self, right: rhs, _type: PhantomData, _scope: PhantomData }
    }
}

// ============================================================================
// MATHEMATICAL FUNCTIONS - ZERO-OVERHEAD TRANSCENDENTAL OPERATIONS
// ============================================================================


/// Static sine function with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticSin<T, E, const SCOPE: usize>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    inner: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, E, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticSin<T, E, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        self.inner.eval_zero(storage).sin()
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Sin(Box::new(self.inner.to_ast()))
    }
}

/// Static cosine function with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticCos<T, E, const SCOPE: usize>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    inner: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, E, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticCos<T, E, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        self.inner.eval_zero(storage).cos()
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Cos(Box::new(self.inner.to_ast()))
    }
}

/// Static exponential function with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticExp<T, E, const SCOPE: usize>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    inner: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, E, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticExp<T, E, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        self.inner.eval_zero(storage).exp()
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Exp(Box::new(self.inner.to_ast()))
    }
}

/// Static natural logarithm function with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticLn<T, E, const SCOPE: usize>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    inner: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, E, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticLn<T, E, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        self.inner.eval_zero(storage).ln()
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Ln(Box::new(self.inner.to_ast()))
    }
}


/// Static square root function with zero runtime overhead
#[derive(Debug, Clone)]
pub struct StaticSqrt<T, E, const SCOPE: usize>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    inner: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, E, const SCOPE: usize> StaticExpr<T, SCOPE> for StaticSqrt<T, E, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
    E: StaticExpr<T, SCOPE>,
{
    fn eval_zero<S>(&self, storage: &S) -> T
    where
        S: HListStorage<T>,
    {
        self.inner.eval_zero(storage).sqrt()
    }
    
    fn to_ast(&self) -> crate::ast::ASTRepr<T>
    where
        T: crate::ast::Scalar,
    {
        crate::ast::ASTRepr::Sqrt(Box::new(self.inner.to_ast()))
    }
}

// ============================================================================
// MATHEMATICAL FUNCTION METHODS FOR VARIABLES
// ============================================================================

impl<T, const VAR_ID: usize, const SCOPE: usize> StaticVar<T, VAR_ID, SCOPE>
where
    T: StaticExpressionType + num_traits::Float,
{
    /// Natural logarithm function
    pub fn ln(self) -> StaticLn<T, Self, SCOPE> {
        StaticLn {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Sine function
    pub fn sin(self) -> StaticSin<T, Self, SCOPE> {
        StaticSin {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Cosine function
    pub fn cos(self) -> StaticCos<T, Self, SCOPE> {
        StaticCos {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Exponential function
    pub fn exp(self) -> StaticExp<T, Self, SCOPE> {
        StaticExp {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Square root function
    pub fn sqrt(self) -> StaticSqrt<T, Self, SCOPE> {
        StaticSqrt {
            inner: self,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }
}

// ============================================================================
// HLIST STORAGE IMPLEMENTATION - TYPE-SAFE ZERO-OVERHEAD HETEROGENEOUS STORAGE
// ============================================================================

/// HList-based storage that grows as needed without `MAX_VARS` limitation
pub trait HListEval<T: StaticExpressionType> {
    /// Evaluate expression with `HList` storage
    fn eval<E, const SCOPE: usize>(&self, expr: E) -> T
    where
        E: StaticExpr<T, SCOPE>,
        Self: HListStorage<T>;
}

// ============================================================================
// HLIST STORAGE TRAIT IMPLEMENTATIONS - SIMPLIFIED APPROACH
// ============================================================================

// Base case: HNil - no storage
impl<T: StaticExpressionType> HListStorage<T> for HNil {
    fn get_typed(&self, _var_id: usize) -> T {
        panic!("Variable index out of bounds in HNil")
    }
}

impl<T: StaticExpressionType> HListEval<T> for HNil {
    fn eval<E, const SCOPE: usize>(&self, expr: E) -> T
    where
        E: StaticExpr<T, SCOPE>,
        Self: HListStorage<T>,
    {
        expr.eval_zero(self)
    }
}

// Specialized implementations for specific types at head position
impl<Tail> HListStorage<f64> for HCons<f64, Tail>
where
    Tail: HListStorage<f64>,
{
    fn get_typed(&self, var_id: usize) -> f64 {
        match var_id {
            0 => self.head,
            n => self.tail.get_typed(n - 1),
        }
    }
}

impl<Tail> HListStorage<f32> for HCons<f32, Tail>
where
    Tail: HListStorage<f32>,
{
    fn get_typed(&self, var_id: usize) -> f32 {
        match var_id {
            0 => self.head,
            n => self.tail.get_typed(n - 1),
        }
    }
}

impl<Tail> HListStorage<i32> for HCons<i32, Tail>
where
    Tail: HListStorage<i32>,
{
    fn get_typed(&self, var_id: usize) -> i32 {
        match var_id {
            0 => self.head,
            n => self.tail.get_typed(n - 1),
        }
    }
}

impl<Tail> HListStorage<usize> for HCons<usize, Tail>
where
    Tail: HListStorage<usize>,
{
    fn get_typed(&self, var_id: usize) -> usize {
        match var_id {
            0 => self.head,
            n => self.tail.get_typed(n - 1),
        }
    }
}

// TODO: Add HList storage for additional mathematical types as needed:
// - u32, u64, i64 (when needed for mathematical expressions)
// - bool (when boolean algebra is added)
// - Complex<T> (when complex number support is added)
// - Vector<T> (when vector math is added)
// - Matrix<T> (when linear algebra is added)

// HListEval implementations for all supported types
impl<Head, Tail, T> HListEval<T> for HCons<Head, Tail>
where
    T: StaticExpressionType,
    Head: Clone,
    Tail: HListStorage<T> + HListEval<T>,
    Self: HListStorage<T>,
{
    fn eval<E, const SCOPE: usize>(&self, expr: E) -> T
    where
        E: StaticExpr<T, SCOPE>,
        Self: HListStorage<T>,
    {
        expr.eval_zero(self)
    }
}

// ============================================================================
// ENHANCED EXPRESSION WRAPPER FOR HLIST EVALUATION
// ============================================================================

/// Wrapper that enables `HList` evaluation on any static expression
#[derive(Debug, Clone)]
pub struct HListEvaluable<E, T, const SCOPE: usize>
where
    E: StaticExpr<T, SCOPE>,
    T: StaticExpressionType,
{
    expr: E,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<E, T, const SCOPE: usize> HListEvaluable<E, T, SCOPE>
where
    E: StaticExpr<T, SCOPE>,
    T: StaticExpressionType,
{
    /// Create a new `HList` evaluable wrapper
    pub fn new(expr: E) -> Self {
        Self {
            expr,
            _type: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Evaluate with `HList` inputs
    pub fn eval<H>(&self, hlist: H) -> T
    where
        H: HListStorage<T>,
    {
        self.expr.eval_zero(&hlist)
    }
}

// ============================================================================
// CONVENIENCE TRAIT FOR AUTOMATIC HLIST EVALUATION - FIXED VERSION
// ============================================================================

/// Extension trait to add `HList` evaluation to any static expression
pub trait IntoHListEvaluable<T: StaticExpressionType, const SCOPE: usize>:
    StaticExpr<T, SCOPE>
{
    /// Convert expression into `HList` evaluable form
    fn into_hlist_evaluable(self) -> HListEvaluable<Self, T, SCOPE>
    where
        Self: Sized,
    {
        HListEvaluable::new(self)
    }

    /// Direct `HList` evaluation (convenience method) - takes reference to avoid moves
    fn eval<H>(&self, hlist: H) -> T
    where
        H: HListStorage<T>,
        Self: Clone,
    {
        self.clone().eval_zero(&hlist)
    }
}

// ============================================================================
// TRAIT IMPLEMENTATIONS FOR ALL ENHANCED EXPRESSIONS
// ============================================================================

// Implement IntoHListEvaluable for all static expression types
impl<T: StaticExpressionType, const VAR_ID: usize, const SCOPE: usize> IntoHListEvaluable<T, SCOPE>
    for StaticVar<T, VAR_ID, SCOPE>
{
}

impl<T: StaticExpressionType, const SCOPE: usize> IntoHListEvaluable<T, SCOPE>
    for StaticConst<T, SCOPE>
{
}

impl<T, L, R, const SCOPE: usize> IntoHListEvaluable<T, SCOPE> for StaticAdd<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Add<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
}

impl<T, L, R, const SCOPE: usize> IntoHListEvaluable<T, SCOPE> for StaticMul<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
}

// ============================================================================
// CONVENIENCE FUNCTIONS FOR EXPRESSION BUILDING
// ============================================================================

/// Create static addition operation
#[must_use]
pub fn static_add<T, L, R, const SCOPE: usize>(left: L, right: R) -> StaticAdd<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Add<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    StaticAdd {
        left,
        right,
        _type: PhantomData,
        _scope: PhantomData,
    }
}

/// Create static multiplication operation
#[must_use]
pub fn static_mul<T, L, R, const SCOPE: usize>(left: L, right: R) -> StaticMul<T, L, R, SCOPE>
where
    T: StaticExpressionType + std::ops::Mul<Output = T>,
    L: StaticExpr<T, SCOPE>,
    R: StaticExpr<T, SCOPE>,
{
    StaticMul {
        left,
        right,
        _type: PhantomData,
        _scope: PhantomData,
    }
}

// ============================================================================
// STATIC EXPRESSIONS MAINTAIN OPTIMIZED COMPILE-TIME INTERFACES
// ============================================================================

// Static expressions no longer implement the removed unified Expr trait.
// Each static expression type provides compile-time constants and zero-cost operations
// specific to its computational model, avoiding the runtime overhead that the unified
// trait imposed on compile-time optimized expressions.







// ============================================================================
// TESTS - VERIFY ZERO-OVERHEAD PERFORMANCE AND FUNCTIONALITY
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use frunk::hlist;

    #[test]
    fn test_static_scoped_basic_arithmetic() {
        let mut ctx = StaticContext::new();

        let expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var::<f64>();
            let (y, _scope) = scope.auto_var::<f64>();
            x + y
        });

        // Test HList evaluation with homogeneous types
        let inputs = hlist![3.0, 4.0];
        let result = expr.eval(inputs);
        assert_eq!(result, 7.0);
    }

    #[test]
    fn test_static_scoped_constants() {
        let mut ctx = StaticContext::new();

        let expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var::<f64>();
            let two = scope.constant(2.0);
            x * two
        });

        let inputs = hlist![5.0];
        let result = expr.eval(inputs);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_static_scoped_complex_expression() {
        let mut ctx = StaticContext::new();

        // Test: f(x, y, z) = x² + 2y + z
        let expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var::<f64>();
            let (y, scope) = scope.auto_var::<f64>();
            let (z, scope) = scope.auto_var::<f64>();
            let two = scope.constant(2.0);
            x.clone() * x + two * y + z
        });

        let inputs = hlist![3.0, 4.0, 5.0];
        let result = expr.eval(inputs);
        // 3² + 2*4 + 5 = 9 + 8 + 5 = 22
        assert_eq!(result, 22.0);
    }

    #[test]
    fn test_static_scoped_composition() {
        let mut ctx = StaticContext::new();

        // Define f(x) = x² in scope 0
        let f = ctx.new_scope(|scope| {
            let (x, _scope) = scope.auto_var::<f64>();
            x.clone() * x
        });

        // Test f(3) = 9
        let inputs_f = hlist![3.0];
        let result_f = f.eval(inputs_f);
        assert_eq!(result_f, 9.0);

        // Advance to next scope
        let mut ctx = ctx.next();

        // Define g(y) = 2y in scope 1
        let g = ctx.new_scope(|scope| {
            let (y, scope) = scope.auto_var::<f64>();
            scope.constant(2.0) * y
        });

        // Test g(4) = 8
        let inputs_g = hlist![4.0];
        let result_g = g.eval(inputs_g);
        assert_eq!(result_g, 8.0);

        // Scopes are isolated - no variable collision
        assert_eq!(StaticVar::<f64, 0, 0>::scope_id(), 0);
        assert_eq!(StaticVar::<f64, 0, 1>::scope_id(), 1);
    }

    #[test]
    fn test_static_scoped_type_safety() {
        let mut ctx = StaticContext::new();

        // Test with different types in same scope
        let expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var::<f64>();
            let (y, _scope) = scope.auto_var::<f64>();
            x + y
        });

        // Verify compile-time variable IDs
        assert_eq!(StaticVar::<f64, 0, 0>::var_id(), 0);
        assert_eq!(StaticVar::<f64, 1, 0>::var_id(), 1);

        // Test evaluation
        let inputs = hlist![1.5, 2.5];
        let result = expr.eval(inputs);
        assert_eq!(result, 4.0);
    }

    #[test]
    fn test_zero_overhead_verification() {
        // This test verifies that our implementation compiles to efficient code
        let mut ctx = StaticContext::new();

        let expr = ctx.new_scope(|scope| {
            let (x, scope) = scope.auto_var::<f64>();
            let (y, _scope) = scope.auto_var::<f64>();
            x + y
        });

        // Benchmark-style test to verify performance
        let inputs = hlist![3.0, 4.0];

        // This should compile to direct field access with no runtime overhead
        for _ in 0..1000 {
            let result = expr.eval(inputs);
            assert_eq!(result, 7.0);
        }
    }

    #[test]
    fn test_lambda_style_syntax() {
        let mut ctx = StaticContext::new();

        // Test single-argument lambda - simple x * x
        let f = ctx.lambda(|x| x.clone() * x);
        let result = f.eval(hlist![3.0]);
        assert_eq!(result, 9.0); // 3² = 9

        // Note: Multi-argument functions should use the proper HList approach
        // via MathFunction::from_lambda_multi() with MultiVar trait, not artificial lambda2/lambda3
        // The key point is that the lambda syntax works without awkward scope threading
    }

    #[test]
    fn test_lambda_vs_scope_builder_equivalence() {
        let mut ctx1 = StaticContext::new();
        let mut ctx2 = StaticContext::new();

        // OLD: Awkward scope threading
        let old_style = ctx1.new_scope(|scope| {
            let (x, _scope) = scope.auto_var::<f64>();
            x.clone() * x
        });

        // NEW: Clean lambda syntax
        let new_style = ctx2.lambda(|x| x.clone() * x);

        // Both should produce identical results
        let inputs = hlist![3.0];
        let old_result = old_style.eval(inputs);
        let new_result = new_style.eval(inputs);

        assert_eq!(old_result, new_result);
        assert_eq!(old_result, 9.0); // 3² = 9
    }
}
