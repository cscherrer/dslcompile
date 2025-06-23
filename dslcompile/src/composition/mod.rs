// Core function composition infrastructure leveraging existing Lambda system
// Provides ergonomic APIs for mathematical function composition

use crate::ast::{ASTRepr, ExpressionType, Scalar, ast_repr::Lambda};
use frunk::{HCons, HNil};
use num_traits::{One, Zero};
use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// Trait for converting tuple types to `HList` structures for multi-argument lambdas
///
/// This enables ergonomic syntax like:
/// - `MultiVar<(f64, f64)>` → `HCons<LambdaVar<f64>, HCons<LambdaVar<f64>, HNil>>`
/// - `MultiVar<(f64, i32, f32)>` → `HCons<LambdaVar<f64>, HCons<LambdaVar<i32>, HCons<LambdaVar<f32>, HNil>>>`
pub trait MultiVar<T> {
    type HList;
}

/// Implementation for two arguments: (A, B)
impl<A: Scalar + ExpressionType + PartialOrd, B: Scalar + ExpressionType + PartialOrd>
    MultiVar<(A, B)> for ()
{
    type HList = HCons<LambdaVar<A>, HCons<LambdaVar<B>, HNil>>;
}

/// Implementation for three arguments: (A, B, C)
impl<
    A: Scalar + ExpressionType + PartialOrd,
    B: Scalar + ExpressionType + PartialOrd,
    C: Scalar + ExpressionType + PartialOrd,
> MultiVar<(A, B, C)> for ()
{
    type HList = HCons<LambdaVar<A>, HCons<LambdaVar<B>, HCons<LambdaVar<C>, HNil>>>;
}

/// Implementation for four arguments: (A, B, C, D)
impl<
    A: Scalar + ExpressionType + PartialOrd,
    B: Scalar + ExpressionType + PartialOrd,
    C: Scalar + ExpressionType + PartialOrd,
    D: Scalar + ExpressionType + PartialOrd,
> MultiVar<(A, B, C, D)> for ()
{
    type HList =
        HCons<LambdaVar<A>, HCons<LambdaVar<B>, HCons<LambdaVar<C>, HCons<LambdaVar<D>, HNil>>>>;
}

/// Implementation for five arguments: (A, B, C, D, E)
impl<
    A: Scalar + ExpressionType + PartialOrd,
    B: Scalar + ExpressionType + PartialOrd,
    C: Scalar + ExpressionType + PartialOrd,
    D: Scalar + ExpressionType + PartialOrd,
    E: Scalar + ExpressionType + PartialOrd,
> MultiVar<(A, B, C, D, E)> for ()
{
    type HList = HCons<
        LambdaVar<A>,
        HCons<LambdaVar<B>, HCons<LambdaVar<C>, HCons<LambdaVar<D>, HCons<LambdaVar<E>, HNil>>>>,
    >;
}

/// Trait for `HList` types that can be converted to lambda variables
/// This enables the unified `lambda_multi` interface that scales naturally
/// with any number of arguments using `HList` structure.
pub trait HListVars<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    /// Create an `HList` of `LambdaVar` from a `FunctionBuilder`, returning both
    /// the variables and their indices for `Lambda::MultiArg` construction
    fn create_vars(builder: &mut FunctionBuilder<T>) -> (Self, Vec<usize>)
    where
        Self: Sized;
}

// Base case: Empty HList (no variables)
impl<T: Scalar + ExpressionType + PartialOrd + Copy> HListVars<T> for HNil {
    fn create_vars(_builder: &mut FunctionBuilder<T>) -> (Self, Vec<usize>) {
        (HNil, vec![])
    }
}

// Recursive case: HList with at least one LambdaVar
impl<T: Scalar + ExpressionType + PartialOrd, Tail> HListVars<T> for HCons<LambdaVar<T>, Tail>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
    Tail: HListVars<T>,
{
    fn create_vars(builder: &mut FunctionBuilder<T>) -> (Self, Vec<usize>) {
        let var_index = builder.next_var;
        builder.next_var += 1;

        let var = LambdaVar::new(ASTRepr::Variable(var_index));
        let (tail_vars, mut tail_indices) = Tail::create_vars(builder);

        let mut indices = vec![var_index];
        indices.append(&mut tail_indices);

        (
            HCons {
                head: var,
                tail: tail_vars,
            },
            indices,
        )
    }
}

/// Wrapper for lambda variables that provides natural mathematical syntax
#[derive(Debug, Clone)]
pub struct LambdaVar<T: Scalar + ExpressionType + PartialOrd> {
    ast: ASTRepr<T>,
}

impl<T: Scalar + ExpressionType + PartialOrd + Copy> LambdaVar<T> {
    /// Create a new lambda variable from an AST node
    pub fn new(ast: ASTRepr<T>) -> Self {
        Self { ast }
    }

    /// Convert back to AST representation
    pub fn into_ast(self) -> ASTRepr<T> {
        self.ast
    }

    /// Get a reference to the AST
    pub fn ast(&self) -> &ASTRepr<T> {
        &self.ast
    }
}

// Implement operator overloading for LambdaVar to provide natural syntax
impl<T: Scalar + ExpressionType + PartialOrd> Add for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn add(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::add_binary(self.ast, rhs.ast))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Add<T> for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy + Zero,
{
    type Output = LambdaVar<T>;

    fn add(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::add_from_array([self.ast, ASTRepr::Constant(rhs)]))
    }
}

// Note: Can't implement Add<LambdaVar<T>> for T due to orphan rules
// Users should write: x + scalar instead of scalar + x

impl<T: Scalar + ExpressionType + PartialOrd> Mul for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::mul_binary(self.ast, rhs.ast))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Mul<T> for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy + One,
{
    type Output = LambdaVar<T>;

    fn mul(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::mul_from_array([self.ast, ASTRepr::Constant(rhs)]))
    }
}

// Note: Can't implement Mul<LambdaVar<T>> for T due to orphan rules
// Users should write: x * scalar instead of scalar * x

impl<T: Scalar + ExpressionType + PartialOrd> Sub for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::Sub(Box::new(self.ast), Box::new(rhs.ast)))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Sub<T> for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn sub(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::Sub(
            Box::new(self.ast),
            Box::new(ASTRepr::Constant(rhs)),
        ))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Div for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn div(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::Div(Box::new(self.ast), Box::new(rhs.ast)))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Div<T> for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn div(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::Div(
            Box::new(self.ast),
            Box::new(ASTRepr::Constant(rhs)),
        ))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Neg for LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn neg(self) -> Self::Output {
        LambdaVar::new(ASTRepr::Neg(Box::new(self.ast)))
    }
}

// Add reference-based operators to avoid move issues
impl<T: Scalar + ExpressionType + PartialOrd> Add<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy + Zero,
{
    type Output = LambdaVar<T>;

    fn add(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::add_from_array([self.ast.clone(), rhs.ast.clone()]))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Mul<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy + One,
{
    type Output = LambdaVar<T>;

    fn mul(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::mul_from_array([self.ast.clone(), rhs.ast.clone()]))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Sub<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn sub(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::Sub(
            Box::new(self.ast.clone()),
            Box::new(rhs.ast.clone()),
        ))
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> Div<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    type Output = LambdaVar<T>;

    fn div(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::Div(
            Box::new(self.ast.clone()),
            Box::new(rhs.ast.clone()),
        ))
    }
}

// Add transcendental functions
impl<T: Scalar + ExpressionType + PartialOrd> LambdaVar<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy + num_traits::Float,
{
    /// Sine function
    pub fn sin(&self) -> Self {
        LambdaVar::new(ASTRepr::Sin(Box::new(self.ast.clone())))
    }

    /// Cosine function
    pub fn cos(&self) -> Self {
        LambdaVar::new(ASTRepr::Cos(Box::new(self.ast.clone())))
    }

    /// Exponential function
    pub fn exp(&self) -> Self {
        LambdaVar::new(ASTRepr::Exp(Box::new(self.ast.clone())))
    }

    /// Natural logarithm
    pub fn ln(&self) -> Self {
        LambdaVar::new(ASTRepr::Ln(Box::new(self.ast.clone())))
    }

    /// Square root
    pub fn sqrt(&self) -> Self {
        LambdaVar::new(ASTRepr::Sqrt(Box::new(self.ast.clone())))
    }

    /// Power function
    pub fn pow(&self, exp: &Self) -> Self {
        LambdaVar::new(ASTRepr::Pow(
            Box::new(self.ast.clone()),
            Box::new(exp.ast.clone()),
        ))
    }

    /// Power with constant exponent
    pub fn powf(&self, exp: T) -> Self {
        LambdaVar::new(ASTRepr::Pow(
            Box::new(self.ast.clone()),
            Box::new(ASTRepr::Constant(exp)),
        ))
    }
}

/// Builder for creating lambda expressions with automatic variable management
pub struct FunctionBuilder<T: Scalar + ExpressionType + PartialOrd> {
    next_var: usize,
    _phantom: PhantomData<T>,
}

impl<T: Scalar + ExpressionType + PartialOrd> Default for FunctionBuilder<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + ExpressionType + PartialOrd> FunctionBuilder<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    /// Create a new function builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_var: 0,
            _phantom: PhantomData,
        }
    }

    /// Create a single-argument lambda function with natural mathematical syntax
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::composition::FunctionBuilder;
    /// let mut builder = FunctionBuilder::<f64>::new();
    /// let square = builder.lambda(|x| x.clone() * x + 1.0);
    /// ```
    pub fn lambda<F>(&mut self, f: F) -> Lambda<T>
    where
        F: FnOnce(LambdaVar<T>) -> LambdaVar<T>,
    {
        let var_index = self.next_var;
        self.next_var += 1;

        let var = LambdaVar::new(ASTRepr::Variable(var_index));
        let result = f(var);

        Lambda::single(var_index, Box::new(result.into_ast()))
    }

    /// Create a multi-argument lambda function using `HList` of variables
    ///
    /// This is the unified approach for multi-argument functions that scales naturally
    /// and is consistent with the `HList` evaluation pattern.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::composition::FunctionBuilder;
    ///
    /// let mut builder = FunctionBuilder::<f64>::new();
    ///
    /// // Note: lambda_multi requires complex HList type annotations
    /// // For most use cases, prefer the single-argument lambda() method
    /// // or use MathFunction::from_lambda_multi for easier syntax
    /// ```
    pub fn lambda_multi<H, F>(&mut self, f: F) -> Lambda<T>
    where
        H: HListVars<T>,
        F: FnOnce(H) -> LambdaVar<T>,
    {
        let (vars_hlist, var_indices) = H::create_vars(self);
        let result = f(vars_hlist);

        Lambda::new(var_indices, Box::new(result.into_ast()))
    }

    /// Get the next variable index without consuming it
    #[must_use]
    pub fn peek_next_var(&self) -> usize {
        self.next_var
    }

    /// Reset the variable counter (useful for independent functions)
    pub fn reset(&mut self) {
        self.next_var = 0;
    }
}

/// High-level mathematical function that wraps Lambda infrastructure
#[derive(Debug, Clone)]
pub struct MathFunction<T: Scalar + ExpressionType + PartialOrd> {
    /// Human-readable function name
    pub name: String,
    /// Underlying lambda expression
    pub lambda: Lambda<T>,
    /// Number of arguments this function expects
    pub arity: usize,
}

/// Wrapper that enables function call syntax in lambda expressions
///
/// This allows writing natural mathematical expressions like:
/// ```rust
/// use dslcompile::composition::MathFunction;
///
/// // First create the functions
/// let square_plus_one = MathFunction::from_lambda("square_plus_one", |builder| {
///     builder.lambda(|x| x.clone() * x + 1.0)
/// });
/// let linear = MathFunction::from_lambda("linear", |builder| {
///     builder.lambda(|x| x * 2.0)
/// });
///
/// let f = square_plus_one.as_callable();
/// let g = linear.as_callable();
/// let composed = MathFunction::from_lambda("composed", |builder| {
///     builder.lambda(|x| f.call(g.call(x)))  // Natural function call syntax!
/// });
/// ```
#[derive(Debug, Clone)]
pub struct CallableFunction<T: Scalar + ExpressionType + PartialOrd> {
    function: MathFunction<T>,
}

impl<T: Scalar + ExpressionType + PartialOrd> CallableFunction<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    /// Create a callable function from a `MathFunction`
    #[must_use]
    pub fn new(function: MathFunction<T>) -> Self {
        Self { function }
    }

    /// Call the function with a lambda variable argument
    /// This creates the AST representation of applying the function to the input
    pub fn call(&self, input: LambdaVar<T>) -> LambdaVar<T> {
        // We need to create an AST that represents function application
        // This is challenging because we need to inline the function's lambda body
        // with the input expression substituted for the lambda variable

        // For now, let's create a special AST node that represents function application
        // This would need to be handled specially during evaluation
        self.apply_function_to_ast(input.into_ast())
    }

    /// Apply the function to an AST, creating a new AST that represents the result
    fn apply_function_to_ast(&self, input_ast: ASTRepr<T>) -> LambdaVar<T> {
        let lambda = &self.function.lambda;

        if lambda.var_indices.is_empty() {
            // Constant lambda - just return the body
            LambdaVar::new(lambda.body.as_ref().clone())
        } else {
            // Substitute the input for the first lambda variable
            let first_var = lambda.var_indices[0];
            let substituted = self.substitute_variable(&lambda.body, first_var, &input_ast);
            LambdaVar::new(substituted)
        }
    }

    /// Substitute a variable with an expression in an AST
    fn substitute_variable(
        &self,
        ast: &ASTRepr<T>,
        var_index: usize,
        replacement: &ASTRepr<T>,
    ) -> ASTRepr<T> {
        match ast {
            ASTRepr::Variable(idx) if *idx == var_index => replacement.clone(),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Constant(val) => ASTRepr::Constant(*val),
            ASTRepr::Add(terms) => {
                let substituted_terms: Vec<_> = terms
                    .elements()
                    .map(|term| self.substitute_variable(term, var_index, replacement))
                    .collect();
                ASTRepr::Add(crate::ast::multiset::MultiSet::from_iter(substituted_terms))
            }
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(self.substitute_variable(left, var_index, replacement)),
                Box::new(self.substitute_variable(right, var_index, replacement)),
            ),
            ASTRepr::Mul(factors) => {
                let substituted_factors: Vec<_> = factors
                    .elements()
                    .map(|factor| self.substitute_variable(factor, var_index, replacement))
                    .collect();
                ASTRepr::Mul(crate::ast::multiset::MultiSet::from_iter(
                    substituted_factors,
                ))
            }
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(self.substitute_variable(left, var_index, replacement)),
                Box::new(self.substitute_variable(right, var_index, replacement)),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(self.substitute_variable(base, var_index, replacement)),
                Box::new(self.substitute_variable(exp, var_index, replacement)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(Box::new(self.substitute_variable(
                inner,
                var_index,
                replacement,
            ))),
            ASTRepr::Sin(inner) => ASTRepr::Sin(Box::new(self.substitute_variable(
                inner,
                var_index,
                replacement,
            ))),
            ASTRepr::Cos(inner) => ASTRepr::Cos(Box::new(self.substitute_variable(
                inner,
                var_index,
                replacement,
            ))),
            ASTRepr::Exp(inner) => ASTRepr::Exp(Box::new(self.substitute_variable(
                inner,
                var_index,
                replacement,
            ))),
            ASTRepr::Ln(inner) => ASTRepr::Ln(Box::new(self.substitute_variable(
                inner,
                var_index,
                replacement,
            ))),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(Box::new(self.substitute_variable(
                inner,
                var_index,
                replacement,
            ))),
            // For other AST variants, recursively substitute
            _ => ast.clone(), // Fallback for complex cases
        }
    }
}

// Note: Function call syntax f(x) requires unstable fn_traits feature
// For now, users can use f.call(x) syntax which is almost as natural

impl<T: Scalar + ExpressionType + PartialOrd> MathFunction<T>
where
    T: Scalar + ExpressionType + PartialOrd + Copy,
{
    /// Create a function from a lambda using the builder pattern
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::composition::MathFunction;
    /// let square = MathFunction::<f64>::from_lambda("square", |builder| {
    ///     builder.lambda(|x| x.clone() * x)
    /// });
    /// ```
    pub fn from_lambda<F>(name: &str, builder_fn: F) -> Self
    where
        F: FnOnce(&mut FunctionBuilder<T>) -> Lambda<T>,
    {
        let mut builder = FunctionBuilder::new();
        let lambda = builder_fn(&mut builder);

        Self {
            name: name.to_string(),
            lambda,
            arity: 1, // For now, assume unary functions
        }
    }

    /// Create a multi-argument function with unified `HList` support
    ///
    /// This replaces the need for separate `from_lambda2`, `from_lambda3`, etc.
    /// The arity is automatically determined from the `Lambda::MultiArg` structure.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    /// use dslcompile::composition::MultiVar;
    ///
    /// // Single argument
    /// let f = MathFunction::from_lambda("square", |builder| {
    ///     builder.lambda(|x| x.clone() * x + 1.0)
    /// });
    /// let result = f.eval(hlist![3.0]); // 3² + 1 = 10
    ///
    /// // Multiple arguments using lambda_multi with MultiVar
    /// let g = MathFunction::from_lambda_multi("weighted_sum", |builder| {
    ///     builder.lambda_multi::<<() as MultiVar<(f64, f64)>>::HList, _>(|vars| {
    ///         vars.head * 2.0 + vars.tail.head * 3.0
    ///     })
    /// });
    /// let result = g.eval(hlist![2.0, 4.0]); // 2*2 + 4*3 = 16
    /// ```
    pub fn from_lambda_multi<F>(name: &str, builder_fn: F) -> Self
    where
        F: FnOnce(&mut FunctionBuilder<T>) -> Lambda<T>,
    {
        let mut builder = FunctionBuilder::new();
        let lambda = builder_fn(&mut builder);

        // Determine arity from the lambda structure
        let arity = lambda.arity();

        Self {
            name: name.to_string(),
            lambda,
            arity,
        }
    }

    /// Create a function directly from a Lambda (for advanced usage)
    #[must_use]
    pub fn from_lambda_direct(name: &str, lambda: Lambda<T>, arity: usize) -> Self {
        Self {
            name: name.to_string(),
            lambda,
            arity,
        }
    }

    /// Compose this function with another: (self ∘ other)(x) = self(other(x))
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::composition::MathFunction;
    /// let f = MathFunction::from_lambda("f", |b| b.lambda(|x| x + 1));
    /// let g = MathFunction::from_lambda("g", |b| b.lambda(|x| x * 2));
    /// let composed = f.compose(&g); // (f ∘ g)(x) = f(g(x)) = (x * 2) + 1
    /// ```
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        // Create a new function that represents the composition using natural syntax
        let f_callable = self.as_callable();
        let g_callable = other.as_callable();

        MathFunction::from_lambda(&format!("({} ∘ {})", self.name, other.name), |builder| {
            builder.lambda(move |x| f_callable.call(g_callable.call(x)))
        })
    }

    /// Extract the underlying AST representation
    /// Useful for integration with existing systems
    #[must_use]
    pub fn to_ast(&self) -> ASTRepr<T> {
        self.lambda.body.as_ref().clone()
    }

    /// Get the underlying lambda (for direct access)
    #[must_use]
    pub fn lambda(&self) -> &Lambda<T> {
        &self.lambda
    }

    /// Convert to a callable function for use in lambda expressions
    ///
    /// # Example
    /// ```rust
    /// use dslcompile::composition::MathFunction;
    ///
    /// // First create the functions
    /// let square_plus_one = MathFunction::from_lambda("square_plus_one", |builder| {
    ///     builder.lambda(|x| x.clone() * x + 1.0)
    /// });
    /// let linear = MathFunction::from_lambda("linear", |builder| {
    ///     builder.lambda(|x| x * 2.0)
    /// });
    ///
    /// let f = square_plus_one.as_callable();
    /// let g = linear.as_callable();
    /// let composed = MathFunction::from_lambda("natural_composition", |builder| {
    ///     builder.lambda(|x| f.call(g.call(x)))
    /// });
    /// ```
    #[must_use]
    pub fn as_callable(&self) -> CallableFunction<T> {
        CallableFunction::new(self.clone())
    }
}

/// Integration helpers for working with existing `DSLCompile` systems
impl<T: Scalar + ExpressionType + PartialOrd> MathFunction<T>
where
    T: Scalar
        + ExpressionType
        + PartialOrd
        + Copy
        + num_traits::Float
        + num_traits::FromPrimitive
        + num_traits::Zero,
{
    /// Evaluate the function with `HList` inputs (unified evaluation interface)
    ///
    /// This is the single evaluation method for `MathFunction`, using `HList` for type-safe
    /// heterogeneous inputs that leverage `DSLCompile`'s existing zero-cost abstractions.
    ///
    /// # Examples
    /// ```rust
    /// use dslcompile::prelude::*;
    /// use frunk::hlist;
    /// use dslcompile::composition::MultiVar;
    ///
    /// // Single argument
    /// let f = MathFunction::from_lambda("square", |builder| {
    ///     builder.lambda(|x| x.clone() * x + 1.0)
    /// });
    /// let result = f.eval(hlist![3.0]); // 3² + 1 = 10
    ///
    /// // Multiple arguments using lambda_multi with MultiVar
    /// let g = MathFunction::from_lambda_multi("weighted_sum", |builder| {
    ///     builder.lambda_multi::<<() as MultiVar<(f64, f64)>>::HList, _>(|vars| {
    ///         vars.head * 2.0 + vars.tail.head * 3.0
    ///     })
    /// });
    /// let result = g.eval(hlist![2.0, 4.0]); // 2*2 + 4*3 = 16
    /// ```
    pub fn eval<H>(&self, inputs: H) -> T
    where
        H: crate::contexts::dynamic::expression_builder::hlist_support::HListEval<T>,
    {
        // All lambda types now use the same evaluation - the body contains the expression
        // and variable indices are properly encoded in the AST
        inputs.eval_expr(&self.lambda.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frunk::hlist;

    #[test]
    fn test_function_builder() {
        let mut builder = FunctionBuilder::<f64>::new();

        // Create a simple square function: x²
        let square_lambda = builder.lambda(|x| x.clone() * x);

        assert_eq!(square_lambda.arity(), 1);
        assert_eq!(square_lambda.var_indices[0], 0);
    }

    #[test]
    fn test_math_function_creation() {
        // Create a function using the builder pattern
        let square =
            MathFunction::<f64>::from_lambda("square", |builder| builder.lambda(|x| x.clone() * x));

        assert_eq!(square.name, "square");
        assert_eq!(square.arity, 1);
    }

    #[test]
    fn test_function_evaluation_with_hlist() {
        // Test single-argument function evaluation
        let square = MathFunction::<f64>::from_lambda("square", |builder| {
            builder.lambda(|x| x.clone() * x + 1.0)
        });

        let result = square.eval(hlist![3.0]);
        assert_eq!(result, 10.0); // 3² + 1 = 10
    }

    #[test]
    fn test_two_argument_function_with_hlist() {
        // Test two-argument function evaluation using lambda_multi with MultiVar
        let add_weighted = MathFunction::<f64>::from_lambda_multi("add_weighted", |builder| {
            builder.lambda_multi::<<() as MultiVar<(f64, f64)>>::HList, _>(|vars| {
                vars.head * 2.0 + vars.tail.head * 3.0
            })
        });

        let result = add_weighted.eval(hlist![2.0, 4.0]);
        assert_eq!(result, 16.0); // 2*2 + 4*3 = 4 + 12 = 16
    }

    #[test]
    fn test_function_composition_with_hlist() {
        // Test function composition using the new natural syntax approach
        let square =
            MathFunction::<f64>::from_lambda("square", |builder| builder.lambda(|x| x.clone() * x));

        let add_one =
            MathFunction::<f64>::from_lambda("add_one", |builder| builder.lambda(|x| x + 1.0));

        // Test composition: add_one(square(x)) = x² + 1
        let composed = add_one.compose(&square);

        let result = composed.eval(hlist![3.0]);
        assert_eq!(result, 10.0); // 3² + 1 = 10
    }

    #[test]
    fn test_lambda_convenience_functions() {
        let identity = Lambda::<f64>::identity();
        assert!(identity.is_identity());
        assert_eq!(identity.arity(), 1);

        let constant = Lambda::constant(42.0);
        assert!(constant.is_constant());
        assert_eq!(constant.arity(), 0);
        match constant.body.as_ref() {
            ASTRepr::Constant(value) => assert_eq!(*value, 42.0),
            _ => panic!("Expected constant expression"),
        }
    }
}
