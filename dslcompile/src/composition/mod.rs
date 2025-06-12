// Core function composition infrastructure leveraging existing Lambda system
// Provides ergonomic APIs for mathematical function composition

use crate::ast::{ASTRepr, Scalar};
use crate::ast::ast_repr::Lambda;
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Div, Neg};
use frunk::{HCons, HNil};

/// Trait for converting tuple types to HList structures for multi-argument lambdas
/// 
/// This enables ergonomic syntax like:
/// - `MultiVar<(f64, f64)>` → `HCons<LambdaVar<f64>, HCons<LambdaVar<f64>, HNil>>`
/// - `MultiVar<(f64, i32, f32)>` → `HCons<LambdaVar<f64>, HCons<LambdaVar<i32>, HCons<LambdaVar<f32>, HNil>>>`
pub trait MultiVar<T> {
    type HList;
}

/// Implementation for two arguments: (A, B)
impl<A, B> MultiVar<(A, B)> for () {
    type HList = HCons<LambdaVar<A>, HCons<LambdaVar<B>, HNil>>;
}

/// Implementation for three arguments: (A, B, C)
impl<A, B, C> MultiVar<(A, B, C)> for () {
    type HList = HCons<LambdaVar<A>, HCons<LambdaVar<B>, HCons<LambdaVar<C>, HNil>>>;
}

/// Implementation for four arguments: (A, B, C, D)
impl<A, B, C, D> MultiVar<(A, B, C, D)> for () {
    type HList = HCons<LambdaVar<A>, HCons<LambdaVar<B>, HCons<LambdaVar<C>, HCons<LambdaVar<D>, HNil>>>>;
}

/// Implementation for five arguments: (A, B, C, D, E)
impl<A, B, C, D, E> MultiVar<(A, B, C, D, E)> for () {
    type HList = HCons<LambdaVar<A>, HCons<LambdaVar<B>, HCons<LambdaVar<C>, HCons<LambdaVar<D>, HCons<LambdaVar<E>, HNil>>>>>;
}

/// Trait for HList types that can be converted to lambda variables
/// This enables the unified `lambda_multi` interface that scales naturally
/// with any number of arguments using HList structure.
pub trait HListVars<T> 
where 
    T: Scalar + Copy,
{
    /// Create an HList of LambdaVar from a FunctionBuilder, returning both
    /// the variables and their indices for Lambda::MultiArg construction
    fn create_vars(builder: &mut FunctionBuilder<T>) -> (Self, Vec<usize>) 
    where 
        Self: Sized;
}

// Base case: Empty HList (no variables)
impl<T> HListVars<T> for HNil 
where 
    T: Scalar + Copy,
{
    fn create_vars(_builder: &mut FunctionBuilder<T>) -> (Self, Vec<usize>) {
        (HNil, vec![])
    }
}

// Recursive case: HList with at least one LambdaVar
impl<T, Tail> HListVars<T> for HCons<LambdaVar<T>, Tail>
where
    T: Scalar + Copy,
    Tail: HListVars<T>,
{
    fn create_vars(builder: &mut FunctionBuilder<T>) -> (Self, Vec<usize>) {
        let var_index = builder.next_var;
        builder.next_var += 1;
        
        let var = LambdaVar::new(ASTRepr::Variable(var_index));
        let (tail_vars, mut tail_indices) = Tail::create_vars(builder);
        
        let mut indices = vec![var_index];
        indices.append(&mut tail_indices);
        
        (HCons { head: var, tail: tail_vars }, indices)
    }
}

/// Wrapper for lambda variables that provides natural mathematical syntax
#[derive(Debug, Clone)]
pub struct LambdaVar<T> {
    ast: ASTRepr<T>,
}

impl<T> LambdaVar<T> 
where 
    T: Scalar + Copy,
{
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
impl<T> Add for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn add(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::Add(
            Box::new(self.ast),
            Box::new(rhs.ast),
        ))
    }
}

impl<T> Add<T> for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn add(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::Add(
            Box::new(self.ast),
            Box::new(ASTRepr::Constant(rhs)),
        ))
    }
}

// Note: Can't implement Add<LambdaVar<T>> for T due to orphan rules
// Users should write: x + scalar instead of scalar + x

impl<T> Mul for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::Mul(
            Box::new(self.ast),
            Box::new(rhs.ast),
        ))
    }
}

impl<T> Mul<T> for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn mul(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::Mul(
            Box::new(self.ast),
            Box::new(ASTRepr::Constant(rhs)),
        ))
    }
}

// Note: Can't implement Mul<LambdaVar<T>> for T due to orphan rules  
// Users should write: x * scalar instead of scalar * x

impl<T> Sub for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::Sub(
            Box::new(self.ast),
            Box::new(rhs.ast),
        ))
    }
}

impl<T> Sub<T> for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn sub(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::Sub(
            Box::new(self.ast),
            Box::new(ASTRepr::Constant(rhs)),
        ))
    }
}

impl<T> Div for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn div(self, rhs: Self) -> Self::Output {
        LambdaVar::new(ASTRepr::Div(
            Box::new(self.ast),
            Box::new(rhs.ast),
        ))
    }
}

impl<T> Div<T> for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn div(self, rhs: T) -> Self::Output {
        LambdaVar::new(ASTRepr::Div(
            Box::new(self.ast),
            Box::new(ASTRepr::Constant(rhs)),
        ))
    }
}

impl<T> Neg for LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn neg(self) -> Self::Output {
        LambdaVar::new(ASTRepr::Neg(Box::new(self.ast)))
    }
}

// Add reference-based operators to avoid move issues
impl<T> Add<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn add(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::Add(
            Box::new(self.ast.clone()),
            Box::new(rhs.ast.clone()),
        ))
    }
}

impl<T> Mul<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn mul(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::Mul(
            Box::new(self.ast.clone()),
            Box::new(rhs.ast.clone()),
        ))
    }
}

impl<T> Sub<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + Copy,
{
    type Output = LambdaVar<T>;

    fn sub(self, rhs: &LambdaVar<T>) -> Self::Output {
        LambdaVar::new(ASTRepr::Sub(
            Box::new(self.ast.clone()),
            Box::new(rhs.ast.clone()),
        ))
    }
}

impl<T> Div<&LambdaVar<T>> for &LambdaVar<T>
where
    T: Scalar + Copy,
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
impl<T> LambdaVar<T>
where
    T: Scalar + Copy + num_traits::Float,
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
pub struct FunctionBuilder<T> {
    next_var: usize,
    _phantom: PhantomData<T>,
}

impl<T> FunctionBuilder<T> 
where 
    T: Scalar + Copy,
{
    /// Create a new function builder
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
    /// let mut builder = FunctionBuilder::<f64>::new();
    /// let square = builder.lambda(|x| x * x + 1.0);
    /// ```
    pub fn lambda<F>(&mut self, f: F) -> Lambda<T>
    where
        F: FnOnce(LambdaVar<T>) -> LambdaVar<T>,
    {
        let var_index = self.next_var;
        self.next_var += 1;
        
        let var = LambdaVar::new(ASTRepr::Variable(var_index));
        let result = f(var);
        
        Lambda::Lambda {
            var_index,
            body: Box::new(result.into_ast()),
        }
    }
    
    /// Create a multi-argument lambda function using HList of variables
    /// 
    /// This is the unified approach for multi-argument functions that scales naturally
    /// and is consistent with the HList evaluation pattern.
    /// 
    /// # Examples
    /// ```rust
    /// use frunk::hlist;
    /// 
    /// let mut builder = FunctionBuilder::<f64>::new();
    /// 
    /// // Two arguments: f(x, y) = x * 2.0 + y * 3.0
    /// let two_arg = builder.lambda_multi(|vars| {
    ///     let (x, (y, _)) = vars.pluck(); // Extract x and y from HList
    ///     x * 2.0 + y * 3.0
    /// });
    /// 
    /// // Three arguments: f(x, y, z) = x * 2.0 + y * 3.0 + z * 4.0  
    /// let three_arg = builder.lambda_multi(|vars| {
    ///     let (x, rest) = vars.pluck();
    ///     let (y, rest) = rest.pluck(); 
    ///     let (z, _) = rest.pluck();
    ///     x * 2.0 + y * 3.0 + z * 4.0
    /// });
    /// ```
    pub fn lambda_multi<H, F>(&mut self, f: F) -> Lambda<T>
    where
        H: HListVars<T>,
        F: FnOnce(H) -> LambdaVar<T>,
    {
        let (vars_hlist, var_indices) = H::create_vars(self);
        let result = f(vars_hlist);
        
        Lambda::MultiArg {
            var_indices,
            body: Box::new(result.into_ast()),
        }
    }

    /// Get the next variable index without consuming it
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
pub struct MathFunction<T> {
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
/// let f = square_plus_one.as_callable();
/// let g = linear.as_callable(); 
/// let composed = MathFunction::from_lambda("composed", |builder| {
///     builder.lambda(|x| f(g(x)))  // Natural function call syntax!
/// });
/// ```
#[derive(Debug, Clone)]
pub struct CallableFunction<T> {
    function: MathFunction<T>,
}

impl<T> CallableFunction<T>
where
    T: Scalar + Copy,
{
    /// Create a callable function from a MathFunction
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
        match &self.function.lambda {
            Lambda::Identity => LambdaVar::new(input_ast),
            Lambda::Constant(expr) => LambdaVar::new(expr.as_ref().clone()),
            Lambda::Lambda { var_index, body } => {
                // Substitute the input for the lambda variable in the body
                let substituted = self.substitute_variable(body, *var_index, &input_ast);
                LambdaVar::new(substituted)
            }
            Lambda::MultiArg { var_indices, body } => {
                // For multi-arg functions in function call syntax, we can only substitute the first variable
                // This is a limitation of the f.call(x) syntax - true multi-arg calls need HList evaluation
                if let Some(&first_var) = var_indices.first() {
                    let substituted = self.substitute_variable(body, first_var, &input_ast);
                    LambdaVar::new(substituted)
                } else {
                    // No variables to substitute
                    LambdaVar::new(body.as_ref().clone())
                }
            }
            Lambda::Compose { f, g } => {
                // For composition, apply g first, then f
                let temp_func_g = MathFunction {
                    name: "temp_g".to_string(),
                    lambda: g.as_ref().clone(),
                    arity: 1,
                };
                let temp_func_f = MathFunction {
                    name: "temp_f".to_string(), 
                    lambda: f.as_ref().clone(),
                    arity: 1,
                };
                
                let g_result = CallableFunction::new(temp_func_g).apply_function_to_ast(input_ast);
                CallableFunction::new(temp_func_f).apply_function_to_ast(g_result.into_ast())
            }
        }
    }
    
    /// Substitute a variable with an expression in an AST
    fn substitute_variable(&self, ast: &ASTRepr<T>, var_index: usize, replacement: &ASTRepr<T>) -> ASTRepr<T> {
        match ast {
            ASTRepr::Variable(idx) if *idx == var_index => replacement.clone(),
            ASTRepr::Variable(idx) => ASTRepr::Variable(*idx),
            ASTRepr::Constant(val) => ASTRepr::Constant(*val),
            ASTRepr::Add(left, right) => ASTRepr::Add(
                Box::new(self.substitute_variable(left, var_index, replacement)),
                Box::new(self.substitute_variable(right, var_index, replacement)),
            ),
            ASTRepr::Sub(left, right) => ASTRepr::Sub(
                Box::new(self.substitute_variable(left, var_index, replacement)),
                Box::new(self.substitute_variable(right, var_index, replacement)),
            ),
            ASTRepr::Mul(left, right) => ASTRepr::Mul(
                Box::new(self.substitute_variable(left, var_index, replacement)),
                Box::new(self.substitute_variable(right, var_index, replacement)),
            ),
            ASTRepr::Div(left, right) => ASTRepr::Div(
                Box::new(self.substitute_variable(left, var_index, replacement)),
                Box::new(self.substitute_variable(right, var_index, replacement)),
            ),
            ASTRepr::Pow(base, exp) => ASTRepr::Pow(
                Box::new(self.substitute_variable(base, var_index, replacement)),
                Box::new(self.substitute_variable(exp, var_index, replacement)),
            ),
            ASTRepr::Neg(inner) => ASTRepr::Neg(
                Box::new(self.substitute_variable(inner, var_index, replacement))
            ),
            ASTRepr::Sin(inner) => ASTRepr::Sin(
                Box::new(self.substitute_variable(inner, var_index, replacement))
            ),
            ASTRepr::Cos(inner) => ASTRepr::Cos(
                Box::new(self.substitute_variable(inner, var_index, replacement))
            ),
            ASTRepr::Exp(inner) => ASTRepr::Exp(
                Box::new(self.substitute_variable(inner, var_index, replacement))
            ),
            ASTRepr::Ln(inner) => ASTRepr::Ln(
                Box::new(self.substitute_variable(inner, var_index, replacement))
            ),
            ASTRepr::Sqrt(inner) => ASTRepr::Sqrt(
                Box::new(self.substitute_variable(inner, var_index, replacement))
            ),
            // For other AST variants, recursively substitute
            _ => ast.clone(), // Fallback for complex cases
        }
    }
}

// Note: Function call syntax f(x) requires unstable fn_traits feature
// For now, users can use f.call(x) syntax which is almost as natural

impl<T> MathFunction<T>
where
    T: Scalar + Copy,
{
    /// Create a function from a lambda using the builder pattern
    ///
    /// # Example
    /// ```rust
    /// let square = MathFunction::from_lambda("square", |builder| {
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
    
    /// Create a multi-argument function with unified HList support
    ///
    /// This replaces the need for separate from_lambda2, from_lambda3, etc.
    /// The arity is automatically determined from the Lambda::MultiArg structure.
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
        let arity = match &lambda {
            Lambda::Lambda { .. } => 1,
            Lambda::MultiArg { var_indices, .. } => var_indices.len(),
            Lambda::Identity => 1,
            Lambda::Constant(_) => 0,
            Lambda::Compose { .. } => 1, // Composition typically preserves input arity
        };
        
        Self {
            name: name.to_string(),
            lambda,
            arity,
        }
    }

    /// Create a function directly from a Lambda (for advanced usage)
    pub fn from_lambda_direct(name: &str, lambda: Lambda<T>, arity: usize) -> Self {
        Self {
            name: name.to_string(),
            lambda,
            arity,
        }
    }

    /// Function composition using existing Lambda::Compose infrastructure
    /// 
    /// Creates (self ∘ other)(x) = self(other(x))
    ///
    /// # Example
    /// ```rust
    /// let f = MathFunction::from_lambda("f", |b| b.lambda(|x| x + 1));
    /// let g = MathFunction::from_lambda("g", |b| b.lambda(|x| x * 2));
    /// let composed = f.compose(&g); // f(g(x)) = (x * 2) + 1
    /// ```
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            name: format!("({} ∘ {})", self.name, other.name),
            lambda: Lambda::Compose {
                f: Box::new(self.lambda.clone()),
                g: Box::new(other.lambda.clone()),
            },
            arity: other.arity, // Composition preserves inner function's arity
        }
    }

    /// Extract the underlying AST representation
    /// Useful for integration with existing systems
    pub fn to_ast(&self) -> ASTRepr<T> {
        match &self.lambda {
            Lambda::Lambda { body, .. } => body.as_ref().clone(),
            Lambda::MultiArg { body, .. } => body.as_ref().clone(),
            Lambda::Identity => ASTRepr::Variable(0),
            Lambda::Constant(expr) => expr.as_ref().clone(),
            Lambda::Compose { .. } => {
                // For composition, we'd need to build the full expression tree
                // This is complex and may be better handled by evaluation
                todo!("Complex composition AST extraction not yet implemented")
            }
        }
    }

    /// Get the underlying lambda (for direct access)
    pub fn lambda(&self) -> &Lambda<T> {
        &self.lambda
    }

    /// Convert to a callable function that supports f(x) syntax in lambda expressions
    /// 
    /// # Example
    /// ```rust
    /// let f = square_plus_one.as_callable();
    /// let g = linear.as_callable();
    /// let composed = MathFunction::from_lambda("natural_composition", |builder| {
    ///     builder.lambda(|x| f(g(x)))  // Natural mathematical syntax!
    /// });
    /// ```
    pub fn as_callable(&self) -> CallableFunction<T> {
        CallableFunction::new(self.clone())
    }
}

/// Convenience functions for common lambda patterns
impl<T> Lambda<T>
where
    T: Scalar + Copy,
{
    /// Create identity lambda: λx.x
    pub fn identity() -> Self {
        Lambda::Identity
    }

    /// Create constant lambda: λx.c
    pub fn constant(value: T) -> Self {
        Lambda::Constant(Box::new(ASTRepr::Constant(value)))
    }

    /// Compose two lambdas using existing infrastructure
    pub fn compose(f: Lambda<T>, g: Lambda<T>) -> Self {
        Lambda::Compose {
            f: Box::new(f),
            g: Box::new(g),
        }
    }
}

/// Integration helpers for working with existing DSLCompile systems
impl<T> MathFunction<T>
where
    T: Scalar + Copy + num_traits::Float + num_traits::FromPrimitive + num_traits::Zero,
{
    /// Evaluate the function with HList inputs (unified evaluation interface)
    /// 
    /// This is the single evaluation method for MathFunction, using HList for type-safe
    /// heterogeneous inputs that leverage DSLCompile's existing zero-cost abstractions.
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
        match &self.lambda {
            crate::ast::ast_repr::Lambda::Lambda { body, .. } => {
                inputs.eval_expr(body)
            }
            crate::ast::ast_repr::Lambda::MultiArg { body, .. } => {
                // Multi-argument lambdas use the same evaluation as single-argument
                // The variable indices are already properly encoded in the body AST
                inputs.eval_expr(body)
            }
            crate::ast::ast_repr::Lambda::Identity => {
                // For identity, we need at least one input
                inputs.get_var(0)
            }
            crate::ast::ast_repr::Lambda::Constant(expr) => {
                inputs.eval_expr(expr)
            }
            crate::ast::ast_repr::Lambda::Compose { f, g } => {
                // For composition, evaluate g first, then f
                let temp_g = MathFunction {
                    name: "temp_g".to_string(),
                    lambda: g.as_ref().clone(),
                    arity: 1,
                };
                let temp_f = MathFunction {
                    name: "temp_f".to_string(), 
                    lambda: f.as_ref().clone(),
                    arity: 1,
                };
                let g_result = temp_g.eval(inputs);
                // For f, we create a single-element HList with g's result
                // This properly handles composition in the HList paradigm
                temp_f.eval(frunk::hlist![g_result])
            }
        }
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
        
        match square_lambda {
            Lambda::Lambda { var_index, .. } => assert_eq!(var_index, 0),
            _ => panic!("Expected Lambda::Lambda"),
        }
    }

    #[test]
    fn test_math_function_creation() {
        // Create a function using the builder pattern
        let square = MathFunction::<f64>::from_lambda("square", |builder| {
            builder.lambda(|x| x.clone() * x)
        });
        
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
        // f(x) = x + 1
        let add_one = MathFunction::<f64>::from_lambda("add_one", |builder| {
            builder.lambda(|x| x + 1.0)
        });
        
        // g(x) = x * 2  
        let double = MathFunction::<f64>::from_lambda("double", |builder| {
            builder.lambda(|x| x * 2.0)
        });
        
        // Compose: f(g(x)) = (x * 2) + 1
        let composed = add_one.compose(&double);
        assert_eq!(composed.name, "(add_one ∘ double)");
        
        // Verify it uses Lambda::Compose internally
        match composed.lambda {
            Lambda::Compose { .. } => {},
            _ => panic!("Expected Lambda::Compose"),
        }
        
        // Test evaluation with HList
        let result = composed.eval(hlist![3.0]);
        assert_eq!(result, 7.0); // double(3) = 6, add_one(6) = 7
    }

    #[test]
    fn test_lambda_convenience_functions() {
        let identity = Lambda::<f64>::identity();
        match identity {
            Lambda::Identity => {},
            _ => panic!("Expected Identity lambda"),
        }
        
        let constant = Lambda::constant(42.0);
        match constant {
            Lambda::Constant(_) => {},
            _ => panic!("Expected Constant lambda"),
        }
    }

    #[test]
    fn test_direct_lambda_composition() {
        let f = Lambda::constant(5.0);
        let g = Lambda::identity();
        
        let composed = Lambda::compose(f, g);
        match composed {
            Lambda::Compose { .. } => {},
            _ => panic!("Expected Compose lambda"),
        }
    }
}