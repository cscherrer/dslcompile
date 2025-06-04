//! Heterogeneous Static Context
//!
//! This module demonstrates a static context that supports different types for each variable
//! while maintaining compile-time type safety and zero runtime overhead.

use std::marker::PhantomData;

// ============================================================================
// CORE TRAITS - More Flexible Than Current NumericType
// ============================================================================

/// Base trait for types that can be used in expressions
pub trait ExpressionType: Clone + std::fmt::Debug + Send + Sync + 'static {
    /// Human-readable name for this type (for debugging/codegen)
    fn type_name() -> &'static str;
}

/// Trait for types that support scalar mathematical operations
pub trait ScalarType:
    ExpressionType
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + Copy // Scalars should be copyable
{
}

/// Trait for types that can be indexed (vectors, arrays, etc.)
pub trait IndexableType<Idx>: ExpressionType {
    type Item: ExpressionType;

    /// Get an item at the given index
    fn index(&self, idx: Idx) -> Self::Item;
}

// ============================================================================
// IMPLEMENTATIONS FOR STANDARD TYPES
// ============================================================================

// Scalar numeric types
impl ExpressionType for f64 {
    fn type_name() -> &'static str {
        "f64"
    }
}
impl ScalarType for f64 {}

impl ExpressionType for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
}
impl ScalarType for f32 {}

impl ExpressionType for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
}
impl ScalarType for i32 {}

// Vector types
impl ExpressionType for Vec<f64> {
    fn type_name() -> &'static str {
        "Vec<f64>"
    }
}

impl IndexableType<usize> for Vec<f64> {
    type Item = f64;

    fn index(&self, idx: usize) -> f64 {
        self[idx]
    }
}

// Index types
impl ExpressionType for usize {
    fn type_name() -> &'static str {
        "usize"
    }
}

// ============================================================================
// HETEROGENEOUS AST REPRESENTATION
// ============================================================================

/// Generic AST that can represent operations on any `ExpressionType`
#[derive(Debug)]
pub enum HeteroASTRepr<T: ExpressionType> {
    /// Variable reference by index
    Variable(usize),

    /// Constant value
    Constant(T),

    /// Function call with heterogeneous arguments
    FunctionCall {
        name: String,
        args: Vec<Box<dyn HeteroASTNode>>, // Type-erased arguments
    },
}

// Manual Clone implementation that handles the trait object
impl<T: ExpressionType> Clone for HeteroASTRepr<T> {
    fn clone(&self) -> Self {
        match self {
            HeteroASTRepr::Variable(idx) => HeteroASTRepr::Variable(*idx),
            HeteroASTRepr::Constant(val) => HeteroASTRepr::Constant(val.clone()),
            HeteroASTRepr::FunctionCall { name, args } => HeteroASTRepr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|arg| arg.clone_box()).collect(),
            },
        }
    }
}

/// Type-erased AST node for storing in collections
pub trait HeteroASTNode: std::fmt::Debug {
    /// Get the type name of this node's result
    fn result_type_name(&self) -> &'static str;

    /// Clone this node
    fn clone_box(&self) -> Box<dyn HeteroASTNode>;
}

impl<T: ExpressionType> HeteroASTNode for HeteroASTRepr<T> {
    fn result_type_name(&self) -> &'static str {
        T::type_name()
    }

    fn clone_box(&self) -> Box<dyn HeteroASTNode> {
        Box::new(self.clone())
    }
}

// ============================================================================
// SPECIFIC OPERATION TYPES
// ============================================================================

/// Scalar addition (only for `ScalarType`)
#[derive(Debug, Clone)]
pub struct ScalarAdd<T: ScalarType> {
    pub left: HeteroASTRepr<T>,
    pub right: HeteroASTRepr<T>,
}

impl<T: ScalarType> HeteroASTNode for ScalarAdd<T> {
    fn result_type_name(&self) -> &'static str {
        T::type_name()
    }

    fn clone_box(&self) -> Box<dyn HeteroASTNode> {
        Box::new(self.clone())
    }
}

/// Array indexing operation
#[derive(Debug, Clone)]
pub struct ArrayIndex<Container, Item>
where
    Container: IndexableType<usize, Item = Item>,
    Item: ExpressionType,
{
    pub array: HeteroASTRepr<Container>,
    pub index: HeteroASTRepr<usize>,
    _phantom: PhantomData<Item>,
}

impl<Container, Item> HeteroASTNode for ArrayIndex<Container, Item>
where
    Container: IndexableType<usize, Item = Item>,
    Item: ExpressionType,
{
    fn result_type_name(&self) -> &'static str {
        Item::type_name()
    }

    fn clone_box(&self) -> Box<dyn HeteroASTNode> {
        Box::new(self.clone())
    }
}

// ============================================================================
// HETEROGENEOUS VARIABLES AND EXPRESSIONS
// ============================================================================

/// A variable with a specific type in a specific scope
#[derive(Debug, Clone)]
pub struct HeteroVar<T: ExpressionType, const ID: usize, const SCOPE: usize> {
    _phantom: PhantomData<T>,
}

impl<T: ExpressionType, const ID: usize, const SCOPE: usize> Default for HeteroVar<T, ID, SCOPE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ExpressionType, const ID: usize, const SCOPE: usize> HeteroVar<T, ID, SCOPE> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Convert to an AST representation
    #[must_use]
    pub fn to_ast(self) -> HeteroASTRepr<T> {
        HeteroASTRepr::Variable(ID)
    }
}

/// Expression trait for heterogeneous expressions
pub trait HeteroExpr<T: ExpressionType, const SCOPE: usize>: Clone {
    /// Convert to AST representation
    fn to_ast(self) -> HeteroASTRepr<T>;
}

// Implement for variables
impl<T: ExpressionType, const ID: usize, const SCOPE: usize> HeteroExpr<T, SCOPE>
    for HeteroVar<T, ID, SCOPE>
{
    fn to_ast(self) -> HeteroASTRepr<T> {
        HeteroASTRepr::Variable(ID)
    }
}

// ============================================================================
// OPERATIONS - TYPE-SAFE COMBINATIONS
// ============================================================================

/// Add two scalar expressions (only works for `ScalarType`)
pub fn scalar_add<T, L, R, const SCOPE: usize>(left: L, right: R) -> impl HeteroExpr<T, SCOPE>
where
    T: ScalarType,
    L: HeteroExpr<T, SCOPE>,
    R: HeteroExpr<T, SCOPE>,
{
    ScalarAddExpr {
        left: left.to_ast(),
        right: right.to_ast(),
    }
}

#[derive(Debug, Clone)]
struct ScalarAddExpr<T: ScalarType> {
    left: HeteroASTRepr<T>,
    right: HeteroASTRepr<T>,
}

impl<T: ScalarType, const SCOPE: usize> HeteroExpr<T, SCOPE> for ScalarAddExpr<T> {
    fn to_ast(self) -> HeteroASTRepr<T> {
        HeteroASTRepr::FunctionCall {
            name: "add".to_string(),
            args: vec![Box::new(self.left), Box::new(self.right)],
        }
    }
}

/// Index into an array/vector
pub fn array_index<Container, Item, A, I, const SCOPE: usize>(
    array: A,
    index: I,
) -> impl HeteroExpr<Item, SCOPE>
where
    Container: IndexableType<usize, Item = Item>,
    Item: ExpressionType,
    A: HeteroExpr<Container, SCOPE>,
    I: HeteroExpr<usize, SCOPE>,
{
    ArrayIndexExpr {
        array: array.to_ast(),
        index: index.to_ast(),
        _phantom: PhantomData,
    }
}

#[derive(Debug, Clone)]
struct ArrayIndexExpr<Container, Item>
where
    Container: IndexableType<usize, Item = Item>,
    Item: ExpressionType,
{
    array: HeteroASTRepr<Container>,
    index: HeteroASTRepr<usize>,
    _phantom: PhantomData<Item>,
}

impl<Container, Item, const SCOPE: usize> HeteroExpr<Item, SCOPE>
    for ArrayIndexExpr<Container, Item>
where
    Container: IndexableType<usize, Item = Item>,
    Item: ExpressionType,
{
    fn to_ast(self) -> HeteroASTRepr<Item> {
        HeteroASTRepr::FunctionCall {
            name: "index".to_string(),
            args: vec![Box::new(self.array), Box::new(self.index)],
        }
    }
}

// ============================================================================
// HETEROGENEOUS CONTEXT AND SCOPE BUILDER
// ============================================================================

/// Context that supports heterogeneous types (no single type parameter!)
#[derive(Debug, Clone)]
pub struct HeteroContext<const SCOPE: usize> {
    // No type parameter needed!
}

impl Default for HeteroContext<0> {
    fn default() -> Self {
        Self::new()
    }
}

impl HeteroContext<0> {
    /// Create a new heterogeneous context
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl<const SCOPE: usize> HeteroContext<SCOPE> {
    /// Create a new scope with heterogeneous variables
    pub fn new_scope<F, R>(&mut self, f: F) -> R
    where
        F: for<'a> FnOnce(HeteroScopeBuilder<SCOPE, 0>) -> R,
    {
        f(HeteroScopeBuilder::new())
    }

    /// Advance to the next scope
    #[must_use]
    pub fn next(self) -> HeteroContext<{ SCOPE + 1 }> {
        HeteroContext {}
    }
}

/// Scope builder that can create variables of different types
#[derive(Debug, Clone, Copy)] // Made Copy to avoid move issues
pub struct HeteroScopeBuilder<const SCOPE: usize, const NEXT_ID: usize> {
    _phantom: PhantomData<[(); SCOPE]>,
}

impl<const SCOPE: usize, const NEXT_ID: usize> Default for HeteroScopeBuilder<SCOPE, NEXT_ID> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SCOPE: usize, const NEXT_ID: usize> HeteroScopeBuilder<SCOPE, NEXT_ID> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Create a variable of any `ExpressionType`
    #[must_use]
    pub fn auto_var<T: ExpressionType>(
        self,
    ) -> (
        HeteroVar<T, NEXT_ID, SCOPE>,
        HeteroScopeBuilder<SCOPE, { NEXT_ID + 1 }>,
    ) {
        (HeteroVar::new(), HeteroScopeBuilder::new())
    }

    /// Create a constant value (doesn't consume self, since it's Copy now)
    pub fn constant<T: ExpressionType>(&self, value: T) -> HeteroConstant<T> {
        HeteroConstant { value }
    }
}

/// Constant value in the heterogeneous system
#[derive(Debug, Clone)]
pub struct HeteroConstant<T: ExpressionType> {
    value: T,
}

impl<T: ExpressionType, const SCOPE: usize> HeteroExpr<T, SCOPE> for HeteroConstant<T> {
    fn to_ast(self) -> HeteroASTRepr<T> {
        HeteroASTRepr::Constant(self.value)
    }
}

// ============================================================================
// DEMO USAGE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heterogeneous_static_context() {
        let mut ctx = HeteroContext::new();

        // Each variable can have a different type!
        let expr = ctx.new_scope(|scope| {
            let (params, scope): (HeteroVar<Vec<f64>, 0, 0>, _) = scope.auto_var();
            let (index, scope): (HeteroVar<usize, 1, 0>, _) = scope.auto_var();
            let (multiplier, _): (HeteroVar<f64, 2, 0>, _) = scope.auto_var();

            // Array indexing: params[index]
            let indexed_value = array_index(params, index);

            // Scalar multiplication: result * multiplier
            scalar_add(indexed_value, multiplier)
        });

        let ast = expr.to_ast();
        println!("Heterogeneous expression AST: {ast:?}");

        // The types are all known at compile time!
        // Variable 0: Vec<f64>
        // Variable 1: usize
        // Variable 2: f64
        // Result: f64
    }

    #[test]
    fn test_autodiff_integration_concept() {
        // This shows how autodiff types could integrate

        let mut ctx = HeteroContext::new();

        let _expr = ctx.new_scope(|scope| {
            // Could support autodiff types like this:
            let (x, scope): (HeteroVar<f64, 0, 0>, _) = scope.auto_var();
            let (params, scope): (HeteroVar<Vec<f64>, 1, 0>, _) = scope.auto_var();

            // Polynomial: params[0] * x^2 + params[1] * x + params[2]
            let param0 = array_index(params.clone(), scope.constant(0_usize));
            let param1 = array_index(params.clone(), scope.constant(1_usize));
            let param2 = array_index(params, scope.constant(2_usize));

            // For now, just add the parameters (would need multiplication)
            scalar_add(scalar_add(param0, param1), param2)
        });

        // With this design, you could easily swap f64 for autodiff types!
    }
}
