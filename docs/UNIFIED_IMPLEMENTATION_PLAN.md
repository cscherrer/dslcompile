# Unified Implementation Plan: Two-Interface Strategy

## Overview

Transform the current fragmented API into two clean interfaces:
1. **`StaticContext`** - Compile-time optimized, same syntax as Dynamic
2. **`DynamicContext`** - Runtime flexible, static with heterogeneous support

## Current Status Analysis

✅ **Already Working**:
- `DynamicContext` exists with good ergonomics
- `HeteroContext` provides zero-overhead heterogeneous types
- Basic operator overloading implemented
- Sum API unification in progress

❌ **Current Problems**:
- Three different APIs (`Context`, `DynamicContext`, `HeteroContext`)
- Complex scope builders for static context
- Users must choose "hetero" vs regular variants
- No unified input/evaluation interface

## Implementation Strategy

### Phase 1: Enhance DynamicContext with Heterogeneous Support

#### 1.1 Add Native Heterogeneous Variable Support

```rust
// File: src/ast/runtime/expression_builder.rs

impl DynamicContext {
    /// Static variable creation - supports all types transparently
    pub fn var<T: ExprType>(&self) -> TypedVar<T> {
        self.registry.borrow_mut().register_typed_variable::<T>()
    }
    
    /// Static expression creation
    pub fn expr_from<T: ExprType>(&self, var: TypedVar<T>) -> Expr<T> {
        Expr::new(ASTRepr::Variable(var.index()), self.registry.clone())
    }
    
    /// Array indexing support
    pub fn index<T: ExprType>(&self, array: Expr<Vec<T>>, index: Expr<usize>) -> Expr<T> {
        Expr::new(ASTRepr::ArrayIndex {
            array: Box::new(array.into_ast()),
            index: Box::new(index.into_ast()),
        }, self.registry.clone())
    }
}
```

#### 1.2 Create Unified Expression Type

```rust
// File: src/unified_expr.rs (new file)

/// Unified expression type that works with both contexts
#[derive(Debug, Clone)]
pub struct Expr<T> {
    ast: ASTRepr<T>,
    registry: Arc<RefCell<VariableRegistry>>,
    _phantom: PhantomData<T>,
}

impl<T: ExprType> Expr<T> {
    /// Array indexing operator
    pub fn index(self, index: Expr<usize>) -> Expr<T> 
    where T: Clone {
        // Implementation that works for Vec<T> -> T
    }
    
    /// Unified operators work regardless of context
    pub fn add<U>(self, other: Expr<U>) -> Expr<T::Output>
    where T: Add<U> {
        // Type-safe addition
    }
    
    // ... other operators
}
```

#### 1.3 Unified Input/Output System

```rust
// File: src/unified_inputs.rs (new file)

/// Unified input system for both contexts
#[derive(Debug, Clone)]
pub struct Inputs {
    scalars: HashMap<String, f64>,
    arrays: HashMap<String, Vec<f64>>,
    indices: HashMap<String, usize>,
    // ... other types as needed
}

impl Inputs {
    pub fn new() -> Self { /* ... */ }
    
    pub fn add<T: InputType>(mut self, name: &str, value: T) -> Self {
        value.insert_into(&mut self, name);
        self
    }
    
    pub fn with<T: InputType>(name: &str, value: T) -> Self {
        Self::new().add(name, value)
    }
}

trait InputType {
    fn insert_into(self, inputs: &mut Inputs, name: &str);
}

impl InputType for f64 {
    fn insert_into(self, inputs: &mut Inputs, name: &str) {
        inputs.scalars.insert(name.to_string(), self);
    }
}

impl InputType for Vec<f64> {
    fn insert_into(self, inputs: &mut Inputs, name: &str) {
        inputs.arrays.insert(name.to_string(), self);
    }
}

impl InputType for usize {
    fn insert_into(self, inputs: &mut Inputs, name: &str) {
        inputs.indices.insert(name.to_string(), self);
    }
}
```

### Phase 2: Create Simplified StaticContext

#### 2.1 Remove Complex Scope Builders

```rust
// File: src/compile_time/static_context.rs (new file)

/// Simplified static context with same API as DynamicContext
#[derive(Debug)]
pub struct StaticContext {
    next_var_id: usize,
    variable_types: HashMap<usize, TypeId>,
}

impl StaticContext {
    pub fn new() -> Self {
        Self {
            next_var_id: 0,
            variable_types: HashMap::new(),
        }
    }
    
    /// Same interface as DynamicContext
    pub fn var<T: ExprType + 'static>(&mut self) -> StaticVar<T> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        self.variable_types.insert(id, TypeId::of::<T>());
        StaticVar::new(id)
    }
    
    /// Same constant creation
    pub fn constant<T: ExprType>(&self, value: T) -> StaticExpr<T> {
        StaticExpr::Constant(value)
    }
    
    /// Same evaluation interface
    pub fn eval<T>(&self, expr: &StaticExpr<T>, inputs: &Inputs) -> T {
        // Compile-time specialized evaluation
        expr.eval_static(inputs)
    }
}
```

#### 2.2 Static Expression Types

```rust
// File: src/compile_time/static_expr.rs (new file)

/// Zero-overhead static expressions
#[derive(Debug, Clone)]
pub enum StaticExpr<T> {
    Variable(usize),
    Constant(T),
    Add(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    Mul(Box<StaticExpr<T>>, Box<StaticExpr<T>>),
    ArrayIndex {
        array: Box<StaticExpr<Vec<T>>>,
        index: Box<StaticExpr<usize>>,
    },
    // ... other operations
}

impl<T: ExprType> StaticExpr<T> {
    /// Zero-overhead evaluation
    pub fn eval_static(&self, inputs: &Inputs) -> T {
        match self {
            StaticExpr::Variable(id) => inputs.get_typed::<T>(*id),
            StaticExpr::Constant(val) => val.clone(),
            StaticExpr::Add(a, b) => a.eval_static(inputs) + b.eval_static(inputs),
            // ... monomorphized operations
        }
    }
    
    /// Same operators as DynamicContext
    pub fn add(self, other: StaticExpr<T>) -> StaticExpr<T> {
        StaticExpr::Add(Box::new(self), Box::new(other))
    }
}

/// Zero-overhead static variables
#[derive(Debug, Clone)]
pub struct StaticVar<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T: ExprType> StaticVar<T> {
    /// Array indexing (when T = Vec<U>)
    pub fn index<U>(&self, index: StaticExpr<usize>) -> StaticExpr<U> 
    where T: AsArray<Item = U> {
        StaticExpr::ArrayIndex {
            array: Box::new(StaticExpr::Variable(self.id)),
            index: Box::new(index),
        }
    }
}
```

### Phase 3: Unify APIs with Trait System

#### 3.1 Common Context Trait

```rust
// File: src/unified_context.rs (new file)

/// Unified trait for both contexts
pub trait UnifiedContext {
    type Var<T: ExprType>: Clone;
    type Expr<T: ExprType>: Clone;
    
    fn new() -> Self;
    fn var<T: ExprType>(&mut self) -> Self::Var<T>;
    fn constant<T: ExprType>(&self, value: T) -> Self::Expr<T>;
    fn eval<T: ExprType>(&self, expr: &Self::Expr<T>, inputs: &Inputs) -> T;
    
    // Common operations
    fn sum<R, F>(&self, range: R, f: F) -> crate::Result<Self::Expr<f64>>
    where
        R: IntoSummableRange,
        F: Fn(Self::Expr<f64>) -> Self::Expr<f64>;
}

impl UnifiedContext for DynamicContext {
    type Var<T: ExprType> = TypedVar<T>;
    type Expr<T: ExprType> = Expr<T>;
    
    // ... implementations delegate to existing methods
}

impl UnifiedContext for StaticContext {
    type Var<T: ExprType> = StaticVar<T>;
    type Expr<T: ExprType> = StaticExpr<T>;
    
    // ... implementations use compile-time specialization
}
```

#### 3.2 Generic Programming Interface

```rust
// File: src/lib.rs additions

/// Generic function that works with both contexts
pub fn build_expression<C: UnifiedContext>(ctx: &mut C) -> C::Expr<f64> {
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    x.add(y) // Same syntax for both!
}

/// Usage examples
fn usage_examples() {
    // Static context - compile-time optimized
    let mut static_ctx = StaticContext::new();
    let static_expr = build_expression(&mut static_ctx);
    let static_result = static_ctx.eval(&static_expr, &inputs);
    
    // Dynamic context - runtime flexible  
    let mut dynamic_ctx = DynamicContext::new();
    let dynamic_expr = build_expression(&mut dynamic_ctx);
    let dynamic_result = dynamic_ctx.eval(&dynamic_expr, &inputs);
    
    // Identical results!
    assert_eq!(static_result, dynamic_result);
}
```

### Phase 4: Migration Path

#### 4.1 Deprecation Strategy

```rust
// Phase out old APIs gradually
#[deprecated(note = "Use StaticContext instead. See migration guide.")]
pub type Context<T, const SCOPE: usize> = StaticContext;

#[deprecated(note = "Use StaticContext or DynamicContext. See migration guide.")]
pub type HeteroContext<const SCOPE: usize, const MAX_VARS: usize> = StaticContext;

// Provide conversion utilities
impl From<OldContext> for StaticContext {
    fn from(old: OldContext) -> Self {
        // Conversion logic
    }
}
```

#### 4.2 Migration Examples

```rust
// Before: Complex scope builders
let mut builder = Context::new_f64();
let expr = builder.new_scope(|scope| {
    let (x, scope) = scope.auto_var();
    let (y, _scope) = scope.auto_var();
    x + y
});

// After: Simple variable creation
let mut ctx = StaticContext::new();
let x = ctx.var::<f64>();
let y = ctx.var::<f64>();
let expr = x + y;
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Create `Inputs` unified input system
- [ ] Enhance `DynamicContext` with typed variables
- [ ] Add array indexing support to `DynamicContext`

### Week 2: Static Context
- [ ] Create simplified `StaticContext`
- [ ] Implement `StaticExpr` with monomorphization
- [ ] Add zero-overhead evaluation

### Week 3: Unification
- [ ] Create `UnifiedContext` trait
- [ ] Implement trait for both contexts
- [ ] Add generic programming support

### Week 4: Migration
- [ ] Update examples and documentation
- [ ] Add deprecation warnings
- [ ] Create migration utilities

## Success Metrics

1. **API Simplicity**: Both contexts use identical method names
2. **Type Safety**: Compile-time type checking for all operations
3. **Performance**: Static context maintains zero overhead
4. **Migration**: Clear path from old to new APIs
5. **User Experience**: Single mental model for both contexts

## Example: Final API

```rust
use dslcompile::prelude::*;

fn unified_example() {
    // Choice 1: Compile-time optimized
    let mut static_ctx = StaticContext::new();
    let x = static_ctx.var::<f64>();
    let data = static_ctx.var::<Vec<f64>>();
    let index = static_ctx.var::<usize>();
    let expr = x + data.index(index);
    
    // Choice 2: Runtime flexible (same syntax!)
    let mut dynamic_ctx = DynamicContext::new();
    let x = dynamic_ctx.var::<f64>();
    let data = dynamic_ctx.var::<Vec<f64>>();
    let index = dynamic_ctx.var::<usize>();
    let expr = x + data.index(index);
    
    // Same evaluation interface
    let inputs = Inputs::new()
        .add("x", 3.0)
        .add("data", vec![1.0, 2.0, 3.0])
        .add("index", 1);
        
    let result1 = static_ctx.eval(&expr, &inputs);  // Zero overhead
    let result2 = dynamic_ctx.eval(&expr, &inputs); // Runtime flexible
    
    assert_eq!(result1, result2); // Identical results!
}
```

This plan eliminates user choice paralysis while maintaining all current capabilities. Users choose based on performance needs (static vs dynamic) rather than implementation details (hetero vs non-hetero). 