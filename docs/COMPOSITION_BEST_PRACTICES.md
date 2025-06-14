# Composition Best Practices in DSLCompile

*How to build composable mathematical functions using lambda calculus and category theory principles*

## Table of Contents

1. [Overview](#overview)
2. [Current Problems & Anti-Patterns](#current-problems--anti-patterns)
3. [PL Research Insights](#pl-research-insights)
4. [DSLCompile's Hidden Strengths](#dslcompiles-hidden-strengths)
5. [Recommended Approaches](#recommended-approaches)
6. [Examples: Wrong vs Right](#examples-wrong-vs-right)
7. [Advanced Patterns](#advanced-patterns)
8. [Performance Considerations](#performance-considerations)
9. [Future Directions](#future-directions)

## Overview

DSLCompile has excellent foundational infrastructure for mathematical function composition based on lambda calculus and category theory. However, many examples in the codebase don't leverage these capabilities properly, leading to manual variable management and error-prone expression recreation.

**Key Insight**: DSLCompile already implements the right abstractions - we just need to use them correctly at the API level.

## Current Problems & Anti-Patterns

### ❌ Manual Context Recreation

```rust
// ANTI-PATTERN: Manual expression recreation
let mut composed_ctx = DynamicContext::<f64>::new();
let x_comp = composed_ctx.var(); // Variable(0) 
let y_comp = composed_ctx.var(); // Variable(1)

// Manually recreating expressions - ERROR PRONE!
let quad_in_comp = &x_comp * &x_comp + 2.0 * &x_comp + 1.0;
let exp_in_comp = y_comp.clone().exp() + 1.0;
let combined = &quad_in_comp + &exp_in_comp;
```

**Problems:**
- Manual variable index management
- Risk of variable collisions
- No true function abstraction
- Not composable or reusable
- Doesn't scale

### ❌ Expression Composition vs Function Composition

```rust
// ANTI-PATTERN: Composing expressions instead of functions
fn compose_expressions(
    expr1: DynamicExpr<f64>, 
    expr2: DynamicExpr<f64>
) -> DynamicExpr<f64> {
    // This is expression manipulation, not function composition
    &expr1 + &expr2
}
```

**Problems:**
- Not mathematically sound
- Variables may have different meanings in different contexts
- No proper lambda abstraction
- Missing higher-order function capabilities

## PL Research Insights

### Lambda Calculus Foundation

**Core Principle**: Functions should be first-class values with proper abstraction and substitution.

```
λx. x² + 2x + 1    -- Quadratic function
λy. eʸ + 1         -- Exponential function
(f ∘ g)(x) = f(g(x)) -- Function composition
```

### Category Theory Structure

Functions form a **category** where:
- **Objects**: Type signatures (contexts)
- **Morphisms**: Functions between types
- **Composition**: Associative operation with identity

**Monoid Properties**:
- **Identity**: `f ∘ id = id ∘ f = f`
- **Associativity**: `(f ∘ g) ∘ h = f ∘ (g ∘ h)`
- **Closure**: Composition of functions is a function

### Higher-Order Functions

**Combinators** enable reusable composition patterns:

```
ADD(f, g) = λx. f(x) + g(x)    -- Pointwise addition
MULT(f, g) = λx. f(x) * g(x)   -- Pointwise multiplication  
COMPOSE(f, g) = λx. f(g(x))    -- Function composition
```

## DSLCompile's Hidden Strengths

### ✅ Already Has Lambda Calculus

```rust
// DSLCompile already implements this!
pub enum Lambda<T> {
    Lambda { var_index: usize, body: Box<ASTRepr<T>> },
    Identity,
    Constant(Box<ASTRepr<T>>),
    Compose { f: Box<Lambda<T>>, g: Box<Lambda<T>> }, // 🔥 Function composition!
}
```

### ✅ Already Has Higher-Order Functions

```rust
// Collection mapping with lambda functions
Collection::Map {
    lambda: Box<Lambda<T>>,
    collection: Box<Collection<T>>,
}
```

### ✅ Already Has Category Theory

```rust
// Function composition with proper evaluation
Lambda::Compose { f, g } => {
    let g_result = self.eval_lambda(g, value, variables);
    self.eval_lambda(f, g_result, variables)
}
```

### ✅ Zero-Cost Abstractions

The lambda infrastructure compiles to efficient native code with no runtime overhead.

## Recommended Approaches

### 1. Use Lambda Infrastructure Directly

```rust
// ✅ GOOD: Use existing Lambda composition
fn compose_functions(f: Lambda<f64>, g: Lambda<f64>) -> Lambda<f64> {
    Lambda::Compose {
        f: Box::new(f),
        g: Box::new(g),
    }
}

// Create lambda functions
let quadratic = Lambda::Lambda {
    var_index: 0,
    body: Box::new(
        ASTRepr::Add(
            Box::new(ASTRepr::Add(
                Box::new(ASTRepr::Mul(
                    Box::new(ASTRepr::Variable(0)),
                    Box::new(ASTRepr::Variable(0))
                )),
                Box::new(ASTRepr::Mul(
                    Box::new(ASTRepr::Constant(2.0)),
                    Box::new(ASTRepr::Variable(0))
                ))
            )),
            Box::new(ASTRepr::Constant(1.0))
        )
    ),
};

let exponential = Lambda::Lambda {
    var_index: 0,
    body: Box::new(
        ASTRepr::Add(
            Box::new(ASTRepr::Exp(Box::new(ASTRepr::Variable(0)))),
            Box::new(ASTRepr::Constant(1.0))
        )
    ),
};

// Compose using existing infrastructure
let composed = compose_functions(quadratic, exponential);
```

### 2. Build Functional API Layer

```rust
/// High-level functional interface
pub struct MathFunction<T> {
    pub name: String,
    pub lambda: Lambda<T>,
    pub arity: usize,
}

impl<T> MathFunction<T> {
    pub fn compose(&self, other: &Self) -> Self {
        MathFunction {
            name: format!("{}∘{}", self.name, other.name),
            lambda: Lambda::Compose {
                f: Box::new(self.lambda.clone()),
                g: Box::new(other.lambda.clone()),
            },
            arity: other.arity, // Arity of the innermost function
        }
    }
    
    pub fn add(&self, other: &Self) -> Self {
        // Create pointwise addition using Collection::Map
        // Implementation details...
    }
}
```

### 3. Smart Variable Management

```rust
/// Automatic De Bruijn index management
pub struct LambdaBuilder {
    next_var: usize,
}

impl LambdaBuilder {
    pub fn lambda<F>(&mut self, f: F) -> Lambda<f64>
    where
        F: FnOnce(ASTRepr<f64>) -> ASTRepr<f64>,
    {
        let var_index = self.next_var;
        self.next_var += 1;
        
        let var = ASTRepr::Variable(var_index);
        let body = f(var);
        
        Lambda::Lambda {
            var_index,
            body: Box::new(body),
        }
    }
}
```

## Examples: Wrong vs Right

### Variable Management

#### ❌ Wrong: Manual Variable Juggling

```rust
// Manual context management - error prone
let mut ctx_a = DynamicContext::<f64>::new();
let x_a = ctx_a.var(); // Variable(0) in ctx_a

let mut ctx_b = DynamicContext::<f64>::new(); 
let y_b = ctx_b.var(); // Variable(0) in ctx_b - COLLISION!

// Manual recreation in shared context
let mut composed_ctx = DynamicContext::<f64>::new();
let x_comp = composed_ctx.var(); // Variable(0)
let y_comp = composed_ctx.var(); // Variable(1)
// Manually recreate expressions...
```

#### ✅ Right: Lambda Abstraction

```rust
// Proper lambda functions with automatic variable management
let quadratic_fn = MathFunction::from_lambda("quadratic", |builder| {
    builder.lambda(|x| x * x + 2.0 * x + 1.0)
});

let exponential_fn = MathFunction::from_lambda("exponential", |builder| {
    builder.lambda(|x| x.exp() + 1.0)
});

// Clean composition using category theory
let composed_fn = quadratic_fn.compose(&exponential_fn);
```

### Function Composition

#### ❌ Wrong: Expression Manipulation

```rust
// This is not function composition!
fn bad_compose(expr1: &DynamicExpr<f64>, expr2: &DynamicExpr<f64>) -> DynamicExpr<f64> {
    expr1 + expr2 // Just adding expressions
}
```

#### ✅ Right: Mathematical Function Composition

```rust
// True function composition: (f ∘ g)(x) = f(g(x))
fn compose(f: &MathFunction<f64>, g: &MathFunction<f64>) -> MathFunction<f64> {
    f.compose(g) // Uses Lambda::Compose internally
}
```

### Higher-Order Functions

#### ❌ Wrong: Hardcoded Combinations

```rust
// Hardcoded pattern - not reusable
fn add_quadratic_and_exp(x: &DynamicExpr<f64>) -> DynamicExpr<f64> {
    (x * x + 2.0 * x + 1.0) + (x.exp() + 1.0)
}
```

#### ✅ Right: Combinator Pattern

```rust
// Reusable combinator
fn pointwise_add<T>(f: Lambda<T>, g: Lambda<T>) -> Lambda<T> {
    // Use Collection::Map to create λx. f(x) + g(x)
    // Implementation leverages existing DSLCompile infrastructure
}

// Usage
let combined = pointwise_add(quadratic_lambda, exponential_lambda);
```

## Advanced Patterns

### Monadic Composition

```rust
// For functions that might fail or have effects
pub enum SafeMathFunction<T> {
    Pure(MathFunction<T>),
    Partial { function: MathFunction<T>, domain: Interval<T> },
    Fallible { function: MathFunction<T>, error_handler: Lambda<T> },
}
```

### Optimization-Aware Composition

```rust
impl MathFunction<f64> {
    pub fn optimize_and_compose(&self, other: &Self) -> Result<Self> {
        let composed = self.compose(other);
        
        // Use existing SymbolicOptimizer
        let mut optimizer = SymbolicOptimizer::new()?;
        let optimized_ast = optimizer.optimize(&composed.to_ast())?;
        
        Ok(MathFunction::from_ast("optimized_composition", optimized_ast))
    }
}
```

### Type-Level Function Signatures

```rust
// Encode function arity in types for compile-time safety
pub struct Fn1<T>(Lambda<T>);
pub struct Fn2<T>(Lambda<T>);

impl<T> Fn1<T> {
    pub fn compose<U>(self, other: Fn1<U>) -> Fn1<T> 
    where U: Into<T> {
        Fn1(Lambda::Compose {
            f: Box::new(self.0),
            g: Box::new(other.0),
        })
    }
}
```

## Performance Considerations

### Zero-Cost Abstractions

DSLCompile's lambda infrastructure compiles to efficient native code:

```rust
// High-level functional composition
let f = quadratic_fn.compose(&exponential_fn);

// Compiles to optimized Rust code like:
// fn compiled_composition(x: f64) -> f64 {
//     let temp = x.exp() + 1.0;
//     temp * temp + 2.0 * temp + 1.0
// }
```

### Optimization Opportunities

1. **Lambda Lifting**: Convert local functions to global functions
2. **Beta Reduction**: Substitute lambda arguments directly  
3. **Eta Conversion**: Eliminate redundant lambda abstractions
4. **Common Subexpression Elimination**: Works across composition boundaries

### Benchmarking

```rust
// Proper performance testing
#[bench]
fn bench_composition_approaches(b: &mut Bencher) {
    let f = create_complex_function();
    let g = create_another_function();
    
    // Test different composition strategies
    b.iter(|| {
        let composed = f.compose(&g);
        let compiled = compiled.call(test_input).unwrap();
        black_box(compiled)
    });
}
```

## Future Directions

### 1. Enhanced Lambda API

```rust
// Planned: Builder pattern for lambda construction
let f = lambda!(|x| x.powi(2) + 2.0 * x + 1.0);
let g = lambda!(|y| y.exp() + 1.0);
let composed = f.compose(g);
```

### 2. Category Theory Extensions

```rust
// Planned: Full category theory support
pub trait Category {
    type Object;
    type Morphism;
    
    fn compose(&self, f: Self::Morphism, g: Self::Morphism) -> Self::Morphism;
    fn identity(&self, obj: Self::Object) -> Self::Morphism;
}
```

### 3. Dependent Types for Mathematical Domains

```rust
// Future: Encode mathematical properties in types
pub struct PositiveReal;
pub struct UnitInterval;

pub fn sqrt<T: NonNegative>(x: T) -> PositiveReal {
    // Type system ensures mathematical validity
}
```

### 4. Integration with External Libraries

```rust
// Future: Interop with scientific computing libraries
impl From<MathFunction<f64>> for nalgebra::DVector<f64> {
    // Convert to numerical optimization format
}

impl From<MathFunction<f64>> for candle::Tensor {
    // Convert to machine learning format
}
```

## Best Practices Summary

### ✅ DO

1. **Use existing Lambda infrastructure** instead of manual expression recreation
2. **Build functions, not expressions** - use lambda abstraction
3. **Leverage Lambda::Compose** for mathematical function composition
4. **Use Collection::Map** for higher-order operations
5. **Let the optimizer work across composition boundaries**
6. **Think in terms of category theory** - functions as morphisms
7. **Use combinators** for reusable composition patterns

### ❌ DON'T

1. **Manual variable index management** - use lambda abstraction instead
2. **Expression concatenation** instead of function composition
3. **Hardcoded composition patterns** - build reusable combinators
4. **Ignore existing infrastructure** - DSLCompile already has the right foundations
5. **Premature optimization** - compose first, optimize second
6. **Type erasure** - preserve type information through composition

## Conclusion

DSLCompile has excellent foundational infrastructure for mathematical function composition. The key is to use the existing `Lambda`, `Collection::Map`, and composition capabilities properly instead of working with raw expressions.

By following these patterns, you get:
- **Mathematical rigor** through lambda calculus
- **Clean composition** through category theory  
- **Performance** through zero-cost abstractions
- **Reusability** through higher-order functions
- **Optimization** across composition boundaries

The path forward is to build a clean functional API layer on top of the existing solid foundations, making the power of DSLCompile's composition capabilities easily accessible to users. 