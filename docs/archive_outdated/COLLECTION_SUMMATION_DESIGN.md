# Collection-Based Summation Design

## Overview

The collection-based summation approach represents a fundamental shift from range-based summation to Map-based collections, providing significantly more mathematical expressiveness and enabling powerful optimizations through egglog's bidirectional rewrite rules.

## Key Innovations

### 1. Map-Based Collections Replace Range-Based Iteration

**Before (Range-Based):**
```rust
// Limited to simple ranges
ctx.sum(1..=10, |i| i * 2.0)
```

**After (Collection-Based):**
```rust
// Rich collection operations
let range = Collection::Range { start: 1.0, end: 10.0 };
let filtered = Collection::Filter { collection, predicate };

ctx.sum_collection(union, lambda)
```

### 2. Lambda Calculus Integration

**Lambda Functions as First-Class Citizens:**
```rust
// Identity function
let identity = Lambda::Identity;

// Constant function  
let constant = Lambda::Constant(Box::new(ASTRepr::Constant(5.0)));

// Complex lambda
let complex = Lambda::Lambda {
    var: "x".to_string(),
    body: Box::new(ASTRepr::Mul(
        Box::new(ASTRepr::Constant(2.0)),
        Box::new(ASTRepr::Variable(0)),
    )),
};

// Function composition
let composed = Lambda::Compose { f: lambda1, g: lambda2 };
```

### 3. Bidirectional Mathematical Identities

The egglog rules enable powerful mathematical transformations:

#### Linearity (Always Valid)
```egglog
; Σ(f(x) + g(x)) ⟷ Σ(f(x)) + Σ(g(x))
(rewrite (Sum ?X (Lambda ?x (Add (App ?f (Var ?x)) (App ?g (Var ?x)))))
         (Add (Sum ?X (Lambda ?x (App ?f (Var ?x))))
              (Sum ?X (Lambda ?x (App ?g (Var ?x))))))
```

#### Inclusion-Exclusion Principle
```egglog
; Σ(f(x) for x in A ∪ B) ⟷ Σ(f(x) for x in A) + Σ(f(x) for x in B) - Σ(f(x) for x in A ∩ B)
(rewrite (Sum (Union ?X ?Y) ?f)
         (Sub (Add (Sum ?X ?f) (Sum ?Y ?f)) 
              (Sum (Intersection ?X ?Y) ?f)))
```

#### Constant Factor Extraction
```egglog
; Σ(k * f(x)) ⟷ k * Σ(f(x)) when k is constant w.r.t. x
(rule ((constant-wrt ?k ?x))
      (= (Sum ?X (Lambda ?x (Mul ?k (App ?f (Var ?x)))))
         (Mul ?k (Sum ?X (Lambda ?x (App ?f (Var ?x)))))))
```

### 4. Automatic Pattern Recognition

The system automatically recognizes and optimizes common mathematical patterns:

#### Arithmetic Series
```rust
// Σ(i for i=1 to n) → n(n+1)/2
let arithmetic = ctx.sum_collection(range, identity_lambda);
// Automatically optimized to closed form
```

#### Geometric Series
```rust
// Σ(r^i for i=0 to n) → (1-r^(n+1))/(1-r)
let geometric = ctx.sum_collection(range, power_lambda);
// Automatically optimized to closed form
```

#### Constant Series
```rust
// Σ(c for i=1 to n) → c * n
let constant = ctx.sum_collection(range, constant_lambda);
// Automatically optimized to multiplication
```

## Architecture

### Core Data Types

```rust
/// Collection types for summation operations
pub enum Collection {
    Empty,                                    // ∅
    Singleton(Box<ASTRepr<f64>>),            // {x}
    Range { start: .., end: .. },            // [a, b]
    DataArray(String),                       // Runtime data binding
    Filter { collection: .., predicate: .. }, // {x ∈ A | P(x)}
}

/// Lambda expressions for mapping functions
pub enum Lambda {
    Lambda { var: String, body: .. },       // λx.body
    Identity,                                // λx.x
    Constant(Box<ASTRepr<f64>>),            // λx.c
    Compose { f: .., g: .. },               // f ∘ g
}

/// Extended expressions with collection operations
pub enum CollectionExpr {
    Sum { collection: Collection, lambda: Lambda },  // Σ(λ(x) for x in C)
    Map { lambda: Lambda, collection: Collection },  // map(λ, C)
    Size(Collection),                                // |C|
    App { lambda: Lambda, arg: .. },                // λ(x)
    Math(ASTRepr<f64>),                             // Regular math expression
}
```

### Optimization Pipeline

```rust
pub struct CollectionSummationOptimizer {
    collection_cache: HashMap<String, Collection>,
    lambda_cache: HashMap<String, Lambda>,
}

impl CollectionSummationOptimizer {
    /// Convert range-based to collection-based
    pub fn convert_range_to_collection(&mut self, ...) -> Result<CollectionExpr>
    
    /// Apply collection-based optimizations
    pub fn optimize_collection_expr(&mut self, expr: &CollectionExpr) -> Result<CollectionExpr>
    
    /// Apply specific summation patterns
    fn apply_summation_patterns(&self, collection: &Collection, lambda: &Lambda) -> Result<CollectionExpr>
    
    /// Convert back to standard AST
    pub fn to_ast(&self, expr: &CollectionExpr) -> Result<ASTRepr<f64>>
}
```

## Benefits Over Range-Based Approach

### 1. Mathematical Expressiveness

**Range-Based Limitations:**
- Only supports simple integer ranges
- No set operations (union, intersection)
- Limited composition capabilities
- No filtering or conditional summation

**Collection-Based Advantages:**
- Rich set operations with mathematical semantics
- Lambda calculus for functional composition
- Filtering and conditional operations
- Extensible to new collection types

### 2. Optimization Power

**Range-Based Optimizations:**
- Basic pattern recognition (arithmetic series)
- Simple constant factor extraction
- Limited rewrite rules

**Collection-Based Optimizations:**
- Bidirectional mathematical identities
- Lambda calculus optimizations (beta reduction, composition)
- Set-theoretic optimizations (inclusion-exclusion)
- Automatic pattern recognition for complex expressions

### 3. Composability

**Range-Based Composition:**
```rust
// Limited composition - must manually combine
let sum1 = ctx.sum(1..=10, |i| i * 2.0);
let sum2 = ctx.sum(11..=20, |i| i * 2.0);
// Manual combination required
```

**Collection-Based Composition:**
```rust
// Natural composition through set operations
let range1 = Collection::Range { start: 1.0, end: 10.0 };
let range2 = Collection::Range { start: 11.0, end: 20.0 };
let result = ctx.sum_collection(union, lambda);
// Automatically optimized
```

### 4. Data Integration

**Range-Based Data Handling:**
- Separate APIs for mathematical vs data summation
- Limited data binding capabilities
- No symbolic data processing

**Collection-Based Data Handling:**
- Unified API for all summation types
- Rich data array collections with symbolic processing
- Runtime data binding with compile-time optimization

## Usage Examples

### Basic Collection Operations

```rust
use dslcompile::prelude::*;
use dslcompile::symbolic::collection_summation::*;

let ctx = DynamicContext::new();

// Create collections
let range = ctx.range_collection(ctx.constant(1.0), ctx.constant(10.0));
let data = ctx.data_collection("sensor_readings");

// Create lambda functions
let identity = ctx.identity_lambda();
let square = ctx.lambda(|x| x.clone() * x)?;
let linear = ctx.lambda(|x| ctx.constant(2.0) * x + ctx.constant(1.0))?;

// Sum operations
let arithmetic_sum = ctx.sum_collection(range.clone(), identity)?;
let sum_of_squares = ctx.sum_collection(range, square)?;
let data_processing = ctx.sum_collection(data, linear)?;
```

### Advanced Set Operations

```rust
// Union of ranges
let range_a = Collection::Range { 
    start: Box::new(ASTRepr::Constant(1.0)), 
    end: Box::new(ASTRepr::Constant(5.0)) 
};
let range_b = Collection::Range { 
    start: Box::new(ASTRepr::Constant(6.0)), 
    end: Box::new(ASTRepr::Constant(10.0)) 
};
let union = Collection::Union { 
    left: Box::new(range_a), 
    right: Box::new(range_b) 
};

let lambda = ctx.lambda(|x| x.clone() * x)?;
let result = ctx.sum_collection(union, lambda)?;
// Automatically applies inclusion-exclusion optimizations
```

### Function Composition

```rust
// Compose functions: (f ∘ g)(x) = f(g(x))
let g = ctx.lambda(|x| x + ctx.constant(1.0))?;  // g(x) = x + 1
let f = ctx.lambda(|x| x.clone() * x)?;          // f(x) = x²

let composed = Lambda::Compose { 
    f: Box::new(f), 
    g: Box::new(g) 
};

let range = ctx.range_collection(ctx.constant(1.0), ctx.constant(5.0));
let result = ctx.sum_collection(range, composed)?;
// Σ((x+1)² for x in [1,5]) with automatic optimization
```

## Integration with Existing APIs

The collection-based approach is designed to complement, not replace, the existing unified sum API:

### Automatic Enhancement

```rust
// Enhanced sum method automatically detects optimization opportunities
let result = ctx.sum_enhanced(1..=1000, |i| {
    i * ctx.constant(2.0) + i.pow(ctx.constant(2.0))
})?;
// Automatically converts to collection-based form when beneficial
```

### Backward Compatibility

```rust
// Existing sum API continues to work
let traditional = ctx.sum(1..=10, |i| i * ctx.constant(2.0))?;

// Collection-based API provides additional power
let advanced = ctx.sum_collection(
    Collection::Union { /* complex collection */ },
    Lambda::Compose { /* composed functions */ }
)?;
```

## Future Extensions

### 1. Additional Collection Types

```rust
// Planned extensions
enum Collection {
    // ... existing types ...
    Cartesian { left: .., right: .. },      // A × B
    PowerSet(Box<Collection>),               // P(A)
    Complement { universe: .., set: .. },    // A^c
    Sequence { generator: Lambda, length: .. }, // Generated sequences
}
```

### 2. Advanced Lambda Operations

```rust
// Planned lambda extensions
enum Lambda {
    // ... existing types ...
    Partial { lambda: .., args: .. },       // Partial application
    Curry { lambda: .. },                   // Currying
    Memoized { lambda: .. },                // Memoization
    Conditional { predicate: .., then_: .., else_: .. }, // Conditional functions
}
```

### 3. Parallel Collection Processing

```rust
// Planned parallel processing
impl CollectionSummationOptimizer {
    pub fn optimize_parallel(&mut self, expr: &CollectionExpr) -> Result<ParallelExpr>
    pub fn generate_parallel_code(&self, expr: &ParallelExpr) -> Result<String>
}
```

## Conclusion

The collection-based summation approach represents a significant advancement in mathematical expression optimization, providing:

1. **Enhanced Mathematical Expressiveness** through rich collection operations and lambda calculus
2. **Powerful Optimization Capabilities** via bidirectional egglog rewrite rules
3. **Natural Composability** through set-theoretic operations
4. **Unified Data Processing** with symbolic optimization
5. **Extensible Architecture** for future enhancements

This approach transforms DSLCompile from a range-based summation system into a comprehensive mathematical collection processing framework, enabling sophisticated optimizations that were previously impossible. 