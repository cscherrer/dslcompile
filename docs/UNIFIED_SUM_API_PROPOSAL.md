# Unified Sum API Proposal

## Problem Statement

Currently, DSLCompile has **three different summation methods** that create confusion:

1. `sum(iterable, closure)` - Handles ranges and data vectors
2. `sum_data(closure)` - Creates symbolic data expressions  
3. `sum_range(range, closure)` - Mathematical range summation

This violates the principle of having **one obvious way to do it**.

## Proposed Solution: Single `sum` Method

### Unified Semantics

**Summation is evaluated as part of constant folding unless it involves unbound variables.**

```rust
// UNIFIED API: Always use sum()
ctx.sum(input, |var| expression)
```

### Input Type Determines Behavior

#### 1. Mathematical Ranges → Closed-Form Optimization
```rust
// Σ(i=1 to 10) 2*i = 2*Σ(i) = 2*55 = 110
let result = ctx.sum(1..=10, |i| ctx.constant(2.0) * i)?;
// → Constant(110.0) - evaluated at build time
```

#### 2. Data Arrays → Runtime Iteration
```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
// Σ(x in data) 3*x = 3*(1+2+3+4+5) = 45
let result = ctx.sum(data, |x| ctx.constant(3.0) * x)?;
// → Constant(45.0) - evaluated at build time (data is bound)
```

#### 3. Symbolic Data → Deferred Evaluation
```rust
// BETTER: Use a clear marker type or method
let param = ctx.var(); // Unbound variable

// Option A: Special marker for symbolic data
let sum_expr = ctx.sum(SymbolicData, |x| x * param.clone())?;

// Option B: Keep the current sum_data method but make it consistent
let sum_expr = ctx.sum_data(|x| x * param.clone())?;

// Option C: Use a clear builder pattern
let sum_expr = ctx.sum_symbolic_data(|x| x * param.clone())?;

// Evaluate later with actual data
let result = ctx.eval_with_data(&sum_expr, &[2.0], &[vec![1.0, 2.0, 3.0]]);
// → 2.0 * (1+2+3) = 12.0
```

## Implementation Strategy

### Phase 1: Extend `IntoSummableRange` Trait

```rust
pub trait IntoSummableRange {
    fn into_summable(self) -> SummableRange;
}

// Add symbolic data support
impl IntoSummableRange for DataVariable {
    fn into_summable(self) -> SummableRange {
        SummableRange::SymbolicData { var_id: self.id }
    }
}
```

### Phase 2: Update `sum` Method Logic

```rust
pub fn sum<I, F>(&self, iterable: I, f: F) -> Result<TypedBuilderExpr<f64>>
where
    I: IntoSummableRange,
    F: Fn(TypedBuilderExpr<f64>) -> TypedBuilderExpr<f64>,
{
    match iterable.into_summable() {
        // Mathematical: Closed-form when possible
        SummableRange::MathematicalRange { start, end } => {
            let expr = f(self.var());
            if self.has_unbound_variables(&expr) {
                // Symbolic - defer evaluation
                Ok(self.create_symbolic_sum(start, end, expr))
            } else {
                // Constant folding - evaluate now
                let optimizer = SummationOptimizer::new();
                let result = optimizer.optimize_summation(start, end, expr.into())?;
                Ok(self.constant(result))
            }
        }
        
        // Data: Bound data gets evaluated, symbolic data gets deferred
        SummableRange::DataIteration { values } => {
            let expr = f(self.var());
            if self.has_unbound_variables(&expr) {
                // Has unbound variables - create symbolic expression
                Ok(self.create_bound_data_sum(values, expr))
            } else {
                // Pure data iteration - evaluate immediately
                let result = self.evaluate_data_sum(&values, &expr);
                Ok(self.constant(result))
            }
        }
        
        // Symbolic: Always defer
        SummableRange::SymbolicData { var_id } => {
            let expr = f(self.var());
            Ok(self.create_symbolic_data_sum(var_id, expr))
        }
    }
}
```

### Phase 3: Remove Deprecated Methods

```rust
#[deprecated(note = "Use sum() instead")]
pub fn sum_data<F>(&self, f: F) -> Result<TypedBuilderExpr<f64>> { ... }

#[deprecated(note = "Use sum() instead")]  
pub fn sum_range<F>(&self, range: RangeInclusive<i64>, f: F) -> Result<TypedBuilderExpr<f64>> { ... }
```

## Benefits

1. **Single Mental Model**: Always use `sum()` - the system figures out the right approach
2. **Clear Semantics**: Constant folding vs deferred evaluation based on variable binding
3. **Type-Driven Dispatch**: Input type determines optimization strategy
4. **Backward Compatible**: Existing `sum()` calls continue to work
5. **Future-Proof**: Easy to add new input types (e.g., `Stream<f64>`, `Iterator<f64>`)

## Migration Path

1. **Phase 1**: Implement unified `sum()` with extended trait support
2. **Phase 2**: Deprecate `sum_data()` and `sum_range()` with helpful error messages
3. **Phase 3**: Update all examples and documentation
4. **Phase 4**: Remove deprecated methods in next major version

## Example: Before vs After

### Before (Confusing)
```rust
// Three different methods for similar concepts
let math_sum = ctx.sum(1..=10, |i| i * ctx.constant(2.0))?;           // Mathematical
let data_sum = ctx.sum(data, |x| x * ctx.constant(3.0))?;             // Bound data  
let symbolic = ctx.sum_data(|x| x * param.clone())?;                  // Symbolic data
```

### After (Clear)
```rust
// One method, clear semantics
let math_sum = ctx.sum(1..=10, |i| i * ctx.constant(2.0))?;           // → Constant folding
let data_sum = ctx.sum(data, |x| x * ctx.constant(3.0))?;             // → Constant folding
let symbolic = ctx.sum(DataVariable::new(), |x| x * param.clone())?;   // → Deferred evaluation
```

The key insight: **The presence of unbound variables determines evaluation strategy, not the method name.** 

## Actually, Maybe Keep Current API?

Looking at the current implementation, the existing API is actually quite clear:

```rust
// Mathematical ranges - immediate evaluation when possible
ctx.sum(1..=10, |i| i * ctx.constant(2.0))

// Data arrays - immediate evaluation when no unbound variables  
ctx.sum(data, |x| x * ctx.constant(3.0))

// Symbolic data - explicit method for deferred evaluation
ctx.sum_data(|x| x * param.clone())
```

The key insight is that **`sum_data()` creates expressions that require `eval_with_data()`** - this is actually a clear semantic distinction! 