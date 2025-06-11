# Summation Integration Prototype

## Goals

1. **Static Context Integration**: Make `sum()` work with all contexts (`Context64`, `HeteroContext16`, etc.)
2. **Runtime Data Binding**: Create truly symbolic expressions that can be evaluated with different data
3. **Partial Evaluation Support**: Enable hybrid optimization where some data is inlined, some remains symbolic

## Two-Tier API Design

### 1. Mathematical Index Summation (Σᵢ₌₁ⁿ f(i))
- For compile-time known ranges: `1..=10`, `start..=end`
- Symbolic optimization with closed-form solutions
- Works with all contexts (Static and Dynamic)

### 2. Symbolic Data Summation (Σ(f(data[i]) for i in data))
- For runtime data binding: `Vec<f64>`, `&[f64]`
- Creates symbolic expressions with data variables
- Evaluation takes both expression parameters AND data

## Implementation Strategy

### Phase 1: Trait-Based Unification
```rust
trait SummationContext {
    type Expr<T>;
    fn sum_range<F>(&self, range: RangeInclusive<i64>, f: F) -> Self::Expr<f64>
    where F: Fn(Self::Expr<f64>) -> Self::Expr<f64>;
    
    fn sum_data<F>(&self, data_var: DataVariable, f: F) -> Self::Expr<f64>
    where F: Fn(Self::Expr<f64>) -> Self::Expr<f64>;
}
```

### Phase 2: Data Variables
```rust
// New AST variant for symbolic data
enum ASTRepr<T> {
    // ... existing variants ...
    DataVariable(DataVarId),  // References runtime data
    DataIndex(DataVarId, Box<ASTRepr<usize>>), // data[i] operation
}

// Runtime evaluation with data binding
fn eval_with_data(&self, params: &[T], data: &[Vec<T>]) -> T
```

### Phase 3: Partial Evaluation
```rust
// Some data inlined, some symbolic
let partially_evaluated = optimizer.partial_eval(
    expression,
    inline_data: &[data_slice_1],
    symbolic_data: &[data_var_2, data_var_3]
)?;
```

## Benefits

1. **Unified API**: Same `sum()` syntax across all contexts
2. **True Symbolic Data**: Data remains symbolic until evaluation time
3. **Performance**: Closed-form optimization for mathematical ranges, efficient iteration for data
4. **Flexibility**: Can mix static optimization with runtime data binding
5. **Type Safety**: Full type checking at all levels

## Implementation Plan

1. ✅ Current: Working `DynamicContext.sum()` with `SummationOptimizer`
2. 🔄 **Next**: Extend to static contexts (`Context64`, `HeteroContext16`)
3. 🔄 **Then**: Add data variables and symbolic data summation
4. 🔄 **Finally**: Integrate partial evaluation for hybrid optimization

## Example Usage

```rust
// Mathematical summation (works with all contexts)
let math_result = ctx.sum(1..=100, |i| i * i)?;

// Symbolic data summation (truly symbolic)
let data_var = ctx.data_variable::<f64>();
let symbolic_sum = ctx.sum_data(data_var, |x| x * x)?;

// Evaluation with different datasets
let result1 = ctx.eval_with_data(&symbolic_sum, &[], &[vec![1.0, 2.0, 3.0]]);
let result2 = ctx.eval_with_data(&symbolic_sum, &[], &[vec![4.0, 5.0, 6.0]]);

// Partial evaluation (some data inlined, some symbolic)
let hybrid = optimizer.partial_eval(
    &symbolic_sum,
    inline_data: &[constants],
    symbolic_data: &[runtime_data]
)?;
```

## Technical Implementation

### Step 1: Static Context Integration
- Add `SummationContext` trait
- Implement for `Context64`, `HeteroContext16`, etc.
- Reuse existing `SummationOptimizer` for mathematical ranges

### Step 2: Data Variables
- Extend AST with `DataVariable` and `DataIndex` variants
- Add data binding to evaluation methods
- Create `DataVariable` creation APIs

### Step 3: Evaluation Infrastructure
- Extend evaluation to handle data parameters
- Add `eval_with_data(expr, params, data_arrays)` methods
- Maintain backward compatibility with existing APIs

This design provides the foundation for true symbolic summation while maintaining the proven performance of mathematical optimization. 