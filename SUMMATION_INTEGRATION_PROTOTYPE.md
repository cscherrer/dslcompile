# SummationExpr Integration Prototype

## Current State Analysis

### Existing Infrastructure ✅
- `SummationSimplifier` - Working implementation with pattern recognition
- `ASTFunction<T>` - Function representation for summands  
- `IntRange` - Range types implementing `RangeType`
- `SummandFunction<T>` trait - Already defined and implemented
- Pattern recognition (arithmetic, geometric, power series)
- Closed-form evaluation
- Multi-dimensional summations

### Current Usage Pattern
```rust
// How summations work today
let simplifier = SummationSimplifier::new();
let range = IntRange::new(1, 10);
let function = ASTFunction::linear("i", 2.0, 3.0);
let result = simplifier.simplify_finite_sum(&range, &function)?;
```

## Proposed Integration

### 1. Implement SummationExpr for ASTEval

```rust
impl SummationExpr for ASTEval {
    fn sum_finite<T, R, F>(range: Self::Repr<R>, function: Self::Repr<F>) -> Self::Repr<T>
    where
        T: NumericType,
        R: RangeType,
        F: SummandFunction<T>,
        Self::Repr<T>: Clone,
    {
        // Create a summation AST node
        ASTRepr::Sum {
            range: Box::new(range),
            function: Box::new(function),
        }
    }

    fn range_to<T: NumericType>(
        start: Self::Repr<T>,
        end: Self::Repr<T>,
    ) -> Self::Repr<IntRange> {
        // Convert expressions to IntRange
        // This requires evaluation or symbolic analysis
        match (start, end) {
            (ASTRepr::Constant(s), ASTRepr::Constant(e)) => {
                ASTRepr::Constant(IntRange::new(s as i64, e as i64))
            }
            _ => {
                // For non-constant ranges, we need symbolic range representation
                ASTRepr::SymbolicRange { start: Box::new(start), end: Box::new(end) }
            }
        }
    }

    fn function<T: NumericType>(
        index_var: &str,
        body: Self::Repr<T>,
    ) -> Self::Repr<ASTFunction<T>> {
        ASTRepr::Constant(ASTFunction::new(index_var, body))
    }
}
```

### 2. Extend ASTRepr for Summations

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ASTRepr<T> {
    // Existing variants...
    Constant(T),
    Variable(usize),
    Add(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    // ... other operations

    // New summation variants
    Sum {
        range: Box<ASTRepr<IntRange>>,
        function: Box<ASTRepr<ASTFunction<T>>>,
    },
    SymbolicRange {
        start: Box<ASTRepr<T>>,
        end: Box<ASTRepr<T>>,
    },
}
```

### 3. Implement SummationExpr for DirectEval

```rust
impl SummationExpr for DirectEval {
    fn sum_finite<T, R, F>(range: Self::Repr<R>, function: Self::Repr<F>) -> Self::Repr<T>
    where
        T: NumericType,
        R: RangeType,
        F: SummandFunction<T>,
        Self::Repr<T>: Clone,
    {
        // Direct evaluation using current SummationSimplifier
        let mut sum = T::zero();
        for i in range.start()..=range.end() {
            let value = function.apply(T::from(i).unwrap_or(T::zero()));
            sum = sum + value; // This requires T: Add<Output=T>
        }
        sum
    }

    fn range_to<T: NumericType>(start: Self::Repr<T>, end: Self::Repr<T>) -> Self::Repr<IntRange> {
        IntRange::new(
            start.to_i64().unwrap_or(0),
            end.to_i64().unwrap_or(0),
        )
    }

    fn function<T: NumericType>(
        index_var: &str,
        body: Self::Repr<T>,
    ) -> Self::Repr<ASTFunction<T>> {
        // For DirectEval, we need to create a function that captures the body value
        // This is tricky because DirectEval::Repr<T> = T (just the value)
        // We might need a different approach here
        ASTFunction::constant_func(index_var, body)
    }
}
```

## Problems Identified

### 1. Type System Mismatch
The current `SummationExpr` trait assumes that ranges and functions can be represented as `Self::Repr<R>` and `Self::Repr<F>`, but:

- `DirectEval::Repr<T> = T` - can't represent complex structures
- `ASTEval::Repr<T> = ASTRepr<T>` - works better but still has issues

### 2. Function Representation Challenge
```rust
// This doesn't work well:
fn function<T: NumericType>(
    index_var: &str,
    body: Self::Repr<T>,
) -> Self::Repr<ASTFunction<T>>;

// Because for DirectEval:
// Self::Repr<T> = T (just a value)
// Self::Repr<ASTFunction<T>> = ASTFunction<T> (but we can't construct this from just T)
```

### 3. Range Construction Issues
Similar problem with ranges - `DirectEval` can't easily construct `IntRange` from just values.

## Alternative Approach: Specialized Summation Interpreters

Instead of forcing summations into the general `MathExpr` trait, create specialized interpreters:

### Option A: Summation-Specific Interpreters

```rust
pub trait SummationInterpreter {
    type SumRepr;
    type RangeRepr;
    type FunctionRepr;

    fn sum_finite(range: Self::RangeRepr, function: Self::FunctionRepr) -> Self::SumRepr;
    fn evaluate_sum(sum: Self::SumRepr, variables: &[f64]) -> f64;
}

pub struct DirectSummationEval;
impl SummationInterpreter for DirectSummationEval {
    type SumRepr = f64;
    type RangeRepr = IntRange;
    type FunctionRepr = ASTFunction<f64>;

    fn sum_finite(range: Self::RangeRepr, function: Self::FunctionRepr) -> Self::SumRepr {
        let mut sum = 0.0;
        for i in range.iter() {
            let value = function.apply(i as f64);
            sum += DirectEval::eval_with_vars(&value, &[]);
        }
        sum
    }

    fn evaluate_sum(sum: Self::SumRepr, _variables: &[f64]) -> f64 {
        sum
    }
}

pub struct ASTSummationEval;
impl SummationInterpreter for ASTSummationEval {
    type SumRepr = ASTRepr<f64>;
    type RangeRepr = IntRange;
    type FunctionRepr = ASTFunction<f64>;

    fn sum_finite(range: Self::RangeRepr, function: Self::FunctionRepr) -> Self::SumRepr {
        // Use SummationSimplifier to get optimized form
        let mut simplifier = SummationSimplifier::new();
        let result = simplifier.simplify_finite_sum(&range, &function).unwrap();
        
        result.closed_form
            .or(result.telescoping_form)
            .unwrap_or_else(|| {
                // Fall back to explicit summation representation
                ASTRepr::Sum { range, function }
            })
    }

    fn evaluate_sum(sum: Self::SumRepr, variables: &[f64]) -> f64 {
        DirectEval::eval_with_vars(&sum, variables)
    }
}
```

### Option B: Enhance Current SummationSimplifier

Keep the current approach but make it more integrated:

```rust
// Enhanced SummationSimplifier that works with final tagless
impl SummationSimplifier {
    /// Create a summation expression that can be used with any MathExpr interpreter
    pub fn create_sum<E: MathExpr>(
        &mut self,
        range: IntRange,
        function: ASTFunction<f64>,
    ) -> Result<E::Repr<f64>>
    where
        E::Repr<f64>: Clone,
    {
        let result = self.simplify_finite_sum(&range, &function)?;
        
        if let Some(closed_form) = result.closed_form {
            // Convert ASTRepr to the target interpreter
            Ok(self.convert_ast_to_interpreter::<E>(closed_form))
        } else {
            // Fall back to numerical evaluation for DirectEval,
            // or create summation AST for ASTEval
            self.create_fallback_sum::<E>(range, function)
        }
    }

    fn convert_ast_to_interpreter<E: MathExpr>(&self, ast: ASTRepr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        match ast {
            ASTRepr::Constant(c) => E::constant(c),
            ASTRepr::Variable(i) => E::var_by_index(i),
            ASTRepr::Add(l, r) => E::add(
                self.convert_ast_to_interpreter::<E>(*l),
                self.convert_ast_to_interpreter::<E>(*r),
            ),
            // ... handle all AST variants
            _ => todo!("Implement all AST conversions"),
        }
    }
}
```

## Recommendation

**Option B (Enhanced SummationSimplifier)** is the most practical approach because:

1. ✅ **Preserves existing functionality** - Current `SummationSimplifier` keeps working
2. ✅ **Integrates with final tagless** - Can convert results to any interpreter
3. ✅ **Maintains performance** - Uses optimized closed forms when available
4. ✅ **Simpler implementation** - No need to extend `ASTRepr` or change core traits
5. ✅ **Backward compatible** - Existing code continues to work

## Implementation Plan

### Phase 1: AST Conversion Utility
```rust
pub struct ASTConverter;

impl ASTConverter {
    pub fn to_interpreter<E: MathExpr>(ast: &ASTRepr<f64>) -> E::Repr<f64>
    where
        E::Repr<f64>: Clone,
    {
        // Convert ASTRepr to any MathExpr interpreter
    }
}
```

### Phase 2: Enhanced SummationSimplifier
```rust
impl SummationSimplifier {
    pub fn create_expression<E: MathExpr>(
        &mut self,
        range: IntRange,
        function: ASTFunction<f64>,
    ) -> Result<E::Repr<f64>>
    where
        E::Repr<f64>: Clone,
    {
        // Implementation using ASTConverter
    }
}
```

### Phase 3: Ergonomic API
```rust
// Usage becomes:
let mut simplifier = SummationSimplifier::new();
let range = IntRange::new(1, 10);
let function = ASTFunction::linear("i", 2.0, 3.0);

// Get DirectEval result
let direct_result: f64 = simplifier.create_expression::<DirectEval>(range.clone(), function.clone())?;

// Get PrettyPrint result  
let pretty_result: String = simplifier.create_expression::<PrettyPrint>(range.clone(), function.clone())?;

// Get AST result
let ast_result: ASTRepr<f64> = simplifier.create_expression::<ASTEval>(range, function)?;
```

## Conclusion

The `SummationExpr` trait as currently defined is **not the right approach**. Instead, we should:

1. **Keep the trait definition** for future reference
2. **Enhance SummationSimplifier** to integrate with final tagless
3. **Create AST conversion utilities** to bridge between systems
4. **Provide ergonomic APIs** that work with all interpreters

This approach provides the benefits of trait-based summations without the type system complications. 