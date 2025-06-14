# Unified Sum Semantics

## Core Principle

**One `sum` method, automatic evaluation strategy based on variable binding.**

```rust
// Always use the same syntax
ctx.sum(input, |var| expression)
```

## Evaluation Rules

### Rule 1: No Unbound Variables → Immediate Evaluation
```rust
// All constants and bound data → evaluate now
ctx.sum(1..=10, |i| i * ctx.constant(2.0))
// → Constant(110.0)

let data = vec![1.0, 2.0, 3.0];
ctx.sum(data, |x| x * ctx.constant(2.0))
// → Constant(12.0)
```

### Rule 2: Has Unbound Variables → Rewrite Rules
```rust
let x = ctx.var(); // Unbound variable

// Factor extraction: Σ(k * f(i)) = k * Σ(f(i))
ctx.sum(1..=10, |i| i * ctx.constant(2.0) * x)
// → x * Constant(110.0)  // Factor out x, evaluate the sum

// Sum splitting: Σ(f(i) + g(i)) = Σ(f(i)) + Σ(g(i))
ctx.sum(1..=10, |i| i + x)
// → Constant(55.0) + x * Constant(10.0)  // Split and optimize each part
```

### Rule 3: Unbound Data → Symbolic Representation
```rust
let data = ctx.var::<Vec<f64>>(); // Unbound data variable

ctx.sum(data, |x| x.powi(2))
// → Sum{data, |x| x²}  // Keep symbolic until data is bound
```

## Rewrite Rules (Rust + egglog)

### Mathematical Optimizations
- **Factor extraction**: `Σ(k * f(i)) = k * Σ(f(i))`
- **Sum splitting**: `Σ(f(i) + g(i)) = Σ(f(i)) + Σ(g(i))`
- **Constant folding**: `Σ(c) = c * n`
- **Closed forms**: `Σ(i) = n(n+1)/2`, `Σ(i²) = n(n+1)(2n+1)/6`

### Nested Sum Handling
```rust
// Nested sums should be processed recursively
ctx.sum(1..=n, |i| ctx.sum(1..=i, |j| j * x))
// → Apply same rules to inner sum first, then outer sum
```

## Compilation to Closure

**After expression building, generate runtime closure:**

```rust
// Expression with unbound variables x, y, data
let expr = build_complex_expression_with_sums();

// Compile to closure
let compiled = ctx.compile(expr);
// → |params: &[f64], data: &[Vec<f64>]| -> f64

// Runtime evaluation
let result = compiled(&[x_val, y_val], &[data_vec]);
```

## Implementation Strategy

### Phase 1: Unified sum() Method
```rust
impl DynamicContext {
    pub fn sum<I, F>(&self, input: I, f: F) -> Result<DynamicExpr<f64>>
    where
        I: IntoSummableInput,
        F: Fn(DynamicExpr<f64>) -> DynamicExpr<f64>,
    {
        let expr = f(self.var());
        
        if self.has_unbound_variables(&expr) {
            // Apply rewrite rules and return symbolic expression
            self.apply_sum_rewrites(input, expr)
        } else {
            // Immediate evaluation
            self.evaluate_sum_immediately(input, expr)
        }
    }
}
```

### Phase 2: Rewrite Rule Engine
```rust
impl SumRewriter {
    fn apply_rewrites(&self, sum_expr: SumExpr) -> DynamicExpr<f64> {
        // 1. Factor extraction
        if let Some(factored) = self.extract_factors(&sum_expr) {
            return factored;
        }
        
        // 2. Sum splitting  
        if let Some(split) = self.split_sums(&sum_expr) {
            return split;
        }
        
        // 3. Closed form recognition
        if let Some(closed_form) = self.recognize_closed_form(&sum_expr) {
            return closed_form;
        }
        
        // 4. Keep symbolic
        self.create_symbolic_sum(sum_expr)
    }
}
```

### Phase 3: Variable Detection
```rust
impl DynamicContext {
    fn has_unbound_variables(&self, expr: &DynamicExpr<f64>) -> bool {
        // Check if expression contains variables that aren't bound to constants
        self.find_unbound_variables(expr).is_empty() == false
    }
    
    fn find_unbound_variables(&self, expr: &DynamicExpr<f64>) -> Vec<VarId> {
        // Walk AST and collect variable IDs that don't have bound values
        // ...
    }
}
```

## Benefits

1. **Single Mental Model**: Always use `sum()`, system figures out the strategy
2. **Automatic Optimization**: Rewrite rules applied transparently  
3. **Partial Evaluation**: Extract what can be computed, defer what can't
4. **Clean Compilation**: Final closure has clear runtime contract

## Examples

```rust
let ctx = DynamicContext::new();
let x = ctx.var();
let y = ctx.var();

// Complex expression with multiple sums
let expr = ctx.sum(1..=10, |i| {
    i * x + ctx.sum(1..=i, |j| j * y)
});

// Automatic rewriting:
// → x * Σ(i) + y * Σ(Σ(j for j=1..i) for i=1..10)
// → x * 55 + y * 220

// Compile to runtime closure
let compiled = ctx.compile(expr);
let result = compiled(&[2.0, 3.0]); // x=2, y=3 → 2*55 + 3*220 = 770
```

This design provides the mathematical elegance of automatic optimization with the practical benefits of efficient runtime evaluation. 