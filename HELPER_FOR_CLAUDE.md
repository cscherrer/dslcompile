# Quick Reference for Claude (Helper File)

## 🚨 STOP AND CHECK FIRST

Before making ANY changes, answer these questions:

1. **Is there already a working example that does this?**
   - Check `dslcompile/examples/` directory
   - If YES: Follow that pattern exactly
   - If NO: Proceed carefully

2. **Am I demonstrating or implementing?**
   - DEMO: Use existing DynamicContext API
   - IMPLEMENT: Only if explicitly requested

3. **What does CURRENT_STATE.md say?**
   - Always check this first for current APIs
   - Trust this over memories or other docs

## 🟢 ALWAYS USE (Current APIs)

```rust
// Expression building
let ctx = DynamicContext::new();          // NOT ExpressionBuilder
let x = ctx.var();                        // Variables
let expr = x * 2.0 + 1.0;                // Expressions

// Evaluation  
let result = ctx.eval(&expr, hlist![3.0]); // NOT .eval_old(&expr, &[3.0])

// Summation
let sum_expr = ctx.sum(1..=10, |i| i * 2.0); // Unified API
```

## 🔴 NEVER USE (Deprecated)

```rust
// WRONG - Deprecated aliases
let ctx = ExpressionBuilder::new();     // Use DynamicContext
let ctx = MathBuilder::new();           // Use DynamicContext

// WRONG - Old evaluation
ctx.eval_old(&expr, &[3.0, 4.0]);      // Use .eval() with hlist![]

// WRONG - Old summation
ctx.sum_hlist();                        // Use .sum()
ctx.sum_data();                         // Use .sum()
```

## 🎯 DEMO PATTERNS (Follow These)

### Basic Expression Demo
```rust
use dslcompile::prelude::*;

let ctx = DynamicContext::new();
let x = ctx.var();
let y = ctx.var();
let expr = (x + y) * 2.0;
let result = ctx.eval(&expr, hlist![3.0, 4.0]);
println!("Result: {result}"); // Should be 14.0
```

### Summation Demo  
```rust
let sum_expr = ctx.sum(1..=5, |i| i * i);
let result = ctx.eval(&sum_expr, hlist![]);
println!("Sum of squares 1-5: {result}"); // Should be 55
```

### Follow Working Examples
- `comprehensive_iid_gaussian_demo.rs` - Complex data-driven summation
- `anf_cse_performance_test.rs` - Optimization pipeline
- `collection_codegen_demo.rs` - Code generation

## 🧠 MEMORY HELPERS

When memories mention:
- "ExpressionBuilder" → It's now DynamicContext
- "eval_hlist()" → It's now eval()  
- "sum_hlist()" → It's now sum()
- "DataArray vs HList" → HList is current

## ⚡ QUICK CHECKS

Before submitting code:
- [ ] Uses `DynamicContext::new()`
- [ ] Uses `.eval()` with `hlist![]`
- [ ] Uses `.sum()` for summations
- [ ] Follows pattern from working example
- [ ] Actually demonstrates (doesn't implement new library code)

## 🎯 CLAUDE'S MISSION

1. **Demonstrate existing capabilities** - Show what DSLCompile can do
2. **Follow established patterns** - Don't reinvent, reuse
3. **Trust working examples** - They are ground truth
4. **Check CURRENT_STATE.md first** - It's authoritative
5. **When confused, find working example** - Don't guess

---

**EMERGENCY BRAKE**: If I'm about to create new infrastructure, STOP and find an existing example first. 