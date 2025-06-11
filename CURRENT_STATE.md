# DSL Compile - CURRENT STATE (Authoritative)

**Last Updated**: June 10, 2025  
**Purpose**: Single source of truth for what actually works and what to use

## 🟢 WORKING SYSTEMS (Use These)

### Primary Context APIs

**✅ DynamicContext** - Primary runtime API
```rust
use dslcompile::prelude::*;

let ctx = DynamicContext::new();
let x = ctx.var();
let expr = x * 2.0 + 1.0;
let result = ctx.eval(&expr, hlist![3.0]); // Modern HList evaluation
```

**✅ Static Contexts** - Compile-time optimization
```rust
// Context<T, SCOPE> for single-type operations  
// HeteroContext for multi-type operations
// See: examples/static_context_iid_test.rs
```

### Evaluation Methods

**✅ Current (Use This)**: `.eval(expr, hlist![...])`
**❌ Deprecated**: `.eval_old(expr, &[...])` - Still works but use HList version

### Summation API

**✅ Current**: `ctx.sum(range, |i| expr)` with unified API
```rust
// Mathematical ranges
let sum1 = ctx.sum(1..=10, |i| i * 2.0);

// Data collections (see memories for HList patterns)  
let sum2 = ctx.sum(data, |x| x * param);
```

### Working Examples (Verified)

1. `comprehensive_iid_gaussian_demo.rs` - **Complete probabilistic programming demo**
2. `anf_cse_performance_test.rs` - **ANF and CSE optimization**
3. `collection_codegen_demo.rs` - **Collection summation with code generation**
4. `static_context_iid_test.rs` - **Static context demonstration**
5. `minimal_egglog_demo.rs` - **Symbolic optimization**

## 🟡 MIGRATION STATUS

According to memories:
- ✅ eval() vs eval_old() migration: **COMPLETE**
- ✅ ExpressionBuilder/MathBuilder type aliases: **DEPRECATED** (use DynamicContext directly)
- ✅ sum_hlist() → sum() renaming: **COMPLETE**
- ✅ DataArray vs HList: **HList is primary, DataArray transitional**

## 🔴 KNOWN ISSUES & DEPRECATED SYSTEMS

### Avoid These

**❌ Type Aliases**: `ExpressionBuilder`, `MathBuilder` - Use `DynamicContext` directly
**❌ Old Eval**: `.eval_old()` - Use `.eval()` with HLists
**❌ DataArray Architecture**: Use HList collections instead

### Documentation Status

**🗑️ Outdated Docs** (ignore these):
- Multiple summation design docs describing deprecated approaches
- Old API unification plans that are superseded
- Migration guides for completed migrations

**✅ Current Docs** (trust these):
- `ROADMAP.md` - Current project status  
- This `CURRENT_STATE.md` - What to use right now
- Working examples in `dslcompile/examples/`

## 🎯 HOW TO USE DSLCOMPILE RIGHT NOW

### 1. Runtime Expressions
```rust
use dslcompile::prelude::*;

let ctx = DynamicContext::new();
let x = ctx.var();
let y = ctx.var();
let expr = (x + y) * 2.0;
let result = ctx.eval(&expr, hlist![3.0, 4.0]); // = 14.0
```

### 2. Mathematical Summations
```rust
let sum_expr = ctx.sum(1..=10, |i| i * i); // Σ i²
let result = ctx.eval(&sum_expr, hlist![]); // = 385
```

### 3. Code Generation
```rust
// See: examples/collection_codegen_demo.rs for working patterns
```

### 4. Optimization
```rust
// See: examples/anf_cse_performance_test.rs for ANF/CSE
// See: examples/minimal_egglog_demo.rs for symbolic optimization
```

## 🚫 WHAT NOT TO DO

1. **Don't create new infrastructure** - Use existing DynamicContext
2. **Don't use deprecated APIs** - Check this doc first
3. **Don't trust outdated documentation** - Only use examples that compile
4. **Don't build library code in demos** - Demos should demonstrate, not implement

## 🧭 DECISION TREE

**Want to build expressions?** → Use `DynamicContext::new()`
**Want to evaluate expressions?** → Use `.eval(expr, hlist![...])`  
**Want summations?** → Use `ctx.sum(range, |var| expr)`
**Want optimization?** → See working examples for proven patterns
**Want performance?** → Use static contexts or code generation
**Confused about API?** → Check a working example first

## 📋 VERIFICATION CHECKLIST

Before making changes:
- [ ] Does `cargo check --all-features --all-targets` pass?
- [ ] Is there a working example that does what I want?
- [ ] Am I using DynamicContext (not deprecated aliases)?
- [ ] Am I using .eval() with HLists (not arrays)?
- [ ] Am I demonstrating existing capabilities (not building new ones)?

---

**RULE**: When in doubt, find a working example and follow its patterns exactly. 