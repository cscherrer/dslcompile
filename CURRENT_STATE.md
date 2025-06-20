# DSL Compile - CURRENT STATE (Authoritative)

**Last Updated**: Current Session  
**Purpose**: Single source of truth for what actually works and what to use

**⚠️ CURRENT STATUS**: Library has compilation issues that prevent full functionality. Core architecture is in place but requires fixes before use.

## 🔄 SYSTEMS UNDER DEVELOPMENT

### Primary Context APIs

**🔄 DynamicContext** - Primary runtime API (needs compilation fixes)
```rust
use dslcompile::prelude::*;

let ctx = DynamicContext::new();
let x = ctx.var();
let expr = x * 2.0 + 1.0;
// Note: Currently has compilation issues
```

**🔄 Static Contexts** - Compile-time optimization (needs compilation fixes)
```rust
// StaticContext design is in place but requires compilation fixes
// Architecture supports zero-overhead compile-time optimization
```

### Evaluation Methods

**🔄 Designed**: `.eval(expr, hlist![...])` - HList-based evaluation (compilation issues)
**❌ Deprecated**: `.eval_old(expr, &[...])` - Array-based evaluation (legacy)

### Summation API

**🔄 Designed**: `ctx.sum(range, |i| expr)` - Collection summation (compilation issues)
```rust
// Mathematical ranges - design complete but needs compilation fixes
// let sum1 = ctx.sum(1..=10, |i| i * 2.0);
```

### Examples Status (Needs Verification)

*Note: These examples require verification after compilation issues are resolved:*
1. `comprehensive_iid_gaussian_demo.rs` - Needs verification
2. `anf_cse_performance_test.rs` - Needs verification  
3. `collection_codegen_demo.rs` - Needs verification
4. `static_context_iid_test.rs` - Needs verification
5. Examples using egg optimization - Need updates for current implementation

## 🟡 ARCHITECTURAL STATUS

Design decisions (from analysis):
- ✅ eval() vs eval_old() migration: **DESIGN COMPLETE** (needs compilation fixes)
- ✅ ExpressionBuilder/MathBuilder type aliases: **DEPRECATED** (use DynamicContext directly)
- ✅ sum_hlist() → sum() renaming: **DESIGN COMPLETE** 
- ✅ DataArray vs HList: **HList is primary** (compilation needs fixes)
- ✅ Egg optimization: **ARCHITECTURE IN PLACE** (replacing previous egglog experiments)

## 🔴 CURRENT ISSUES & DEPRECATED SYSTEMS

### Critical Issues

**⚠️ Compilation Errors**: Library currently has type errors preventing compilation
**⚠️ Examples Unverified**: Working examples need verification after compilation fixes
**⚠️ Performance Unverified**: Benchmarks cannot run due to compilation issues

### Avoid These

**❌ Type Aliases**: `ExpressionBuilder`, `MathBuilder` - Use `DynamicContext` directly
**❌ Old Eval**: `.eval_old()` - Use `.eval()` with HLists (once compilation is fixed)
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

## 🚫 HOW TO USE DSLCOMPILE RIGHT NOW

**⚠️ IMPORTANT: DSLCompile is currently not usable due to compilation issues.**

### Current Status
- **Architecture**: Core design is complete and sound
- **Dependencies**: Clean dependency tree with egg optimization
- **Compilation**: Type errors prevent building and testing
- **Priority**: Fix compilation issues before feature development

### Once Compilation is Fixed
```rust
// This is the intended API once issues are resolved:
use dslcompile::prelude::*;

let ctx = DynamicContext::new();
let x = ctx.var();
let expr = x * 2.0 + 1.0;
// let result = ctx.eval(&expr, hlist![3.0]); // Will work after fixes
```

## 🚫 WHAT NOT TO DO

1. **Don't create new infrastructure** - Use existing DynamicContext
2. **Don't use deprecated APIs** - Check this doc first
3. **Don't trust outdated documentation** - Only use examples that compile
4. **Don't build library code in demos** - Demos should demonstrate, not implement

## 🧭 DECISION TREE

**Want to use DSLCompile?** → ❌ **Fix compilation issues first**
**Want to understand the design?** → ✅ **Read architecture docs**
**Want to contribute?** → ✅ **Help fix type errors**
**Want working mathematical DSL?** → ❌ **Wait for compilation fixes**
**Confused about status?** → ✅ **This document is authoritative**

## 📋 VERIFICATION CHECKLIST

Current development priorities:
- [ ] Fix compilation errors in core library
- [ ] Verify examples compile and run correctly
- [ ] Update performance benchmarks with working code
- [ ] Validate egg optimization integration
- [ ] Test HList evaluation functionality

**Status**: ❌ Library currently fails compilation - fixes needed before feature work

---

**RULE**: When in doubt, prioritize fixing compilation issues over new feature development. The architecture is sound but requires working implementation. 