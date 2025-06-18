# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- `cargo test` - Run all tests
- `cargo test --package dslcompile` - Test main package only
- `cargo test test_name` - Run specific test
- `cargo test --features all` - Test with all features enabled

### Building & Checking
- `cargo build` - Build the project
- `cargo build --release` - Release build with optimizations
- `cargo check` - Fast syntax/type checking
- `cargo clippy` - Lint checking

### Benchmarking
- `cargo bench` - Run performance benchmarks (uses Divan)
- `cargo run --example expression_optimization --release` - Optimization benchmarks

### Examples
- `cargo run --example multiset_demonstration` - Multiset functionality demo
- `cargo run --example simple_math_demo` - Basic mathematical operations demo
- `cargo run --example measures_library_demo` - Measures library patterns (in root `examples/`)
- Examples are in `dslcompile/examples/`

## Architecture Overview

DSLCompile is a mathematical expression compiler with a three-layer optimization strategy:

### 1. Expression Building Layer (Final Tagless)
- **Core trait**: `MathExpr` with Generic Associated Types (GATs)
- **Two primary contexts**:
  - `StaticContext`: Compile-time optimization, zero-overhead, HList support
  - `DynamicContext`: Runtime flexibility, JIT compilation, symbolic optimization
- **Key types**: `ASTRepr<T>`, `DynamicExpr`, `StaticExpr`

### 2. Symbolic Optimization Layer
- **Engine**: Uses egglog for algebraic simplification
- **Location**: `src/symbolic/` module
- **Entry point**: `SymbolicOptimizer` class
- **Rules**: Mathematical simplification rules in `src/egglog_rules/*.egg`

### 3. Compilation Backend Layer
- **Primary**: Rust hot-loading compilation (`src/backends/rust_codegen.rs`)
- **Optional**: Cranelift JIT (feature-gated)
- **Output**: Native performance compiled functions

## Key Design Patterns

### Final Tagless Approach
The codebase solves the "expression problem" using final tagless design:
- **Interpreters**: `DirectEval`, `PrettyPrint`, `ASTEval`
- **Extensions**: `StatisticalExpr`, `SummationExpr` traits
- **Benefits**: Easy to add operations or interpreters without modifying existing code

### Two-Context Architecture
After recent consolidation, there are exactly two clean interfaces:

1. **StaticContext** (`src/contexts/static_context/`):
   - Zero-overhead, compile-time scoped variables
   - HList heterogeneous type support
   - Automatic scope management

2. **DynamicContext** (`src/contexts/dynamic/`):
   - Runtime flexibility with variable registries
   - Operator overloading syntax
   - Integration with symbolic optimization

### Variable Management
- **Internal**: Uses integer indices for performance
- **External**: String names for user convenience
- **Registry**: `VariableRegistry` maps names ↔ indices thread-safely

## Module Structure

### Core Modules
- `src/ast/`: Abstract syntax tree representations and evaluation
- `src/contexts/`: StaticContext and DynamicContext implementations
- `src/symbolic/`: Symbolic optimization using egglog
- `src/backends/`: Code generation and compilation
- `src/error.rs`: Unified error handling

### Important Files
- `src/lib.rs`: Main API exports and prelude
- `src/contexts/dynamic/expression_builder.rs`: DynamicContext implementation
- `src/contexts/static_context/static_scoped.rs`: StaticContext core
- `src/symbolic/symbolic.rs`: SymbolicOptimizer implementation
- `src/backends/rust_codegen.rs`: Rust code generation

## Workspace Structure

This is a Cargo workspace with two packages:
- `dslcompile/`: Main library package
- `dslcompile-macros/`: Procedural macros for compile-time optimization

## Features

- `default = ["optimization"]`: Includes egglog symbolic optimization
- `optimization`: Enables symbolic optimization with egglog
- `ad_trait`: Automatic differentiation trait support
- `all`: All features enabled

## Testing Strategy

- **Unit tests**: In `src/` modules and `tests/` directory
- **Property tests**: Uses `proptest` for algebraic property verification
- **Integration tests**: End-to-end pipeline testing
- **Benchmarks**: Performance regression testing with Divan

## Current API Quick Reference

### 🚨 Pre-Development Checklist

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

### ✅ Current APIs (Always Use)

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

### ❌ Deprecated APIs (Never Use)

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

### 🎯 Standard Demo Patterns

#### Basic Expression Demo
```rust
use dslcompile::prelude::*;

let ctx = DynamicContext::new();
let x = ctx.var();
let y = ctx.var();
let expr = (x + y) * 2.0;
let result = ctx.eval(&expr, hlist![3.0, 4.0]);
println!("Result: {result}"); // Should be 14.0
```

#### Summation Demo  
```rust
let sum_expr = ctx.sum(1..=5, |i| i * i);
let result = ctx.eval(&sum_expr, hlist![]);
println!("Sum of squares 1-5: {result}"); // Should be 55
```

#### Reference Working Examples
- `multiset_demonstration.rs` - Demonstrates multiset functionality and operations
- `simple_math_demo.rs` - Basic mathematical expression building and evaluation
- `measures_library_demo.rs` - Complex probabilistic programming patterns (in root examples)

### 🧠 API Translation Guide

When memories or old documentation mention:
- "ExpressionBuilder" → It's now DynamicContext
- "eval_hlist()" → It's now eval()  
- "sum_hlist()" → It's now sum()
- "DataArray vs HList" → HList is current

### ⚡ Pre-Submission Checklist

Before submitting code:
- [ ] Uses `DynamicContext::new()`
- [ ] Uses `.eval()` with `hlist![]`
- [ ] Uses `.sum()` for summations
- [ ] Follows pattern from working example
- [ ] Actually demonstrates (doesn't implement new library code)

### 🎯 Development Philosophy

1. **Demonstrate existing capabilities** - Show what DSLCompile can do
2. **Follow established patterns** - Don't reinvent, reuse
3. **Trust working examples** - They are ground truth
4. **Check CURRENT_STATE.md first** - It's authoritative
5. **When confused, find working example** - Don't guess

**EMERGENCY BRAKE**: If about to create new infrastructure, STOP and find an existing example first.

## Common Development Tasks

### Adding New Mathematical Operations
1. Extend the `MathExpr` trait or create extension trait
2. Implement for all interpreters: `DirectEval`, `PrettyPrint`, `ASTEval`
3. Add corresponding `ASTRepr` variant if needed
4. Update symbolic optimization rules in `.egg` files

### Adding New Optimization Rules
1. Create or modify `.egg` files in `src/egglog_rules/`
2. Update `rule_loader.rs` to include new rules
3. Test with symbolic optimization benchmarks

### Performance Optimization
- Profile with `cargo bench` using Divan
- Check generated code with `RustCodeGenerator`
- Verify optimization effectiveness with symbolic simplification

## Git Workflow

- Main development branch: `dev`
- Production branch: `master` (use for PRs)
- Current status shows modified file: `dslcompile/src/contexts/dynamic/expression_builder.rs`

## Project Philosophical Memory

- Our main goal is to have an intuitive API for building function expressions in a way that they can be arbitrarily composed, then optimized using egglog. 
- Our rewrite rules must be thoroughly tested for semantic correctness. 
- We must organize things in such a way as to allow aggressive rewrites without combinatorial explosion of the e-graph. 
- Following this rewrite step, users can either call directly as an interpreter, or compile to very fast Rust code. In the latter case, the Rust can be dynamically or statically linked.

## Future Library Design: Probability Measures and Densities

Eventually dslcompile will be a dependency for a "measures" library. Most libraries for distributions have a few problems:
- They assume log-density with respect to Lebesgue or counting measure, and are unable to account for simplifications for log-densities with other base measures
- They have no good way of caching values. They either compute in advance (whether or not a value is needed) or use something like a OnceCell that has significant overhead

Instead, we should be able to define, e.g. a "Normal" log-density wrt Lebesgue measure, and then automatically simplify if a user requests the log-density wrt another normal. Also, we should be able to define an iid combinator that takes a measure and returns a new measure representing iid copies.

So we'll have (1) a Normal struct taking mean and std dev. These will be type-parameterized, so we can for example create a Normal where mean and/or std dev is a dslcompile variable. Then (2) an iid struct wrapping another measure. 

In our mini version of this, we should define these structs with log_density methods. For now leave out the base measure. We'll just show composability and the ability to optimize. As always, it's critical that summation be represented symbolically. And we need to avoid the mistake of defining a placeholder to iterate over - this will instead be passed as a variable. This way we can compile the code once and then call it multiple times.

## Development Guidance
- DO NOT create a src at top level. Use @dslcompile/src/ or @dslcompile-macros/src/ instead

## Egglog Programming Patterns

### ✅ Correct Egglog Patterns (from /home/chad/git/dslcompile/egglog/tests)

**Pattern 1: Expression Binding + Guards (from interval.egg)**
```egglog
(rule ((= mul (Mul a b))           ; Bind composite expression to variable
       (= loa (lo a))              ; Guard: require function values exist
       (= lob (lo b))
       (= hia (hi a))
       (= hib (hi b)))
      ((set (lo mul)               ; Then compute result
          (min (min (* loa lob) (* loa hib))
               (min (* hia lob) (* hia hib))))))
```

**Pattern 2: Relations for Complex Conditions (from math.egg)**
```egglog
(relation evals-to (Math i64))
(rule ((= e (Add a b)) (evals-to a va) (evals-to b vb))
      ((evals-to e (+ va vb))))
```

### ❌ Patterns That Cause Infinite Loops

**Anti-Pattern: Direct Function Calls in RHS**
```egglog
; DON'T DO THIS - causes infinite loops
(rule ((Add ?a ?b))
      ((compute-free-vars ?a)      ; This triggers the rule again!
       (compute-free-vars ?b)
       (compute-free-vars (Add ?a ?b))))
```

**Anti-Pattern: Self-Triggering Rules**
```egglog
; DON'T DO THIS - rule fires on its own output
(rule ((Add ?a ?b))
      ((set (free-vars (Add ?a ?b)) (UnionSet (free-vars ?a) (free-vars ?b)))))
```

### 🎯 Dependency Analysis Pattern

For tracking which variables an expression depends on:

```egglog
(datatype VarSet
  (EmptySet)
  (SingleVar i64)
  (UnionSet VarSet VarSet))

(function free-vars (Math) VarSet :merge (UnionSet old new))

; Base cases
(rule ((Num ?c))
      ((set (free-vars (Num ?c)) (EmptySet))))

(rule ((UserVar ?v))
      ((set (free-vars (UserVar ?v)) (SingleVar ?v))))

; Composite expressions - use binding + guards
(rule ((= add_expr (Add ?a ?b))
       (= dep_a (free-vars ?a))
       (= dep_b (free-vars ?b)))
      ((set (free-vars add_expr) (UnionSet dep_a dep_b))))
```

### 🔧 Function Merge Strategies

- `:merge (UnionSet old new)` - Combine dependency sets
- `:merge (min old new)` - Take minimum (for intervals)
- `:merge (max old new)` - Take maximum
- `:no-merge` - No automatic merging

### 📚 Reference Examples
- `egglog/tests/interval.egg` - Functions on composite expressions
- `egglog/tests/math.egg` - Relations and complex conditions  
- `egglog/tests/set.egg` - Set operations and no-merge functions

## Egglog Algebraic Rewrite Rules Guidance
- For egglog algebraic rewrite rules, we'll start simple. When we see a result that should simplify further, we'll think carefully about what rules to add

## Helpful Egglog Examples
- there are some helpful egglog examples in /home/chad/git/dslcompile/egglog-tests
- Key reference files in /home/chad/git/dslcompile/egglog/tests/

## Recent Changes
- Added egglog-cheatsheet.md to the root project directory