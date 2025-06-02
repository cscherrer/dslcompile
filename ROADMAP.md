# MathCompile Development Roadmap

## Project Overview

MathCompile is a mathematical expression compiler that transforms symbolic mathematical expressions into executable code. The project provides tools for mathematical computation with symbolic optimization.

## Current Status (June 2025)

### Implemented Features
- **Compile-Time Egglog Optimization**: Procedural macro system with safe termination rules
- **Domain-Aware Runtime Optimization**: ANF integration with interval analysis and mathematical safety
- **Final Tagless Expression System**: Type-safe expression building with multiple interpreters
- **Multiple Compilation Backends**: Rust hot-loading and optional Cranelift JIT
- **Index-Only Variable System**: High-performance variable tracking with zero-cost execution

#### Safe Egglog Implementation
```rust
// SAFE SIMPLIFICATION RULES (no expansion)
(rewrite (Add a (Num 0.0)) a)           // x + 0 → x
(rewrite (Mul a (Num 1.0)) a)           // x * 1 → x  
(rewrite (Ln (Exp x)) x)                // ln(exp(x)) → x
(rewrite (Pow a (Num 1.0)) a)           // x^1 → x

// STRICT LIMITS:
(run 3)  // Limited iterations prevent runaway optimization
```

#### Index-Only Variable System (NEW - June 2, 2025)
- **TypedVariableRegistry**: Pure index-based variable tracking with compile-time type safety
- **Zero-Cost Execution**: No string lookups during evaluation - only integer indexing
- **Type Category System**: Compile-time type tracking with automatic promotion rules
- **Composable Design**: Optional string mapping for development convenience without runtime overhead
- **Backward Compatibility**: Maintains existing APIs while enabling high-performance execution
- **Documentation Alignment (June 2, 2025)**: All doctests and documentation updated to match index-only API

```rust
// NEW API - Index-Only Variables
let math = MathBuilder::new();
let x = math.var();  // Returns var_0, tracked by index
let y = math.var();  // Returns var_1, tracked by index

// Zero-cost evaluation with direct indexing
let result = math.eval(&expr, &[3.0, 4.0]);  // No string lookup overhead
```

#### Domain-Aware ANF Integration
- **DomainAwareANFConverter Implementation**: Core domain-aware ANF conversion with interval analysis
- **Safety Validation**: Mathematical operation safety (ln requires x > 0, sqrt requires x >= 0, div requires non-zero)
- **Variable Domain Tracking**: Domain information propagation through ANF transformations
- **Error Handling**: DomainError variant with proper error formatting and conservative fallback
- **Integration**: Full export in lib.rs and prelude

### Recent Improvements (June 2025)
- **Documentation Cleanup**: Removed promotional language, unfounded performance claims, and sales talk
- **Technical Focus**: Updated documentation to focus on technical implementation details
- **Consistent Messaging**: Aligned all documentation with factual, technical descriptions
- **Variable System Overhaul (June 2, 2025)**: Implemented index-only variable tracking for maximum performance and composability
- **API Documentation Fix (June 2, 2025)**: Corrected all doctests to match the implemented index-only API, resolving test failures and ensuring documentation accuracy

---

## System Architecture

### Dual-Path Optimization Strategy
```
User Code (mathematical syntax)
     ↓
┌─────────────────┬─────────────────┐
│  COMPILE-TIME   │   RUNTIME       │
│  PATH           │   PATH          │
├─────────────────┼─────────────────┤
│ Known           │ Dynamic         │
│ Expressions     │ Expressions     │
│                 │                 │
│ Procedural      │ AST →           │
│ Macro           │ Normalize →     │
│ ↓               │ ANF+CSE →       │
│ Safe Egglog     │ Domain-Aware    │
│ (3 iterations)  │ Egglog →        │
│ ↓               │ Extract →       │
│ Direct Code     │ Denormalize     │
│ Generation      │ (Variable)      │
└─────────────────┴─────────────────┘
     ↓                    ↓
Optimized Execution   Safe Execution
```

### When to Use Each Path

#### **Compile-Time Path** (Procedural Macro)
- **Use for**: Known expressions, performance-critical code
- **Benefits**: Compile-time optimization with egglog
- **Status**: Implemented

#### **Runtime Path** (Domain-Aware ANF)
- **Use for**: Dynamic expressions, complex optimization scenarios
- **Benefits**: Mathematical safety with runtime adaptability
- **Status**: Implemented with comprehensive safety

---

## Development Phases

### Phase 1: Foundation (Completed)
- ✅ **Implemented `optimize_compile_time!` procedural macro**
  - ✅ **Egglog optimization** with safe termination rules
  - ✅ **Safe iteration limits** preventing infinite expansion (3 iterations max)
  - ✅ **Direct Rust code generation** for optimized patterns
  - ✅ **Memory safety**: Normal compilation behavior

- ✅ **Completed domain-aware runtime optimization**
  - ✅ **Complete normalization pipeline**: Canonical form transformations
  - ✅ **Dynamic rule system**: Organized rule loading with multiple configurations
  - ✅ **Native egglog integration**: Domain-aware optimizer with interval analysis
  - ✅ **ANF integration**: Domain-aware A-Normal Form with mathematical safety
  - ✅ **Mathematical correctness**: Domain safety implementation

### Phase 2: System Integration (Current)
- ✅ **Documentation Cleanup**: Removed sales talk and unfounded claims
- [ ] **Hybrid Bridge Implementation**
  - Add `into_ast()` method to compile-time traits
  - Enable seamless compile-time → runtime egglog pipeline
  - Benchmark hybrid optimization performance

- [ ] **Expand safe egglog capabilities**
  - Add more mathematical optimization rules with safety guarantees
  - Support complex multi-variable expressions with termination bounds
  - Advanced pattern matching with controlled expansion

- [ ] **Complete ANF Integration**
  - **Safe Common Subexpression Elimination**
  - Enhance CSE to use domain analysis for safety checks
  - Prevent CSE of expressions with different domain constraints
  - Add domain-aware cost models for CSE decisions

### Phase 3: Advanced Features (Planned)
- [ ] **SummationExpr implementation via safe egglog**
  - Integrate summation patterns with bounded egglog optimization
  - Support finite/infinite/telescoping sums with termination guarantees
  - Generate optimized loops or closed-form expressions safely

- [ ] **Advanced safe optimization patterns**
  - Trigonometric identities with expansion limits
  - Logarithmic and exponential optimizations with bounds
  - Polynomial factorization with controlled complexity

### Phase 4: Production Ready (Planned)
- [ ] **Performance optimization and validation**
  - Benchmark against existing approaches
  - Optimize safe egglog compilation time
  - Validate correctness and termination across edge cases

- [ ] **Documentation & ecosystem**
  - Complete API documentation with safety examples
  - Safe egglog optimization guide
  - Migration guide from existing systems

---

## Technical Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Safe Egglog Macro** | ✅ Implemented | Compile-time optimization with termination guarantees |
| **Domain-Aware Runtime** | ✅ Implemented | Mathematical safety with interval analysis |
| **Index-Only Variables** | ✅ Implemented | Zero-cost variable tracking with type safety |
| **Compile-Time Traits** | ✅ Implemented | Type-safe expression building |
| **Final Tagless AST** | ✅ Implemented | Multiple interpreter support |
| **ANF Integration** | ✅ Implemented | Domain-aware A-Normal Form |
| **JIT Compilation** | ✅ Implemented | Optional Cranelift backend |
| **Documentation** | ✅ Cleaned | Technical focus, removed promotional content |

---

## Optimization Routes

### Egglog-Based Optimization
- **Compile-time egglog**: Safe rule application with iteration limits
- **Runtime egglog**: Dynamic optimization with domain awareness
- **Hybrid approach**: Compile-time traits bridging to runtime optimization

### Mathematical Safety
- **Domain analysis**: Interval-based safety checking
- **Conservative fallbacks**: Safe behavior for edge cases
- **Error handling**: Proper error propagation and recovery

### Performance Considerations
- **Compilation time**: Balance optimization depth with build time
- **Runtime performance**: Optimize hot paths while maintaining safety
- **Memory usage**: Efficient representation and transformation

---

## Contributing

The project follows standard Rust development practices:

1. **Testing**: Comprehensive test suite including property-based tests
2. **Documentation**: Technical documentation focusing on implementation details
3. **Safety**: Mathematical correctness and domain safety as primary concerns
4. **Performance**: Optimization balanced with correctness and maintainability

For specific implementation details, see the [Developer Notes](DEVELOPER_NOTES.md).

## ✅ Completed Features

### Core Infrastructure (2025-06-02)
- **Index-Only Variable System Migration**: ✅ COMPLETED
  - Removed old string-based `VariableRegistry` and `ExpressionBuilder`
  - Migrated `MathExpr` trait to use `fn var(index: usize)` instead of `fn var(name: &str)`
  - All variable operations now use indices for maximum performance
  - **Breaking Change**: This removes backward compatibility with string-based variable names
  - **Status**: Trait definition complete, interpreter implementations in progress

### Previous Completions
- **Symbolic ANF (A-Normal Form)**: ✅ COMPLETED (2025-05-30)
  - Clean representation separating computations from variables
  - Efficient Rust code generation
  - Integration with interval domain analysis

- **Interval Domain Analysis**: ✅ COMPLETED (2025-05-30)
  - Range propagation through mathematical expressions
  - Optimization opportunities through bounds analysis
  - Foundation for symbolic manipulation

- **Cranelift JIT Backend**: ✅ COMPLETED
  - High-performance native code generation
  - Type-safe compilation from AST representations
  - Integration with final tagless interpreters
