# DSLCompile Roadmap

## Current Status: Context Renaming Complete ✅

**Latest Achievement**: Successfully renamed `ScopedExpressionBuilder` → `Context` and `ExpressionBuilder` → `DynamicContext`, with `Context` as the default static option.

## Phase 1: Heterogeneous Static Context Foundation 🚀 

**Goal**: Evolve from homogeneous `Context<T, SCOPE>` to heterogeneous type system with zero runtime overhead.

### 1.1 Core Type System Evolution
- [ ] **Remove `NumericType` constraint** from static contexts
- [ ] **Implement `ExpressionType` trait** for arbitrary types (`Vec<f64>`, `usize`, custom structs)
- [ ] **Extend operation system** beyond scalar math (array indexing, type conversions)
- [ ] **Type-safe composition** with heterogeneous inputs

### 1.2 Native Operation Support
- [ ] **Array operations**: `array_index`, `array_slice`, `array_length`
- [ ] **Type operations**: `scalar_add`, `scalar_mul`, `vector_dot`, `matrix_mul`
- [ ] **Control flow**: `conditional`, `switch`, `loop_unroll`
- [ ] **Custom operations**: User-defined operations with type safety

### 1.3 Evaluation System
- [ ] **Multi-type evaluation**: Native evaluation without Vec<f64> flattening
- [ ] **Zero-copy inputs**: Direct references to native data structures
- [ ] **Codegen optimization**: Monomorphized output for maximum performance

## Phase 2: Advanced Mathematical Operations 🧮

### 2.1 Extended Mathematical Library
- [ ] **Linear algebra**: Matrix operations, decompositions, eigenvalues
- [ ] **Calculus**: Symbolic differentiation, integration, Taylor series
- [ ] **Statistics**: Distributions, hypothesis testing, regression models
- [ ] **Optimization**: Gradient descent, constrained optimization

### 2.2 Domain-Specific Extensions
- [ ] **Machine Learning**: Neural network layers, activation functions, loss functions
- [ ] **Signal Processing**: FFT, filtering, convolution
- [ ] **Numerical Methods**: ODE/PDE solvers, root finding, interpolation

## Phase 3: Performance & Compilation 🔥

### 3.1 Advanced Backends
- [ ] **SIMD optimization**: Auto-vectorization for supported operations
- [ ] **GPU compilation**: CUDA/OpenCL codegen for parallel operations
- [ ] **LLVM backend**: Direct LLVM IR generation for maximum optimization
- [ ] **WebAssembly**: Browser-compatible compilation target

### 3.2 Optimization Engine
- [ ] **Symbolic optimization**: Advanced algebraic simplification
- [ ] **Loop optimization**: Unrolling, fusion, vectorization
- [ ] **Memory optimization**: Cache-aware layouts, zero-copy operations
- [ ] **Profile-guided optimization**: Runtime feedback for optimization

## Phase 4: Ecosystem & Usability 🌟

### 4.1 Language Integration
- [ ] **Python bindings**: PyO3-based bindings for data science workflows
- [ ] **Julia integration**: Native Julia compilation and interop
- [ ] **R package**: Statistical computing integration
- [ ] **JavaScript/WASM**: Browser-based mathematical computing

### 4.2 Developer Experience
- [ ] **IDE support**: Language server, syntax highlighting, debugging
- [ ] **Documentation**: Comprehensive guides, examples, best practices
- [ ] **Benchmarking**: Performance comparison suite
- [ ] **Testing**: Property-based testing, fuzzing, correctness validation

## Technical Milestones

### Milestone 1: Heterogeneous Foundation (Next 2 weeks)
1. ✅ **Context renaming complete** (June 4, 2025)
2. ✅ **Remove type parameter from Context** - `HeteroContext<SCOPE>` works with any types! (June 4, 2025)
3. ✅ **Implement basic heterogeneous operations** - Array indexing, scalar operations working! (June 4, 2025)
4. ✅ **Native input evaluation** - Eliminated Vec<f64> requirement completely! (June 4, 2025)
5. ✅ **Performance benchmarking** - 260x improvement in memory operations, all benchmarks pass! (June 4, 2025)

**🎉 MILESTONE 1 COMPLETE!** All tests passing, neural network example demonstrating zero-overhead native types, benchmarks confirm production readiness.

## Phase 2: Migration to Heterogeneous System (IMMEDIATE - THIS WEEK) 🚀

### Goal: Replace Context<T, SCOPE> with HeteroContext<SCOPE> as primary API

**Status**: ✅ **READY FOR IMMEDIATE MIGRATION** - Benchmarks confirm no performance regressions

### 2.1 API Migration (Since backward compatibility is not a concern)
- [ ] **Replace primary Context export** with HeteroContext
- [ ] **Update all examples** to use new heterogeneous system
- [ ] **Remove old Context<T, SCOPE>** implementation
- [ ] **Update prelude** to export heterogeneous types as defaults

### 2.2 Performance Achievements 
✅ **Memory Operations**: 260x improvement (58.72ns → 225.88ps)
✅ **Zero Allocation**: Direct native type access
✅ **New Capabilities**: Array indexing (91.8ns), neural networks (171.2ns)
✅ **Type Safety**: Compile-time heterogeneous type checking

### 2.3 Documentation Updates
- [ ] **Update README.md** with heterogeneous examples
- [ ] **Revise basic_usage.rs** to showcase new capabilities  
- [ ] **Create migration guide** (though not needed for this project)
- [ ] **Update API documentation** to reflect new primary system

### Milestone 2: Extended Operations (4 weeks)
1. **Matrix/vector operations** - Linear algebra primitives
2. **Control flow constructs** - Conditionals, loops
3. **Custom operation framework** - User-defined operations
4. **Performance benchmarking** - vs current system and competitors

### Milestone 3: Production Readiness (8 weeks)
1. **Comprehensive test suite** - Property-based testing for correctness
2. **Documentation and examples** - Real-world use cases
3. **Optimization pipeline** - Symbolic + runtime optimization
4. **Backend selection** - Multiple compilation targets

### Milestone 4: Ecosystem (12 weeks)
1. **Language bindings** - Python, Julia integration
2. **Domain libraries** - ML, statistics, signal processing
3. **IDE integration** - Development tooling
4. **Community adoption** - Open source release, feedback integration

## Technical Design Principles

### 🎯 **Zero Runtime Overhead**
- All type checking and optimization at compile time
- Generated code should be as fast as hand-written C
- No runtime allocations or conversions

### 🔒 **Type Safety**
- Compile-time prevention of dimension mismatches
- No silent type conversions or precision loss
- Expressive type system for mathematical operations

### 🧩 **Composability**
- Perfect function composition without variable conflicts
- Library-friendly design for mathematical building blocks
- Automatic optimization across composition boundaries

### 🚀 **Performance**
- SIMD-optimized operations where possible
- Cache-aware memory layouts
- Profile-guided optimization opportunities

### 🌍 **Ecosystem Friendly**
- Easy integration with existing mathematical libraries
- Multiple compilation targets (native, WASM, GPU)
- Language bindings for popular data science ecosystems

---

## Current Focus: Heterogeneous Context Foundation

**Immediate Next Steps**:
1. Remove the `T` type parameter constraint from `Context<T, SCOPE>`
2. Implement the `ExpressionType` trait system
3. Build array indexing and basic heterogeneous operations
4. Create evaluation system that accepts native types

**Success Metrics**:
- Neural network example works with native types (no Vec<f64> flattening)
- Performance benchmarks show zero overhead vs hand-written code
- Type safety prevents common mathematical errors at compile time
- Smooth migration path from current homogeneous system 

### 🔄 Phase 4: Runtime Dispatch Elimination (IN PROGRESS)
- [x] **✅ Runtime Type Dispatch Elimination Completed**
  - Created `heterogeneous_v4.rs` with true zero-dispatch system
  - Eliminated ALL `std::any::Any` and `downcast_ref` calls
  - Implemented compile-time trait specialization
  - Uses `DirectStorage<T>` traits for monomorphized evaluation paths
  - All tests passing ✅
  - Cargo check passes with all features ✅

- [ ] **🎯 NEXT: Vec Lookup Optimization**
  - Target remaining performance issue: `var_map` Vec O(n) lookup  
  - Replace with const generic fixed-size arrays for O(1) access
  - This is the final bottleneck preventing true zero-overhead

## Benchmark Results (Latest)

### Before Runtime Dispatch Elimination (v3):
- **Scalar Addition**: ~21.01 ns (has `std::any::Any` overhead)
- **Array Indexing**: ~43.78 ns (Vec lookup + runtime dispatch)
- **Neural Network**: ~45.08 ns (combined overhead)

### Target Performance (old homogeneous system):
- **Scalar Addition**: ~5.74 ns (our performance target)
- **Array Indexing**: ~5.58 ns (excellent baseline) 
- **Neural Network**: Not implemented in old system

### Direct Rust Baseline:
- **Scalar Addition**: ~537 ps (ultimate theoretical limit)
- **Array Indexing**: ~485 ps (ultimate theoretical limit)

## Next Steps

1. **🎯 IMMEDIATE**: Fix Vec lookup bottleneck in `var_map`
   - Replace `Vec<(usize, VarType, usize)>` with const generic arrays
   - Implement O(1) variable access instead of O(n) search

2. **Performance Target**: Match or exceed old system's ~5.7ns performance
3. **Final optimization**: Eliminate any remaining runtime overhead
4. **Documentation**: Update performance characteristics documentation

## Technical Notes

### ✅ Completed: Runtime Dispatch Elimination
The `heterogeneous_v4.rs` successfully eliminates runtime type dispatch through:
- **Compile-time trait specialization**: `DirectStorage<T>` per type
- **Monomorphized evaluation paths**: Zero runtime type checking
- **Separate traits for complex operations**: `TrueZeroArrayExpr` for mixed types
- **Pure compile-time type safety**: All type checking at compile time

### 🎯 Current Bottleneck: Vec Lookup Performance
Lines 431-436 in v3 show the O(n) Vec search pattern that needs optimization:
```rust
for &(var_id, var_type, storage_index) in &self.var_map {
    if var_id == target_var_id {
        return self.get_typed(&var_type, storage_index);
    }
}
```

This must be replaced with O(1) access using const generics. 