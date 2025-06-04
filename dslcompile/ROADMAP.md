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

**🎉 MILESTONE 1 COMPLETE!** All tests passing, neural network example demonstrating zero-overhead native types.

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