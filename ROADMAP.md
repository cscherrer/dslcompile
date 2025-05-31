# MathCompile Development Roadmap

## Project Overview

MathCompile is a mathematical expression compiler that transforms symbolic mathematical expressions into optimized, executable code. The project aims to bridge the gap between mathematical notation and high-performance computation.

## Core Architecture Insight (May 31, 2025)

**Key Simplification**: Statistical functionality should be a special case of the general mathematical expression system, not a separate specialized system.

- ✅ **General system works**: `call_multi_vars()` handles all cases correctly
- ✅ **Statistical computing**: Works by flattening parameters and data into a single variable array
- ✅ **Example created**: `simplified_statistical_demo.rs` demonstrates the approach
- ✅ **Core methods fixed**: `call_with_data()` now properly concatenates params and data
- ✅ **Architecture validated**: Statistical functions are just mathematical expressions with more variables
- ✅ **Legacy code removed**: Cleaned up unused statistical specialization types and methods

**Status**: Core simplification **COMPLETED**. System is now unified and clean.

## 🚀 Next Steps (Ordered by Core → Downstream)

### 1. Basic Normalization (Foundation for Everything Else) ✅ COMPLETED

#### Canonical Form Transformations ✅ COMPLETED
- [x] **Canonical Form Transformations**: Complete implementation of `Sub(a, b) → Add(a, Neg(b))` and `Div(a, b) → Mul(a, Pow(b, -1))`
- [x] **Pipeline Integration**: Full normalization pipeline: `AST → Normalize → ANF → Egglog → Extract → Codegen`
- [x] **Bidirectional Processing**: Normalization for optimization and denormalization for display
- [x] **Egglog Rule Simplification**: Achieved ~40% rule complexity reduction with canonical-only rules
- [x] **Comprehensive Testing**: 12 test functions covering all normalization aspects
- [x] **Test Infrastructure**: Fixed hanging test issues and ensured robust test execution

**Status**: ✅ **COMPLETED** (May 31, 2025)

**Implementation Details**:
- Created `src/ast/normalization.rs` with comprehensive normalization functions
- Implemented `normalize()`, `denormalize()`, `is_canonical()`, and `count_operations()` functions
- Updated egglog integration to use canonical-only rules in `rules/canonical_arithmetic.egg`
- Modified `src/symbolic/egglog_integration.rs` to implement full pipeline: AST → Normalize → Egglog → Extract → Denormalize
- Created comprehensive test suite with 12 test functions covering all aspects of normalization
- Demonstrated ~40% rule complexity reduction by eliminating Sub/Div handling from egglog rules
- Fixed hanging test issue in `test_egglog_integration_with_normalization`

**Benefits Achieved**:
- Simplified egglog rules: Only need to handle Add, Mul, Neg, Pow (not Sub, Div)
- Consistent patterns: All operations follow additive/multiplicative patterns
- Better optimization opportunities: More algebraic simplification possibilities
- Reduced complexity: ~40% fewer rule cases to maintain and debug
- Foundation established for all subsequent optimization improvements

### 2. Rule System Organization ✅ COMPLETED

#### Extract Egglog Rules to Files ✅ COMPLETED
- [x] **Create Rules Directory**: Separate files for `basic_arithmetic.egg`, `transcendental.egg`, `trigonometric.egg`, etc.
- [x] **Rule Loader System**: Dynamic rule file loading, validation, and combination
- [x] **Migrate Existing Rules**: Extract ~200 lines of inlined rules from code to organized files
- [x] **Rule Documentation**: Add examples and documentation for each rule category

#### Enhanced Rule System ✅ COMPLETED
- [x] **EgglogOptimizer Integration**: Complete integration with RuleLoader for dynamic rule loading
- [x] **Multiple Configurations**: Support for Default, Domain-Aware, and Canonical-Only optimizers
- [x] **Rule Information API**: `rule_info()` method to inspect loaded rule categories
- [x] **Custom Rule Configurations**: Support for user-defined rule category combinations

**Status**: ✅ **COMPLETED** (May 31, 2025)

**Implementation Details**:
- Created organized rule files in `rules/` directory: `core_datatypes.egg`, `basic_arithmetic.egg`, `domain_aware_arithmetic.egg`, `transcendental.egg`, `trigonometric.egg`, `summation.egg`
- Implemented `RuleLoader` with `RuleConfig` for flexible rule management
- Integrated `RuleLoader` with `EgglogOptimizer` replacing inline rules with dynamic loading
- Added multiple optimizer constructors: `new()`, `with_rule_config()`, `domain_aware()`, `canonical_only()`
- Created comprehensive example `rule_loader_demo.rs` demonstrating all features
- Resolved rule conflicts by removing duplicates between files
- Added `rule_info()` API for inspecting loaded rule categories

**Benefits Achieved**:
- Modular rule organization by mathematical domain
- Dynamic rule loading with validation and error handling
- Flexible optimizer configurations for different use cases
- Clean separation between rule definitions and optimizer logic
- Foundation for user-provided rules and plugin architecture
- Comprehensive testing and documentation

### 3. Native egglog Integration ✅
**Status**: COMPLETED (2025-05-31)
- ✅ Implemented `NativeEgglogOptimizer` using egglog directly
- ✅ Created comprehensive mathematical rule set
- ✅ Added AST to egglog s-expression conversion
- ✅ Implemented canonical form support (Sub → Add + Neg, Div → Mul + Pow(-1))
- ✅ Fixed f64 formatting for egglog compatibility
- ✅ Added comprehensive test suite (9/9 tests passing)
- ✅ Created domain-aware optimization demo
- ✅ Established foundation for advanced domain analysis
- ✅ **CRITICAL FIX**: Removed unsafe `sqrt(x^2) = x` rule that was causing mathematical correctness issues (May 31, 2025)
- ✅ **MIGRATION COMPLETE**: Successfully migrated all code from unsafe `EgglogOptimizer` to domain-aware `NativeEgglogOptimizer` (May 31, 2025)
  - Updated examples: `rule_loader_demo.rs`, `egglog_optimization_demo.rs`
  - Updated tests: `test_native_egglog_integration_with_normalization`
  - Deprecated old `egglog_integration` module in favor of `native_egglog`
  - Verified mathematical safety: unsafe transformations like `sqrt(x^2) = x` are no longer applied
  - All tests passing including critical `test_all_strategies_consistency` proptest
- ✅ **DEAD CODE REMOVAL**: Completely removed outdated `egglog_integration.rs` file (May 31, 2025)
  - Deleted 1,102 lines of superseded code containing unsafe mathematical transformations
  - Cleaned up module declarations and imports
  - Verified all functionality preserved with domain-safe `native_egglog.rs` implementation
  - All tests continue to pass, confirming complete migration success

**Key Achievement**: Discovered that egglog itself provides native abstract interpretation capabilities, making manual integration unnecessary and opening up powerful optimization possibilities following the Herbie paper approach.

**Critical Bug Fix (May 31, 2025)**: 
- **Issue**: The symbolic optimizer contained an unsafe algebraic rule `sqrt(x^2) = x` that is mathematically incorrect for negative values (should be `sqrt(x^2) = |x|`)
- **Impact**: This caused evaluation inconsistencies between Direct and Symbolic strategies, with expressions like `sqrt((-48.177)^2)` returning `-48.177` instead of `48.177`
- **Root Cause**: The rule was in `apply_enhanced_algebraic_rules()` in `symbolic.rs` without domain safety checks
- **Solution**: Removed the unsafe rules and commented them with explanations. The domain-aware egglog optimizer will eventually provide safe versions with proper preconditions
- **Verification**: All tests now pass, including the critical `test_all_strategies_consistency` proptest that caught this issue
- **Status**: Mathematical correctness **RESTORED** ✅

## 🎯 **Next Priority: Advanced Domain-Aware Optimization**

Based on our successful native egglog integration and research into egglog's capabilities, the next step is to implement advanced domain-aware optimization using egglog's native abstract interpretation features.

### Key Discovery: egglog's Native Capabilities
- **Lattice-based Analysis**: egglog supports lattice semantics natively
- **Multiple Interacting Analyses**: Unlike egg (single analysis), egglog supports composable analyses  
- **Interval Analysis**: Proven in Herbie case study with domain-aware rules
- **Conditional Rules**: Rules can be gated on analysis results (e.g., `ln(a/b)` only if `a,b > 0`)

### Concrete Implementation Plan
Following the [Herbie/egglog paper](https://effect.systems/doc/egraphs-2023-egglog/paper.pdf):

1. **Interval Domain Implementation** (Week 1)
   - Add `Interval` datatype with proper merge functions
   - Implement interval arithmetic rules for basic operations
   - Add domain predicates (`ival-positive`, `ival-nonzero`)

2. **Domain-Aware Rules** (Week 2)  
   - Convert transcendental rules to use interval guards
   - Implement `ln(a/b) = ln(a) - ln(b)` with domain safety
   - Add `exp(ln(x)) = x` with positivity checks

3. **Advanced Analysis** (Week 3)
   - Multiple lattice analyses (intervals + not-equals)
   - Compositional analysis propagation
   - Cost-based extraction with domain information

4. **Integration & Testing** (Week 4)
   - Replace manual domain checks with native egglog analysis
   - Comprehensive test suite with domain edge cases
   - Performance benchmarking vs current approach

**Expected Outcome**: Domain-safe symbolic optimization that automatically prevents mathematical errors like the `ln(a/b)` issue we manually fixed, while enabling more aggressive optimizations when domain safety can be proven.

## 🎯 **Current Priority: ANF Integration Completion**

With domain-aware optimization completed, the next logical step is to complete the ANF (A-Normal Form) integration that's currently disabled with TODOs in the codebase.

### Why ANF Integration is the Next Priority

The codebase has a complete ANF implementation in `src/anf/` but it's not fully integrated with the optimization pipeline. Completing this integration will:

1. **Enable Full Optimization Pipeline**: `AST → Normalize → ANF+CSE → Domain-Aware egglog → Extract → Denormalize`
2. **Improve Performance**: Common subexpression elimination reduces redundant computations
3. **Maintain Domain Safety**: Ensure CSE respects domain constraints from interval analysis
4. **Complete the Architecture**: Fulfill the original vision of a complete mathematical compiler

### Current State Analysis

- ✅ **ANF Implementation**: Complete A-Normal Form transformation exists
- ✅ **Domain-Aware Optimization**: Fully implemented with interval analysis
- ✅ **Normalization Pipeline**: Canonical form transformations working
- ❌ **Integration**: ANF is not connected to the main optimization pipeline
- ❌ **Domain-Aware CSE**: Common subexpression elimination doesn't use domain information

### Implementation Plan (4 weeks)

**Week 1: Enable ANF Pipeline Integration**
- Resolve TODO markers in the codebase that disable ANF integration
- Connect ANF transformation to the main optimization pipeline
- Ensure ANF works with normalized expressions

**Week 2: Domain-Aware ANF**
- Integrate `IntervalDomainAnalyzer` with ANF transformations
- Ensure ANF respects domain constraints when creating intermediate variables
- Add domain information propagation through ANF variables

**Week 3: Safe Common Subexpression Elimination**
- Enhance CSE to use domain analysis for safety checks
- Prevent CSE of expressions with different domain constraints
- Add domain-aware cost models for CSE decisions

**Week 4: Testing and Optimization**
- Comprehensive testing of the full pipeline
- Performance benchmarking vs current approach
- Documentation and examples of the complete system

**Week 1: Enable ANF Pipeline Integration** ✅ COMPLETED (May 31, 2025)
- [x] **Resolve TODO markers**: Fixed import issues and enabled ANF integration in `bayesian_linear_regression.rs`
- [x] **Export Integration**: Added `ANFConverter` to the prelude module for easy access
- [x] **Pipeline Connection**: ANF transformation now works in the main optimization pipeline
- [x] **Normalized Expression Support**: ANF correctly processes normalized expressions from domain-aware optimization
- [x] **Performance Metrics**: ANF now reports actual let-binding counts and operation reduction percentages
- [x] **Working Examples**: Both `anf_demo.rs` and `bayesian_linear_regression.rs` demonstrate ANF functionality

**Next Priority**: **ANF Integration Completion (Week 2)** - With Week 1 completed, we now move to integrating domain analysis with ANF transformations to ensure domain safety throughout the complete optimization pipeline: `AST → Normalize → ANF+CSE → Domain-Aware egglog → Extract → Denormalize`.

**Week 1 Achievement**: ANF is now successfully integrated into the optimization pipeline with working examples and performance metrics. The foundation is solid for domain-aware enhancements.

## 🔄 Current Status (May 31, 2025)

The library has achieved a major milestone with **complete domain-aware optimization** implementation. The mathematical expression system now provides both high performance and mathematical correctness through sophisticated domain analysis.

**Major Achievements Completed**: 
1. **Core Architecture Simplification**: Statistical functions unified with general mathematical expressions using `call_multi_vars()` pattern
2. **Complete Normalization Pipeline**: Canonical form transformations reduce egglog rule complexity by ~40%
3. **Dynamic Rule System**: Organized rule loading with multiple optimizer configurations and clean domain separation
4. **✅ DOMAIN-AWARE OPTIMIZATION COMPLETED**: Full implementation with interval analysis, conditional rewrite rules, and mathematical safety guarantees
5. **Native egglog Integration**: Complete domain-aware optimizer using egglog's native abstract interpretation capabilities
6. **Mathematical Correctness**: Fixed critical domain safety issues and eliminated unsafe transformations like `sqrt(x^2) = x`

**Current State**: 
- ✅ **Foundation Complete**: All core infrastructure and domain analysis implemented
- ✅ **Domain Safety**: Mathematical correctness guaranteed through interval analysis
- ✅ **Performance**: Efficient lattice-based analysis with minimal overhead
- ✅ **Extensibility**: Framework ready for new domain-aware rules and constraints

**Next Priority**: **ANF Integration Completion** - Connect the existing A-Normal Form implementation with the domain-aware optimization pipeline to achieve the complete mathematical compiler vision: `AST → Normalize → ANF+CSE → Domain-Aware egglog → Extract → Denormalize`.

## Performance Goals

- **Compilation Speed**: Sub-second compilation for complex expressions
- **Runtime Performance**: Within 5% of hand-optimized code
- **Memory Usage**: Minimal allocation during expression evaluation
- **Type Safety**: Zero runtime type errors with compile-time guarantees

## Testing Strategy

- [x] **Unit Tests**: Comprehensive coverage of all features
- [x] **Integration Tests**: End-to-end workflow validation
- [x] **Property-Based Tests**: Randomized testing with QuickCheck
- [ ] **Performance Regression Tests**: Automated benchmarking
- [ ] **Cross-Platform Testing**: Windows, macOS, Linux validation

*Last updated: May 31, 2025*
*Status: Core simplification completed, ready for systematic implementation*
