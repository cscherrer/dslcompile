# Variable Dependency Analysis Implementation Plan

Based on ideas from `toomuch.diff`, this document outlines a gradual implementation plan for improving variable dependency analysis in DSLCompile.

## Phase 1: Core Dependency Types (PRIORITY: HIGH)

### 1.1 Add VarSet Datatype to Egglog Grammar
```egglog
(datatype VarSet
  (EmptySet)                         ; No dependencies
  (SingleVar i64)                    ; Single variable dependency  
  (UnionSet VarSet VarSet)           ; Union of two dependency sets
)
```

### 1.2 Add Dependency Functions
```egglog
(function free-vars (Math) VarSet :merge (UnionSet old new))
(function bound-vars (Math) VarSet :merge (UnionSet old new))
(function contains-var (VarSet i64) Bool :merge (or old new))
(function is-independent-of (Math i64) Bool :merge (or old new))
```

**Benefits:**
- Enables compile-time dependency checking
- Prevents variable capture bugs
- Foundation for safe optimization rules

## Phase 2: Automatic Dependency Computation (PRIORITY: HIGH)

### 2.1 Basic Expression Dependencies
```egglog
; Constants have no dependencies
(set (free-vars (Num 0.0)) (EmptySet))
(set (free-vars (Num 1.0)) (EmptySet))

; Variables contribute their index
(set (free-vars (Variable 0)) (SingleVar 0))
(set (free-vars (Variable 1)) (SingleVar 1))

; BoundVars are not free (they're bound by lambda)
(set (free-vars (BoundVar 0)) (EmptySet))
(set (free-vars (BoundVar 1)) (EmptySet))
```

### 2.2 Compositional Dependency Rules  
```egglog
; Binary operations union dependencies
(rule () ((set (free-vars (Add ?a ?b)) (UnionSet (free-vars ?a) (free-vars ?b)))))
(rule () ((set (free-vars (Mul ?a ?b)) (UnionSet (free-vars ?a) (free-vars ?b)))))
(rule () ((set (free-vars (Div ?a ?b)) (UnionSet (free-vars ?a) (free-vars ?b)))))

; Unary operations inherit dependencies
(rule () ((set (free-vars (Neg ?a)) (free-vars ?a))))
(rule () ((set (free-vars (Ln ?a)) (free-vars ?a))))
```

**Benefits:**
- Automatic dependency tracking
- No manual dependency annotations needed
- Compositional - works for nested expressions

## Phase 3: Safe Rewrite Rules (PRIORITY: MEDIUM)

### 3.1 Independence Checking
```egglog
; A coefficient is independent if it doesn't contain the bound variable
(rule ((= ?coeff-deps (free-vars ?coeff))
       (= false (contains-var ?coeff-deps ?bound-var)))
      ((set (is-independent-of ?coeff ?bound-var) true)))
```

### 3.2 Safe Factorization Rules
```egglog
; Safe factorization - only if coefficient is independent  
(rule ((= lhs (LambdaFunc ?var (Mul ?coeff ?term)))
       (= true (is-independent-of ?coeff ?var)))
      ((union lhs (Factor ?coeff (LambdaFunc ?var ?term))))
      :ruleset factorization)
```

**Benefits:**
- Prevents variable capture in optimization
- Eliminates runtime dependency checks
- Safer than current unconditional rules

## Phase 4: Enhanced Cost Model (PRIORITY: LOW)

### 4.1 Dependency-Aware Costs
```egglog
; Prefer expressions with simpler dependency structure
(rule () ((set (cost (Factor ?coeff ?lambda)) (+ (cost ?coeff) (cost ?lambda) 5))))
```

**Benefits:**
- Extraction favors expressions with cleaner variable structure
- Better optimization decisions

## Implementation Strategy

### Step 1: Create New Egglog Rule File
Create `dslcompile/src/egglog_rules/dependency_aware_core.egg` with:
1. VarSet datatype
2. Basic dependency functions  
3. Simple test cases

### Step 2: Integrate with NativeEgglogOptimizer
1. Update `NativeEgglogOptimizer` to load new rules
2. Add dependency extraction methods
3. Create unit tests

### Step 3: Gradual Rule Migration
1. Start with simple expressions (Add, Mul, Constant)
2. Add lambda/summation support
3. Migrate complex optimization rules

### Step 4: Test and Validate
1. Create comprehensive test suite
2. Compare optimization results with old system
3. Performance benchmarking

## Risk Mitigation

### Compatibility
- Keep old rules as fallback during transition
- Feature flag for dependency-aware optimizations
- Gradual migration path

### Performance
- Benchmark dependency computation overhead
- Optimize hot path rules
- Consider caching strategies

### Correctness
- Extensive test coverage
- Property-based testing
- Cross-validation with manual analysis

## Expected Benefits

1. **Correctness**: Eliminates variable capture bugs
2. **Performance**: Removes runtime dependency checks
3. **Maintainability**: Clearer rule specifications
4. **Extensibility**: Foundation for advanced optimizations
5. **Debugging**: Better error messages for dependency issues

## Files to Create/Modify

### New Files:
- `dslcompile/src/egglog_rules/dependency_aware_core.egg`
- `dslcompile/tests/dependency_analysis_tests.rs`
- `dslcompile/examples/dependency_aware_demo.rs`

### Modified Files:
- `dslcompile/src/symbolic/native_egglog.rs`
- `dslcompile/src/egglog_rules/staged_core_math.egg`

This plan provides a structured approach to implementing the variable dependency analysis improvements while maintaining compatibility and reducing risk. 