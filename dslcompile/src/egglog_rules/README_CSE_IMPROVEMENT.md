# Improved Common Subexpression Elimination (CSE) Rules

## Overview

This document explains the improved **structural CSE approach** that replaces the previous hacky hardcoded binding ID system with a mathematically principled approach using egglog's built-in equality saturation.

## The Problem with the Old Approach

The previous `cse_rules.egg` used hardcoded binding IDs (1000, 1001, 1002, etc.) which had several serious issues:

1. **Collision Risk**: Hardcoded IDs could conflict with user variables or nested expressions
2. **Non-canonical**: Different discovery order created different Let bindings for the same expression  
3. **Limited Scope**: Only handled basic `expr op expr` patterns
4. **Not Structural**: Didn't identify semantically identical subexpressions in different forms
5. **Maintenance Burden**: Required manual ID management and collision avoidance

## The New Structural Approach

The improved approach leverages **egglog's native equality saturation** and **mathematical identities** instead of manual binding ID assignment.

### Key Principles

1. **Mathematical Identity-Based**: Uses proven algebraic transformations
2. **Bidirectional Rewrite Rules**: Automatically work in both directions  
3. **Canonical Forms**: Transforms expressions to mathematically equivalent canonical forms
4. **Collision-Free**: No manual ID management - relies on egglog's structural equality
5. **Extensible**: Easy to add new patterns without ID coordination

### Core Transformations

```egglog
; Canonical form: x * x → x^2 (more canonical for CSE analysis)
(rewrite (Mul ?x ?x) (Pow ?x (Num 2.0)) :ruleset cse_rules)

; Canonical form: x + x → 2*x (more canonical for CSE analysis)  
(rewrite (Add ?x ?x) (Mul (Num 2.0) ?x) :ruleset cse_rules)

; Factor common subexpressions in addition: a*c + b*c → (a + b)*c
(rewrite (Add (Mul ?a ?shared) (Mul ?b ?shared))
         (Mul (Add ?a ?b) ?shared)
         :ruleset cse_rules)

; Gaussian standardization pattern: ((x-μ)/σ)² 
(rewrite (Mul (Div (Add ?x (Neg ?mu)) ?sigma) (Div (Add ?x (Neg ?mu)) ?sigma))
         (Pow (Div (Add ?x (Neg ?mu)) ?sigma) (Num 2.0))
         :ruleset cse_rules)
```

## Performance Impact

### Before (Hacky Approach)
```
BEFORE: Mul(Div(Add(x,Neg(mu)),sigma), Div(Add(x,Neg(mu)),sigma))  
AFTER:  Let(1004, Div(Add(x,Neg(mu)),sigma), Mul(BoundVar(1004), BoundVar(1004)))
```
- ✅ Eliminates redundant computation
- ❌ Risk of variable collision with hardcoded ID 1004
- ❌ Non-canonical - different rules could use different IDs for same pattern

### After (Structural Approach)  
```
BEFORE: Mul(Div(Add(x,Neg(mu)),sigma), Div(Add(x,Neg(mu)),sigma))
AFTER:  Pow(Div(Add(x,Neg(mu)),sigma), Num(2.0))
```
- ✅ Eliminates redundant computation
- ✅ No collision risk - uses mathematical identities
- ✅ Canonical form - same pattern always produces same result
- ✅ Mathematically sound transformation

## Integration with Cost Model

The cost model in `core_datatypes.egg` automatically prefers canonical forms that reduce operation count:

- `Pow(x, 2.0)` is cheaper than `Mul(x, x)` when `x` is complex
- `Mul(shared, Add(a, b))` is cheaper than `Add(Mul(shared, a), Mul(shared, b))`
- Canonical forms expose more optimization opportunities in subsequent phases

## Testing and Verification

To verify the improved CSE is working:

1. **Expression Factoring**: Check that expressions with repeated subexpressions get factored
2. **Canonical Consistency**: Verify that canonical forms are consistently applied
3. **Operation Count Reduction**: Measure operation count reduction in extracted expressions  
4. **Gaussian Pattern**: Test specifically with probabilistic programming patterns like `((x-μ)/σ)²`
5. **Semantic Preservation**: Ensure all transformations preserve mathematical semantics

## Usage in Optimization Pipeline

The structural CSE rules integrate seamlessly with the existing staged optimization schedule:

```egglog
(run-schedule 
  (seq
    (saturate stage1_partitioning)
    (saturate stage2_constants) 
    (saturate cse_rules)          ; Apply structural CSE rules
    (saturate let_evaluation)     ; Clean up any remaining let bindings
    (saturate stage3_summation)
    (saturate stage4_simplify)
    (saturate let_evaluation)     ; Final cleanup
    (saturate stage2_constants)   ; Final constant folding
  ))
```

## Future Extensions

The structural approach makes it easy to add new CSE patterns:

```egglog
; Transcendental function CSE
(rewrite (Add (Sin ?x) (Mul ?c (Sin ?x)))
         (Mul (Add (Num 1.0) ?c) (Sin ?x))
         :ruleset cse_rules)

; Polynomial pattern CSE  
(rewrite (Add (Mul ?shared (Add ?c ?d)) (Mul ?shared (Add ?e ?f)))
         (Mul ?shared (Add (Add ?c ?d) (Add ?e ?f)))
         :ruleset cse_rules)
```

## Conclusion

The improved structural CSE approach provides:

1. **Mathematical Soundness**: Based on proven algebraic identities
2. **Collision Safety**: No manual ID management or collision risks
3. **Canonical Consistency**: Same patterns always produce same results
4. **Extensibility**: Easy to add new optimization patterns
5. **Integration**: Works seamlessly with existing optimization phases

This eliminates the maintenance burden and correctness risks of the previous hardcoded approach while providing the same or better optimization results. 