# Variable Composition in Mathematical Expression Systems

## The Problem: Shared Variable Composition

When building mathematical expressions independently and then composing them, we encounter the **variable collision problem**. Consider:

```rust
// Define f(x,y) = x² + xy + y² independently
let f = build_function(); // uses variables [0, 1] for [x, y]

// Define g(y,z) = 2y + 3z independently  
let g = build_function(); // uses variables [0, 1] for [y, z]

// Want: h(x,y,z) = f(x,y) + g(y,z)
// Problem: Both f and g use variable index 0, but they mean different things!
```

This is a fundamental issue in **compositional semantics** for mathematical expression systems.

## Programming Language Theory Perspective

### 1. **Variable Scoping and Binding**

This problem is analogous to **variable capture** in lambda calculus and programming languages:

```haskell
-- Lambda calculus example
let f = λx.λy. x + y     -- f uses variables x, y
let g = λy.λz. y * z     -- g uses variables y, z (y is shared!)
let h = λx.λy.λz. f x y + g y z  -- Correct composition
```

In our system, we need to implement **α-conversion** (alpha conversion) - the systematic renaming of bound variables to avoid capture.

### 2. **De Bruijn Indices and Variable Management**

Our current system uses **De Bruijn indices** (variables as numbers), which is common in:
- Lambda calculus implementations
- Functional programming language compilers
- Theorem provers (Coq, Lean, etc.)

The challenge is that De Bruijn indices are **context-dependent**:
```
f(x,y) internally: λ.λ. var[1] + var[0]  // y + x (De Bruijn style)
g(y,z) internally: λ.λ. var[1] * var[0]  // z * y (De Bruijn style)
```

### 3. **Nominal vs. Structural Approaches**

**Structural Approach** (current): Variables are just indices
- Fast and memory-efficient
- Requires explicit remapping for composition
- Used in: most compilers, mathematical software

**Nominal Approach**: Variables carry semantic names
- More intuitive for composition
- Higher memory overhead
- Used in: symbolic math systems (Mathematica, SymPy)

## Practical Solutions

### Solution 1: Manual Variable Remapping

```rust
// Manual approach - explicit control
let mut g_var_map = HashMap::new();
g_var_map.insert(0, 1); // g's y (index 0) -> global y (index 1)
g_var_map.insert(1, 2); // g's z (index 1) -> global z (index 2)
let g_remapped = remap_variables(&g_ast, &g_var_map);

let h = f_ast + g_remapped; // Now safe to compose
```

### Solution 2: Automatic Variable Analysis

```rust
// Systematic approach with name tracking
struct NamedFunction {
    ast: ASTRepr<f64>,
    var_names: Vec<String>, // Semantic variable names
}

fn compose_functions(functions: &[NamedFunction]) -> ASTRepr<f64> {
    // 1. Collect all unique variable names
    // 2. Create global variable ordering
    // 3. Remap each function to global indices
    // 4. Compose safely
}
```

### Solution 3: Type-Level Variable Management (Compile-Time)

```rust
// Compile-time approach using Rust's type system
let f = var::<0>().mul(var::<1>());  // f(x,y) with explicit type-level indices
let g = var::<1>().mul(var::<2>());  // g(y,z) with non-colliding indices
let h = f.add(g);                    // Safe composition at compile time
```

## Advanced Approaches

### 1. **Hindley-Milner Style Type Inference**

We could implement a type system that automatically infers variable relationships:

```rust
// Hypothetical advanced API
let f = expr! { |x, y| x*x + x*y + y*y };
let g = expr! { |y, z| 2*y + 3*z };
let h = compose! { |x, y, z| f(x, y) + g(y, z) }; // Automatic variable unification
```

### 2. **Category Theory Approach**

Mathematical expressions form a **category** where:
- Objects: Variable contexts (e.g., `[x, y]`, `[y, z]`)
- Morphisms: Functions between contexts
- Composition: Pullback along shared variables

```rust
// Category-theoretic composition
let f: Context([x, y]) -> f64 = ...;
let g: Context([y, z]) -> f64 = ...;
let h: Context([x, y, z]) -> f64 = compose_pullback(f, g);
```

### 3. **Dependent Types for Variable Safety**

Using dependent types (like in Idris or Agda), we could make variable composition statically safe:

```idris
-- Hypothetical Idris-style API
data Vars : List String -> Type
f : Vars ["x", "y"] -> f64
g : Vars ["y", "z"] -> f64
h : Vars ["x", "y", "z"] -> f64  -- Automatically inferred union type
```

## Implementation in DSLCompile

### Current Implementation

We've implemented **Solution 1** and **Solution 2**:

1. **`remap_variables`**: Core remapping function
2. **`combine_expressions_with_remapping`**: Automatic non-overlapping assignment
3. **`NamedFunction`**: Semantic variable tracking
4. **Compile-time safety**: Type-level variable indices

### Usage Examples

```rust
// Simple case: h(x,y) = f(x) + g(y)
let (remapped, _) = combine_expressions_with_remapping(&[f_ast, g_ast]);
let h = remapped[0] + remapped[1];

// Complex case: h(x,y,z) = f(x,y) + g(y,z)
let f_named = NamedFunction::new(f_ast, vec!["x", "y"]);
let g_named = NamedFunction::new(g_ast, vec!["y", "z"]);
let h = compose_with_shared_variables(&[f_named, g_named]);
```

## Theoretical Foundations

### 1. **Substitution and α-Conversion**

Our variable remapping implements **α-conversion** from lambda calculus:
```
α-conversion: λx.M ≡ λy.M[x := y]  (if y not free in M)
```

In our context:
```rust
remap_variables(expr, {0 -> 2}) ≡ α-convert variable 0 to variable 2
```

### 2. **Compositional Semantics**

We maintain **compositionality**: the meaning of a composite expression depends only on the meanings of its parts and their combination rule.

```
⟦f + g⟧(env) = ⟦f⟧(env) + ⟦g⟧(env)
```

Where `env` is a properly constructed environment that maps all variables correctly.

### 3. **Variable Context Management**

Our approach implements **context extension** and **context morphisms**:

```
Γ ⊢ f : τ    Δ ⊢ g : τ    Γ ∪ Δ well-formed
─────────────────────────────────────────
Γ ∪ Δ ⊢ f + g : τ
```

## Future Directions

### 1. **Automatic Variable Inference**

Implement a constraint solver that automatically determines optimal variable layouts:

```rust
let h = auto_compose! {
    f(x, y) = x*x + y*y,
    g(y, z) = y + z,
    result = f + g  // Automatically infers h(x,y,z) = f(x,y) + g(y,z)
};
```

### 2. **Linear Types for Variable Uniqueness**

Use Rust's affine types to ensure variables are used correctly:

```rust
struct UniqueVar<const ID: usize>(PhantomData<()>);
// Variables can only be used once, preventing accidental reuse
```

### 3. **Effect System for Variable Dependencies**

Track variable dependencies in the type system:

```rust
fn compose<F, G, Vars>(f: F, g: G) -> impl Fn(Vars) -> f64
where
    F: Fn(Vars::Subset1) -> f64,
    G: Fn(Vars::Subset2) -> f64,
    Vars: Union<Vars::Subset1, Vars::Subset2>,
```

## Conclusion

The variable composition problem in mathematical expression systems is a rich area that connects:

- **Programming Language Theory**: Variable binding, scoping, α-conversion
- **Type Theory**: Dependent types, linear types, effect systems  
- **Category Theory**: Pullbacks, context morphisms, compositional semantics
- **Practical Implementation**: Performance, usability, safety

Our implementation in DSLCompile provides both manual control and automatic inference, supporting both runtime flexibility and compile-time safety. The theoretical foundations ensure correctness while the practical implementation focuses on performance and usability.

This approach scales to complex mathematical software where function composition is a core operation, providing a solid foundation for symbolic mathematics, automatic differentiation, and code generation systems. 