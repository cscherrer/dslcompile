# Operation Structure Design for MathCompile

## The Challenge: Different Mathematical Structures Need Different Rules

When designing a mathematical expression system, a fundamental question arises: **How do we decide which operations to include as primitives vs. expressing them in terms of others?**

For example:
- Should we have `Sub` or just express it as `Add` + `Neg`?
- Should we have `Div` or just express it as `Mul` + `Pow(-1)`?
- What about matrix operations where `A * B ≠ B * A`?

## Our Solution: Context-Aware Rule System

### 1. **Hybrid Approach with Strategic Normalization**

We maintain **both surface operations and canonical forms**:

```rust
// Surface operations (for user convenience and specific contexts)
ASTRepr::Sub(a, b)     // Subtraction
ASTRepr::Div(a, b)     // Division
ASTRepr::MatrixMul(a, b)  // Matrix multiplication (non-commutative!)

// Canonical forms (for optimization)
// Sub(a, b) → Add(a, Neg(b))  [via egglog rules]
// Div(a, b) → Mul(a, Pow(b, -1))  [for algebraic manipulation]
```

### 2. **Context-Specific Function Categories**

Different mathematical domains have different algebraic properties:

#### **Scalar Arithmetic** (Commutative)
```rust
// These operations follow standard commutative rules
Add(a, b) = Add(b, a)     // Commutative
Mul(a, b) = Mul(b, a)     // Commutative
Sub(a, b) → Add(a, Neg(b)) // Normalize for optimization
```

#### **Linear Algebra** (Non-Commutative)
```rust
// Matrix operations have different rules!
MatrixMul(A, B) ≠ MatrixMul(B, A)  // NOT commutative
MatrixAdd(A, B) = MatrixAdd(B, A)  // Still commutative
LeftDivide(A, B) = A⁻¹ * B         // A \ B
RightDivide(A, B) = A * B⁻¹        // A / B (different!)
```

#### **Quaternions** (Non-Commutative)
```rust
// Quaternion multiplication is also non-commutative
QuatMul(q1, q2) ≠ QuatMul(q2, q1)
```

### 3. **Rule File Organization by Context**

```
egglog_rules/
├── core_arithmetic.egg      # Scalar operations (commutative)
├── linear_algebra.egg       # Matrix/vector ops (mixed commutativity)
├── trigonometric.egg        # Trig identities
├── logarithmic.egg          # Log/exp rules
├── complex_numbers.egg      # Complex arithmetic
├── quaternions.egg          # Quaternion algebra
└── differential.egg         # Calculus rules
```

## Key Design Principles

### 1. **Preserve Surface Operations for Context**

**Why keep both `Sub` and `Add + Neg`?**

- **Numerical Stability**: `a - b` vs `a + (-b)` can have different floating-point behavior
- **Code Generation**: Direct subtraction is often more efficient
- **User Intent**: Preserves the original mathematical expression structure
- **Domain-Specific Rules**: Some contexts have special subtraction rules

### 2. **Strategic Normalization**

**When to normalize:**
```rust
// Normalize for algebraic manipulation
Sub(a, b) → Add(a, Neg(b))  // Enables simpler associativity rules
Div(a, b) → Mul(a, Pow(b, -1))  // For symbolic algebra

// But preserve for numerical computation
Sub(a, b) → Sub(a, b)  // Keep for floating-point stability
```

### 3. **Context-Aware Rule Application**

**Different contexts, different rules:**

```rust
// Scalar context: commutative
Mul(a, b) → Mul(b, a)  ✓

// Matrix context: non-commutative  
MatrixMul(A, B) → MatrixMul(B, A)  ✗ WRONG!

// But matrix addition is still commutative
MatrixAdd(A, B) → MatrixAdd(B, A)  ✓
```

## Implementation Strategy

### 1. **Function Categories with Trait System**

```rust
trait FunctionCategory<T> {
    fn to_egglog(&self) -> String;
    fn apply_local_rules(&self, expr: &ASTRepr<T>) -> Option<ASTRepr<T>>;
}

// Each category knows its own rules
impl FunctionCategory<T> for LinearAlgebraCategory<T> {
    fn apply_local_rules(&self, expr: &ASTRepr<T>) -> Option<ASTRepr<T>> {
        match expr {
            // Matrix multiplication: NO commutativity
            MatrixMul(A, B) => {
                // Apply matrix-specific rules only
                if is_identity_matrix(B) { Some(A.clone()) }
                else if is_zero_matrix(A) { Some(zero_matrix()) }
                else { None }
            }
        }
    }
}
```

### 2. **Separate Rule Files by Domain**

**Linear Algebra Rules** (`linear_algebra.egg`):
```lisp
;; Matrix multiplication is NOT commutative
;; A * I = A (right identity)
(rewrite (LinAlg (MatMulFunc ?A (Identity))) ?A)
;; I * A = A (left identity)  
(rewrite (LinAlg (MatMulFunc (Identity) ?A)) ?A)

;; But matrix addition IS commutative
(rewrite (LinAlg (MatAddFunc ?A ?B)) (LinAlg (MatAddFunc ?B ?A)))

;; Left vs Right division are different!
;; A \ B = A^(-1) * B
(rewrite (LinAlg (LeftDivFunc ?A ?B))
         (LinAlg (MatMulFunc (LinAlg (InvFunc ?A)) ?B)))
;; A / B = A * B^(-1)  
(rewrite (LinAlg (RightDivFunc ?A ?B))
         (LinAlg (MatMulFunc ?A (LinAlg (InvFunc ?B)))))
```

### 3. **Extensible Architecture**

**Adding New Mathematical Domains:**

1. **Create new function category**:
   ```rust
   struct QuaternionCategory<T> { ... }
   impl FunctionCategory<T> for QuaternionCategory<T> { ... }
   ```

2. **Add corresponding rule file**:
   ```lisp
   ;; quaternions.egg
   ;; Quaternion multiplication is non-commutative
   ;; But has different rules than matrices
   ```

3. **Register with rule loader**:
   ```rust
   loader.load_rule_file("quaternions", "Quaternion algebra", 200)?;
   ```

## Benefits of This Approach

### 1. **Correctness**
- **No incorrect optimizations**: Matrix `A*B` never becomes `B*A`
- **Context-appropriate rules**: Each domain gets its proper algebraic laws
- **Preserves mathematical meaning**: User intent is maintained

### 2. **Performance**
- **Targeted optimization**: Rules only apply where they're valid
- **Efficient rule application**: Context-specific rule sets are smaller
- **Better code generation**: Can choose optimal representation per context

### 3. **Extensibility**
- **Easy to add new domains**: Just implement the trait and add rule files
- **Composable**: Different mathematical structures can coexist
- **Maintainable**: Rules are organized by mathematical domain

### 4. **User Experience**
- **Natural syntax**: Users can write `A \ B` for left division
- **Correct semantics**: Operations behave as mathematically expected
- **Rich functionality**: Support for diverse mathematical domains

## Examples

### Matrix Operations
```rust
// These are all different operations!
let left_div = LinearAlgebraCategory::left_divide(A, B);   // A \ B = A⁻¹ * B
let right_div = LinearAlgebraCategory::right_divide(A, B); // A / B = A * B⁻¹
let matrix_mul = LinearAlgebraCategory::matrix_mul(A, B);  // A * B ≠ B * A

// But matrix addition is commutative
let mat_add = LinearAlgebraCategory::matrix_add(A, B);    // A + B = B + A ✓
```

### Cross Product (Anti-Commutative)
```rust
// Cross product: a × b = -(b × a)
let cross = LinearAlgebraCategory::cross_product(a, b);
// Egglog rule automatically handles: a × b → -(b × a)
```

### Scalar vs Matrix Context
```rust
// Scalar multiplication: commutative
let scalar_mul = a * b;  // = b * a

// Matrix multiplication: non-commutative  
let matrix_mul = A.matrix_mul(B);  // ≠ B.matrix_mul(A)
```

## 5. **Feature Flags for Optional Functionality**

### Linear Algebra Feature Flag

Complex mathematical domains like linear algebra can add significant compilation time and binary size. We use feature flags to make them optional:

```toml
[features]
default = ["optimization"]
optimization = ["egglog"]
linear_algebra = []  # Enable matrix operations and linear algebra
full = ["optimization", "linear_algebra"]  # All features
```

**Benefits:**
- **Reduced Binary Size**: Users who don't need matrices don't pay for them
- **Faster Compilation**: Conditional compilation reduces build times
- **Modular Dependencies**: Linear algebra might require additional crates
- **Clear Separation**: Makes it obvious which features are optional

**Usage:**
```rust
// Only available with linear_algebra feature
#[cfg(feature = "linear_algebra")]
let matrix_expr = ASTRepr::matrix_mul(a, b);

// Always available
let scalar_expr = ASTRepr::add(x, y);
```

### Future Feature Flags

Other domains that could benefit from feature flags:
- `statistics` - Statistical functions and distributions
- `quantum` - Quantum computing operations
- `symbolic_integration` - Advanced symbolic integration
- `numerical_methods` - Numerical solvers and approximations

**Special Functions Feature Flag**

Advanced mathematical functions like gamma, beta, Bessel functions, and Lambert W functions are provided through integration with the [`special` crate](https://docs.rs/special/latest/special/):

```toml
[features]
special_functions = ["special"]  # Enable special mathematical functions
```

**Available Functions:**
- **Gamma Functions**: `Γ(x)`, `log Γ(x)`
- **Beta Functions**: `B(a,b)`, `log B(a,b)`
- **Error Functions**: `erf(x)`, `erfc(x)`, `erf⁻¹(x)`, `erfc⁻¹(x)`
- **Bessel Functions**: `J₀(x)`, `J₁(x)`, `Jₙ(x)`, `Y₀(x)`, `Y₁(x)`, `Yₙ(x)`, `I₀(x)`, `I₁(x)`, `Iₙ(x)`, `K₀(x)`, `K₁(x)`, `Kₙ(x)`
- **Lambert W Function**: `W₀(x)`, `W₋₁(x)`

**Benefits:**
- **High-Quality Implementation**: Uses the well-tested `special` crate
- **Type Support**: Works with `f64` and `f32` through the special crate
- **Mathematical Identities**: Includes proper mathematical relationships and simplifications
- **Optional Dependency**: Only included when needed

**Usage:**
```rust
#[cfg(feature = "special_functions")]
use mathcompile::ast::function_categories::SpecialCategory;

#[cfg(feature = "special_functions")]
let gamma_expr = ASTRepr::Function(Box::new(SpecialCategory::gamma(x)));
```

## 6. **Implementation Strategy**

## Conclusion

This **context-aware rule system** solves the fundamental tension between:
- **Generality** (supporting diverse mathematical structures)
- **Correctness** (respecting algebraic properties)  
- **Performance** (enabling targeted optimizations)
- **Usability** (providing natural mathematical syntax)

By organizing operations into **function categories** with **domain-specific rules**, we can support everything from basic arithmetic to advanced linear algebra while maintaining mathematical correctness and enabling powerful optimizations. 