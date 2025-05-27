# MathJIT: High-Performance Symbolic Mathematics

MathJIT is a high-performance symbolic mathematics library built around the **final tagless** approach, providing zero-cost abstractions for mathematical expressions with multiple evaluation strategies.

## 🚀 Key Features

- **Final Tagless Architecture**: Zero-cost abstractions using Generic Associated Types (GATs)
- **Multiple Interpreters**: Same expression definition, multiple evaluation strategies
- **Type Safety**: Compile-time guarantees without runtime overhead
- **Extensible Design**: Easy to add new operations and interpreters
- **High Performance**: Direct evaluation with native Rust operations

## 🎯 Design Goals

1. **Entirely Final Tagless**: Clean separation between expression definition and interpretation
2. **Egglog Optimization**: Symbolic optimization capabilities (coming soon)
3. **Cranelift JIT**: Native code compilation for maximum performance (coming soon)
4. **General Purpose**: Fast and flexible for evaluating mathematical expressions

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mathjit = "0.1.0"
```

## 🔧 Quick Start

```rust
use mathjit::final_tagless::{MathExpr, DirectEval, PrettyPrint};

// Define a polymorphic mathematical expression
fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64>
where
    E::Repr<f64>: Clone,
{
    let a = E::constant(2.0);
    let b = E::constant(3.0);
    let c = E::constant(1.0);
    
    E::add(
        E::add(
            E::mul(a, E::pow(x.clone(), E::constant(2.0))),
            E::mul(b, x)
        ),
        c
    )
}

fn main() {
    // Direct evaluation
    let result = quadratic::<DirectEval>(DirectEval::var("x", 2.0));
    println!("quadratic(2) = {}", result); // 15.0
    
    // Pretty printing
    let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
    println!("Expression: {}", pretty); // (((2 * (x ^ 2)) + (3 * x)) + 1)
}
```

## 🏗️ Architecture

### Core Traits

- **`MathExpr`**: Defines basic mathematical operations (arithmetic, transcendental functions)
- **`StatisticalExpr`**: Extends `MathExpr` with statistical functions (logistic, softplus)
- **`NumericType`**: Helper trait bundling common numeric type requirements

### Interpreters

- **`DirectEval`**: Immediate evaluation using native Rust operations (`type Repr<T> = T`)
- **`PrettyPrint`**: String representation generation (`type Repr<T> = String`)
- **`JITEval`**: Native code compilation via Cranelift IR (coming soon)

## 🔬 Advanced Usage

### Polynomial Evaluation with Horner's Method

MathJIT provides efficient polynomial evaluation using Horner's method, which reduces the number of multiplications from O(n²) to O(n) and provides better numerical stability:

```rust
use mathjit::final_tagless::{DirectEval, PrettyPrint, polynomial};

// Evaluate 5 + 4x + 3x² + 2x³ using Horner's method
let coeffs = [5.0, 4.0, 3.0, 2.0]; // [constant, x, x², x³]
let result = polynomial::horner::<DirectEval, f64>(&coeffs, DirectEval::var("x", 2.0));
println!("Polynomial at x=2: {}", result); // 41

// Pretty print the Horner structure
let pretty = polynomial::horner::<PrettyPrint, f64>(&coeffs, PrettyPrint::var("x"));
println!("Horner form: {}", pretty); // ((((((2 * x) + 3) * x) + 4) * x) + 5)

// Create polynomial from roots: (x-1)(x-2)(x-3)
let roots = [1.0, 2.0, 3.0];
let poly = polynomial::from_roots::<DirectEval, f64>(&roots, DirectEval::var("x", 0.0));
println!("Polynomial at x=0: {}", poly); // -6

// Evaluate polynomial derivative
let derivative = polynomial::horner_derivative::<DirectEval, f64>(&coeffs, DirectEval::var("x", 2.0));
println!("Derivative at x=2: {}", derivative); // 40
```

### Statistical Functions

```rust
use mathjit::final_tagless::{StatisticalExpr, DirectEval};

fn logistic_regression<E: StatisticalExpr>(x: E::Repr<f64>, theta: E::Repr<f64>) -> E::Repr<f64> {
    E::logistic(E::mul(theta, x))
}

let result = logistic_regression::<DirectEval>(
    DirectEval::var("x", 2.0),
    DirectEval::var("theta", 1.5)
);
println!("logistic_regression(2, 1.5) = {}", result);
```

### Extension Example

Adding new operations is straightforward with trait extension:

```rust
use mathjit::final_tagless::*;
use num_traits::Float;

// Extend with hyperbolic functions
trait HyperbolicExpr: MathExpr {
    fn tanh<T: NumericType + Float>(x: Self::Repr<T>) -> Self::Repr<T>
    where
        Self::Repr<T>: Clone,
    {
        let exp_x = Self::exp(x.clone());
        let exp_neg_x = Self::exp(Self::neg(x));
        let numerator = Self::sub(exp_x.clone(), exp_neg_x.clone());
        let denominator = Self::add(exp_x, exp_neg_x);
        Self::div(numerator, denominator)
    }
}

// Automatically works with all existing interpreters
impl HyperbolicExpr for DirectEval {}
impl HyperbolicExpr for PrettyPrint {}
```

## 🚧 Roadmap

- [ ] **JIT Compilation**: Cranelift-based native code generation
- [ ] **Egglog Optimization**: Symbolic expression optimization
- [ ] **Automatic Differentiation**: Support for dual numbers and gradients
- [ ] **SIMD Support**: Vectorized operations for batch evaluation
- [ ] **GPU Acceleration**: CUDA/OpenCL backends for parallel computation

## 🔍 Technical Details

### Final Tagless Approach

The final tagless approach solves the expression problem by:

1. **Parameterizing representation types**: Operations are defined over abstract representation types `Repr<T>`
2. **Trait-based extensibility**: New operations can be added via trait extension
3. **Zero intermediate representation**: Expressions compile directly to target representations

### Performance

- **Zero-cost abstractions**: No runtime overhead for polymorphic expressions
- **Compile-time optimization**: Rust's optimizer can inline and optimize across interpreter boundaries
- **Type-driven dispatch**: No dynamic dispatch or boxing required

## 📚 Examples

See the `examples/` directory for more comprehensive examples:

- `basic_usage.rs`: Introduction to the final tagless approach
- `polynomial_demo.rs`: Comprehensive polynomial evaluation using Horner's method
- More examples coming soon!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT OR Apache-2.0 license.

## 🔗 Related Projects

- [symbolic-math](../symbolic-math): The original symbolic mathematics crate that inspired MathJIT
- [measures](../): The parent project for measure theory and probability distributions 