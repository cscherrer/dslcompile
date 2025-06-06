# ANF Developer Cheat Sheet
*Basic A-Normal Form implementation for dslcompile*

## Quick Usage

```rust
use dslcompile::anf::{convert_to_anf, generate_rust_code, ANFCodeGen};
use dslcompile::final_tagless::{ASTEval, ASTMathExpr, VariableRegistry};

// 1. Create expression
let mut registry = VariableRegistry::new();
let x = ASTEval::var(registry.register_variable("x"));
let expr = ASTEval::add(x.clone(), x); // x + x

// 2. Convert to ANF
let anf = convert_to_anf(&expr)?;

// 3. Generate code
let code = generate_rust_code(&anf, &registry);
// Result: { let t0 = x + x; t0 }
```

## What ANF Does

ANF (A-Normal Form) transforms mathematical expressions into an intermediate representation where:
- Every operation has atomic (non-compound) arguments
- Intermediate results are bound to temporary variables
- Common subexpressions are automatically eliminated

### Basic CSE Examples

| Input Expression | ANF Output | Benefit |
|------------------|------------|---------|
| `(x + 1) + (x + 1)` | `{ let t0 = x + 1; { let t1 = t0 + t0; t1 } }` | Reuses `x + 1` |
| `sin(x + y) + cos(x + y)` | `{ let t0 = x + y; { let t1 = t0.sin(); { let t2 = t0.cos(); t1 + t2 } } }` | Reuses `x + y` |
| `x * x * x` | `{ let t0 = x * x; { let t1 = t0 * x; t1 } }` | Reuses `x * x` |

## Key Data Structures

```rust
// Variables: hybrid user + generated
pub enum VarRef {
    User(usize),    // Original variables (x, y, z)
    Bound(u32),     // Temporaries (t0, t1, t2)
}

// Atomic values only
pub enum ANFAtom<T> {
    Constant(T),
    Variable(VarRef),
}

// Operations with atomic args
pub enum ANFComputation<T> {
    Add(ANFAtom<T>, ANFAtom<T>),
    Sin(ANFAtom<T>),
    // ... all math operations
}

// Complete expressions
pub enum ANFExpr<T> {
    Atom(ANFAtom<T>),                    // x, 42.0
    Let(VarRef, ANFComputation<T>, Box<ANFExpr<T>>),  // let t0 = x + 1 in ...
}
```

## Debugging Commands

```rust
// Print structure
println!("ANF: {:#?}", anf);

// Count bindings  
println!("Let count: {}", anf.let_count());

// List variables
println!("Variables: {:?}", anf.used_variables());

// Generate with debug info
let codegen = ANFCodeGen::new(&registry);
let func = codegen.generate_function("debug", &anf);
println!("Function:\n{}", func);
```

## Common Patterns

### Add New Operation
```rust
// 1. Add to ANFComputation
pub enum ANFComputation<T> {
    // ... existing ops ...
    MyNewOp(ANFAtom<T>),
}

// 2. Add to converter
ASTRepr::MyNewAst(inner) => {
    self.convert_unary_op_with_cse(expr, inner, ANFComputation::MyNewOp)
}

// 3. Add to code generator
ANFComputation::MyNewOp(operand) => {
    format!("{}.my_method()", self.generate_atom(operand))
}
```

### Reuse Converter for Better CSE
```rust
// Reuse converter for better CSE across multiple expressions
let mut converter = ANFConverter::new();
let anf1 = converter.convert(&expr1)?; // Builds cache
let anf2 = converter.convert(&expr2)?; // Reuses cache

// Clear cache if memory is a concern
let converter = ANFConverter::new(); // Fresh cache
```

## Current Limitations

- **No dead code elimination**: Unused let-bindings are not removed
- **No constant folding**: Constant expressions are not evaluated at compile time
- **No optimization metrics**: No quantitative analysis of CSE effectiveness
- **Basic scope management**: Simple depth-based scope tracking
- **Memory usage**: CSE cache grows without bounds

## Troubleshooting

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| "Invalid variable reference" | Scope bug in CSE | Check cache scope validation |
| "Stack overflow" | Recursive expression | Add cycle detection (not implemented) |
| "Out of memory" | Cache growth | Clear converter cache |
| "Wrong variable name" | Registry mismatch | Ensure consistent registry usage |

## Integration Points

- **Input**: `ASTRepr<T>` from existing mathematical expressions
- **Output**: Clean Rust code via `generate_rust_code()`
- **Variables**: Uses existing `VariableRegistry` system
- **Testing**: Property-based tests for robustness

## Future Work

See ROADMAP.md for planned enhancements including:
- Constant folding and dead code elimination
- Optimization metrics and performance analysis
- Better memory management for CSE cache
- Integration with egglog and other optimization backends 