# Type-Level Variable Scoping Design

## Overview

This design proposes a type-level solution to variable collision problems that works for both compile-time (static) and runtime (dynamic) expression systems.

## Core Concept: Scoped Variables

### 1. **Scope-Aware Variables**

```rust
// Compile-time: Variables carry scope information in their type
struct Var<const ID: usize, const SCOPE: usize>;

// Runtime: Variables carry scope information via phantom types
struct ScopedVar<const SCOPE: usize> {
    index: usize,
    _scope: PhantomData<[(); SCOPE]>,
}
```

### 2. **Scope-Aware Builders**

```rust
// Compile-time: No builder needed, direct type construction
fn create_scoped_expression<const SCOPE: usize>() -> impl MathExpr {
    let x = var::<0, SCOPE>();
    let y = var::<1, SCOPE>();
    x.add(y)
}

// Runtime: Builder carries scope information
struct ScopedMathBuilder<const SCOPE: usize> {
    registry: Arc<RefCell<VariableRegistry>>,
    _scope: PhantomData<[(); SCOPE]>,
}
```

## Implementation Strategy

### **Phase 1: Compile-Time System Enhancement**

```rust
// Static compile-time variables with scope
#[derive(Clone, Debug)]
pub struct Var<const ID: usize, const SCOPE: usize>;

impl<const ID: usize, const SCOPE: usize> MathExpr for Var<ID, SCOPE> {
    fn eval(&self, vars: &[f64]) -> f64 {
        // Scope-aware variable access
        let scoped_vars = get_scoped_vars::<SCOPE>(vars);
        scoped_vars.get(ID).copied().unwrap_or(0.0)
    }
}

// Scope composition at the type level
pub trait ScopeCompose<Other> {
    type Output;
    fn compose(self, other: Other) -> Self::Output;
}

impl<const SCOPE1: usize, const SCOPE2: usize> 
    ScopeCompose<Expr<SCOPE2>> for Expr<SCOPE1> 
{
    type Output = Expr<{SCOPE1 + SCOPE2}>;
    
    fn compose(self, other: Expr<SCOPE2>) -> Self::Output {
        // Automatic variable remapping at compile time
        let remapped_self = self.remap_scope::<{SCOPE1 + SCOPE2}>();
        let remapped_other = other.remap_scope::<{SCOPE1 + SCOPE2}>()
            .offset_variables::<SCOPE1>();
        
        remapped_self.add(remapped_other)
    }
}
```

### **Phase 2: Runtime System Enhancement**

```rust
// Scoped runtime builder
impl<const SCOPE: usize> ScopedMathBuilder<SCOPE> {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RefCell::new(VariableRegistry::new())),
            _scope: PhantomData,
        }
    }
    
    pub fn var(&self) -> ScopedBuilderExpr<f64, SCOPE> {
        let index = self.registry.borrow_mut().register_variable();
        ScopedBuilderExpr::new(
            ASTRepr::Variable(index), 
            self.registry.clone()
        )
    }
    
    // Type-safe composition
    pub fn compose_with<const OTHER_SCOPE: usize>(
        self, 
        other: ScopedMathBuilder<OTHER_SCOPE>
    ) -> ComposedBuilder<SCOPE, OTHER_SCOPE> {
        ComposedBuilder::new(self, other)
    }
}

// Scoped expressions that prevent variable collision
#[derive(Debug, Clone)]
pub struct ScopedBuilderExpr<T, const SCOPE: usize> {
    ast: ASTRepr<T>,
    registry: Arc<RefCell<VariableRegistry>>,
    _scope: PhantomData<[(); SCOPE]>,
}

// Only allow composition between compatible scopes
impl<T, const SCOPE: usize> Add<ScopedBuilderExpr<T, SCOPE>> 
    for ScopedBuilderExpr<T, SCOPE> 
{
    type Output = ScopedBuilderExpr<T, SCOPE>;
    
    fn add(self, rhs: ScopedBuilderExpr<T, SCOPE>) -> Self::Output {
        // Safe: same scope, no collision possible
        ScopedBuilderExpr::new(
            self.ast + rhs.ast,
            self.registry
        )
    }
}

// Cross-scope composition requires explicit handling
impl<T, const SCOPE1: usize, const SCOPE2: usize> 
    ScopedBuilderExpr<T, SCOPE1> 
{
    pub fn compose_with<F>(
        self, 
        other: ScopedBuilderExpr<T, SCOPE2>,
        combiner: F
    ) -> ScopedBuilderExpr<T, {SCOPE1 + SCOPE2}>
    where
        F: FnOnce(ASTRepr<T>, ASTRepr<T>) -> ASTRepr<T>
    {
        // Automatic variable remapping
        let remapped_other = remap_variables(
            &other.ast, 
            &create_offset_map(SCOPE1)
        );
        
        let combined_ast = combiner(self.ast, remapped_other);
        
        ScopedBuilderExpr::new(
            combined_ast,
            merge_registries(self.registry, other.registry)
        )
    }
}
```

## **Benefits for Both Systems**

### **Compile-Time Benefits**
1. **Zero Runtime Cost**: All scope checking happens at compile time
2. **Impossible Variable Collisions**: Type system prevents invalid compositions
3. **Clear Intent**: Scope information is explicit in types
4. **Automatic Remapping**: Compiler generates optimal variable layouts

### **Runtime Benefits**
1. **Type-Safe Composition**: Prevents accidental variable collisions
2. **Clear API**: Scope requirements are explicit in function signatures
3. **Automatic Registry Management**: No manual variable index tracking
4. **Performance**: Scope information guides optimization

## **Usage Examples**

### **Compile-Time Usage**

```rust
// Define functions in separate scopes
fn define_f() -> impl MathExpr {
    let x = var::<0, 0>();  // Scope 0, variable 0
    let y = var::<1, 0>();  // Scope 0, variable 1
    x.mul(x).add(y.mul(constant(2.0)))
}

fn define_g() -> impl MathExpr {
    let y = var::<0, 1>();  // Scope 1, variable 0 (different scope!)
    let z = var::<1, 1>();  // Scope 1, variable 1
    y.mul(constant(3.0)).add(z)
}

// Safe composition
let f = define_f();
let g = define_g();
let h = f.compose(g);  // Automatic scope merging and variable remapping

// Evaluation with properly scoped variables
let result = h.eval(&[1.0, 2.0, 3.0, 4.0]);  // [f_x, f_y, g_y, g_z]
```

### **Runtime Usage**

```rust
// Define functions with scoped builders
let math_f = ScopedMathBuilder::<0>::new();
let x_f = math_f.var();  // Scoped to builder 0
let y_f = math_f.var();  // Scoped to builder 0
let f_expr = &x_f * &x_f + 2.0 * &y_f;

let math_g = ScopedMathBuilder::<1>::new();
let y_g = math_g.var();  // Scoped to builder 1 - no collision!
let z_g = math_g.var();  // Scoped to builder 1
let g_expr = 3.0 * &y_g + &z_g;

// Type-safe composition
let h_expr = f_expr.compose_with(g_expr, |f, g| {
    ASTRepr::Add(Box::new(f), Box::new(g))
});

// Evaluation with automatic variable layout
let result = h_expr.eval(&[1.0, 2.0, 3.0, 4.0]);
```

## **Migration Strategy**

### **Phase 1: Backward Compatible Enhancement**
- Add scoped variants alongside existing APIs
- Default scope = 0 for backward compatibility
- Gradual migration of examples and tests

### **Phase 2: Advanced Features**
- Scope inference for common patterns
- Automatic scope optimization
- Integration with procedural macros

### **Phase 3: Full Integration**
- Make scoped APIs the primary interface
- Deprecate unscoped variants
- Complete documentation and examples

## **Implementation Challenges**

### **Const Generic Limitations**
- Rust's const generics have limitations with complex expressions
- May need to use type-level numbers or other workarounds
- Consider using procedural macros for complex scope calculations

### **Runtime Performance**
- Scope information should be zero-cost at runtime
- Registry merging needs to be efficient
- Variable remapping should be optimized

### **API Complexity**
- Balance between type safety and usability
- Provide good error messages for scope mismatches
- Consider ergonomic helpers for common patterns

## **Conclusion**

Type-level variable scoping can work for both static and dynamic systems:

- **Static System**: Perfect fit, leverages Rust's type system fully
- **Dynamic System**: Requires more design work but provides significant safety benefits

The key insight is using phantom types and const generics to carry scope information without runtime cost, while providing compile-time guarantees about variable safety.

This approach eliminates the need for runtime HashMap-based remapping while providing stronger guarantees than the current manual approach. 