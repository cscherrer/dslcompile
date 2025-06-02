# Migration Plan: HashMap → Type-Level Scoped Variables

**Status**: Ready to Execute  
**Priority**: High  
**Estimated Effort**: 2-3 days  

## 🎯 **Migration Overview**

Replace the HashMap-based variable remapping system with our superior type-level scoped variables implementation.

## 📋 **Current HashMap Usage**

### **Core Implementation**
- `src/ast/ast_utils.rs`: `combine_expressions_with_remapping()`, `remap_variables()`
- `src/final_tagless/math_builder.rs`: `compose_functions()` method

### **Test Coverage**
- `tests/function_composition_solution.rs`: Demonstrates HashMap remapping
- `tests/shared_variable_composition.rs`: Complex composition scenarios
- `tests/independent_function_composition.rs`: Variable collision examples

### **Documentation**
- `docs/VARIABLE_COMPOSITION_THEORY.md`: Theoretical background

## 🔄 **Migration Steps**

### **Phase 1: Deprecation Warnings** (Day 1)
1. Add deprecation warnings to HashMap-based functions
2. Update documentation to recommend scoped variables
3. Add migration examples

### **Phase 2: API Updates** (Day 2)
1. Add scoped variable alternatives to `MathBuilder`
2. Update examples to use scoped approach
3. Create compatibility layer for existing code

### **Phase 3: Removal** (Day 3)
1. Remove HashMap-based functions
2. Clean up deprecated code
3. Update all tests to use scoped variables

## 📝 **Specific Changes Required**

### **1. Update `MathBuilder`**

```rust
// OLD: HashMap-based composition
impl MathBuilder {
    pub fn compose_functions(&self, expressions: &[ASTRepr<f64>]) -> Vec<ASTRepr<f64>> {
        // Uses combine_expressions_with_remapping internally
    }
}

// NEW: Scoped composition
impl MathBuilder {
    pub fn compose_scoped<L, R, const SCOPE1: usize, const SCOPE2: usize>(
        &self,
        left: L,
        right: R,
    ) -> ComposedExpr<L, R, SCOPE1, SCOPE2>
    where
        L: ScopedMathExpr<SCOPE1>,
        R: ScopedMathExpr<SCOPE2>,
    {
        compose(left, right)
    }
}
```

### **2. Update Examples**

```rust
// OLD: Manual HashMap remapping
let mut var_map = HashMap::new();
var_map.insert(0, 1);
let g_remapped = remap_variables(g_ast, &var_map);
let h = f_ast + g_remapped;

// NEW: Type-safe scoped composition
let f = scoped_var::<0, 0>().mul(scoped_var::<0, 0>()); // f(x) = x²
let g = scoped_var::<0, 1>().mul(scoped_constant::<1>(2.0)); // g(y) = 2y
let h = compose(f, g).add(); // h(x,y) = f(x) + g(y)
```

### **3. Performance Comparison**

```rust
// Benchmark: HashMap vs Scoped Variables
#[bench]
fn bench_hashmap_composition(b: &mut Bencher) {
    // Current HashMap approach
}

#[bench] 
fn bench_scoped_composition(b: &mut Bencher) {
    // New scoped approach
}
```

## ✅ **Benefits After Migration**

### **Performance**
- **Zero runtime overhead**: No HashMap lookups
- **Compile-time optimization**: Better compiler optimization
- **Memory efficiency**: No HashMap storage

### **Safety**
- **Compile-time guarantees**: Impossible variable collisions
- **Type safety**: Clear scope boundaries
- **Better error messages**: Compiler catches scope violations

### **Developer Experience**
- **Clearer intent**: Scope information in types
- **Self-documenting**: Code shows variable relationships
- **Easier debugging**: Compile-time error detection

## 🧪 **Validation Plan**

### **1. Correctness Testing**
- All existing composition tests must pass with scoped variables
- Property-based testing for mathematical equivalence
- Edge case validation

### **2. Performance Testing**
- Benchmark composition operations
- Memory usage comparison
- Compilation time impact

### **3. API Compatibility**
- Ensure smooth migration path for existing users
- Provide clear migration examples
- Document breaking changes

## 📅 **Timeline**

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1** | 1 day | Deprecation warnings, updated docs |
| **Phase 2** | 1 day | New scoped APIs, compatibility layer |
| **Phase 3** | 1 day | HashMap removal, test updates |
| **Total** | **3 days** | **Complete migration** |

## 🎉 **Success Criteria**

- [ ] All HashMap-based variable remapping removed
- [ ] All tests pass with scoped variables
- [ ] Performance improvements demonstrated
- [ ] Documentation updated
- [ ] Migration guide available
- [ ] Zero runtime overhead achieved

## 🔮 **Future Enhancements**

After migration, we can add:
- **Hierarchical scopes**: Nested scope support
- **Scope inference**: Automatic scope detection
- **Advanced composition**: More complex scope operations
- **Integration**: Scope-aware optimization passes

---

**Ready to execute!** The type-level scoped variables are production-ready and provide superior performance and safety compared to the HashMap approach. 