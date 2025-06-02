impl MathBuilder {
    /// Compose multiple independent expressions with automatic variable remapping
    /// 
    /// **DEPRECATED**: This method is being removed in favor of type-level scoped variables.
    /// Use `mathcompile::compile_time::scoped::compose()` instead for zero-overhead composition.
    /// 
    /// # Example with Scoped Variables (Recommended)
    /// ```rust
    /// use mathcompile::compile_time::scoped::{scoped_var, scoped_constant, compose};
    /// 
    /// // Define f(x) = x² in scope 0
    /// let x_f = scoped_var::<0, 0>();
    /// let f = x_f.clone().mul(x_f);
    /// 
    /// // Define g(y) = 2y in scope 1
    /// let y_g = scoped_var::<0, 1>();
    /// let g = y_g.mul(scoped_constant::<1>(2.0));
    /// 
    /// // Compose h = f + g with automatic remapping
    /// let h = compose(f, g).add();
    /// 
    /// // Evaluate h(3, 4) = f(3) + g(4) = 9 + 8 = 17
    /// let result = h.eval(&[3.0, 4.0]);
    /// assert_eq!(result, 17.0);
    /// ```
    #[deprecated(note = "Use type-level scoped variables instead")]
    pub fn compose_functions(&self, expressions: &[crate::ast::ASTRepr<f64>]) -> Vec<crate::ast::ASTRepr<f64>> {
        use crate::ast::ast_utils::combine_expressions_with_remapping;
        let (remapped_expressions, _total_vars) = combine_expressions_with_remapping(expressions);
        remapped_expressions
    }

    /// Compose expressions using type-level scoped variables (Recommended)
    /// 
    /// This method provides type-safe composition with zero runtime overhead.
    /// Variables from different scopes cannot be accidentally mixed.
    /// 
    /// # Example
    /// ```rust
    /// use mathcompile::compile_time::scoped::{scoped_var, scoped_constant, ScopedMathExpr};
    /// use mathcompile::prelude::*;
    /// 
    /// let math = MathBuilder::new();
    /// 
    /// // Define f(x) = x² in scope 0
    /// let x_f = scoped_var::<0, 0>();
    /// let f = x_f.clone().mul(x_f);
    /// 
    /// // Define g(y) = 2y in scope 1
    /// let y_g = scoped_var::<0, 1>();
    /// let g = y_g.mul(scoped_constant::<1>(2.0));
    /// 
    /// // Compose with type safety
    /// let composed = math.compose_scoped(f, g);
    /// let h = composed.add();
    /// 
    /// // Zero-overhead evaluation
    /// let result = h.eval(&[3.0, 4.0]);
    /// assert_eq!(result, 17.0);
    /// ```
    pub fn compose_scoped<L, R, const SCOPE1: usize, const SCOPE2: usize>(
        &self,
        left: L,
        right: R,
    ) -> crate::compile_time::scoped::ComposedExpr<L, R, SCOPE1, SCOPE2>
    where
        L: crate::compile_time::scoped::ScopedMathExpr<SCOPE1>,
        R: crate::compile_time::scoped::ScopedMathExpr<SCOPE2>,
    {
        crate::compile_time::scoped::compose(left, right)
    }

    /// Create a composed function from independent expressions with a combiner function
    /// 
    /// **DEPRECATED**: This method is being removed in favor of type-level scoped variables.
    /// Use `compose_scoped()` instead for better performance and safety.
    #[deprecated(note = "Use compose_scoped() with type-level scoped variables instead")]
    pub fn compose_with_combiner<F, T>(&self, expressions: &[crate::ast::ASTRepr<f64>], combiner: F) -> T
    where
        F: FnOnce(&[crate::ast::ASTRepr<f64>]) -> T,
    {
        let remapped = self.compose_functions(expressions);
        combiner(&remapped)
    }
} 