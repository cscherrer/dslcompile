impl MathBuilder {
    /// Compose multiple independent expressions with automatic variable remapping
    /// 
    /// This method takes expressions that were built independently (potentially with
    /// overlapping variable indices) and combines them with proper variable remapping
    /// to avoid collisions.
    /// 
    /// # Example
    /// ```rust
    /// use mathcompile::prelude::*;
    /// 
    /// // Define f(x) = x² + 2x + 1 independently
    /// let math_f = MathBuilder::new();
    /// let x_f = math_f.var();
    /// let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0;
    /// 
    /// // Define g(y) = 3y + 5 independently
    /// let math_g = MathBuilder::new();
    /// let y_g = math_g.var();
    /// let g_expr = 3.0 * &y_g + 5.0;
    /// 
    /// // Compose h(x,y) = f(x) + g(y) with automatic remapping
    /// let math_h = MathBuilder::new();
    /// let (f_remapped, g_remapped) = math_h.compose_functions(&[
    ///     f_expr.as_ast(),
    ///     g_expr.as_ast()
    /// ]);
    /// 
    /// // Now we can safely add them
    /// let h_ast = crate::ast::ASTRepr::Add(
    ///     Box::new(f_remapped[0].clone()),
    ///     Box::new(f_remapped[1].clone())
    /// );
    /// ```
    pub fn compose_functions(&self, expressions: &[crate::ast::ASTRepr<f64>]) -> Vec<crate::ast::ASTRepr<f64>> {
        use crate::ast::ast_utils::combine_expressions_with_remapping;
        let (remapped_expressions, _total_vars) = combine_expressions_with_remapping(expressions);
        remapped_expressions
    }

    /// Create a composed function from independent expressions with a combiner function
    /// 
    /// This is a higher-level interface that automatically handles variable remapping
    /// and applies a combiner function to create the final expression.
    /// 
    /// # Example
    /// ```rust
    /// use mathcompile::prelude::*;
    /// 
    /// // Define independent functions
    /// let math_f = MathBuilder::new();
    /// let x_f = math_f.var();
    /// let f_expr = &x_f * &x_f + 2.0 * &x_f + 1.0; // f(x) = x² + 2x + 1
    /// 
    /// let math_g = MathBuilder::new();
    /// let y_g = math_g.var();
    /// let g_expr = 3.0 * &y_g + 5.0; // g(y) = 3y + 5
    /// 
    /// // Create h(x,y) = f(x) + g(y)
    /// let math_h = MathBuilder::new();
    /// let h_ast = math_h.compose_with_combiner(
    ///     &[f_expr.as_ast(), g_expr.as_ast()],
    ///     |exprs| {
    ///         crate::ast::ASTRepr::Add(
    ///             Box::new(exprs[0].clone()),
    ///             Box::new(exprs[1].clone())
    ///         )
    ///     }
    /// );
    /// ```
    pub fn compose_with_combiner<F>(&self, expressions: &[crate::ast::ASTRepr<f64>], combiner: F) -> crate::ast::ASTRepr<f64>
    where
        F: FnOnce(&[crate::ast::ASTRepr<f64>]) -> crate::ast::ASTRepr<f64>,
    {
        let remapped_expressions = self.compose_functions(expressions);
        combiner(&remapped_expressions)
    }
} 