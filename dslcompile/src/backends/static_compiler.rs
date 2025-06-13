//! Static Compilation Backend - Zero Overhead Inline Code Generation
//!
//! This module provides true static compilation by generating inline Rust code
//! that can be embedded directly into user programs, eliminating all FFI overhead
//! and achieving performance identical to hand-written Rust.

use crate::{
    ast::{ASTRepr, Scalar, VariableRegistry},
    backends::RustCodeGenerator,
    error::Result,
};
use num_traits::Float;
use std::collections::HashMap;

/// Static compiler that generates inline Rust code for zero-overhead evaluation
pub struct StaticCompiler {
    /// Code generator for creating Rust expressions
    codegen: RustCodeGenerator,
    /// Cache of generated inline functions
    function_cache: HashMap<String, String>,
}

impl StaticCompiler {
    /// Create a new static compiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            codegen: RustCodeGenerator::new(),
            function_cache: HashMap::new(),
        }
    }

    /// Generate inline Rust code that can be embedded directly in user programs
    ///
    /// This generates pure Rust expressions with no FFI overhead, achieving
    /// performance identical to hand-written Rust code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use dslcompile::backends::StaticCompiler;
    /// use dslcompile::prelude::*;
    ///
    /// let mut ctx = DynamicContext::new();
    /// let x = ctx.var();
    /// let expr = &x * &x + 2.0 * &x + 1.0;
    /// let ast = ctx.to_ast(&expr);
    ///
    /// let mut compiler = StaticCompiler::new();
    /// let inline_code = compiler.generate_inline_function(&ast, "my_func").unwrap();
    ///
    /// // The generated code can be embedded directly:
    /// // fn my_func(var_0: f64) -> f64 { (var_0 * var_0) + ((2_f64 * var_0) + 1_f64) }
    /// ```
    pub fn generate_inline_function<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
        function_name: &str,
    ) -> Result<String> {
        // Check cache first
        let cache_key = format!("{function_name}_{}", std::any::type_name::<T>());
        if let Some(cached) = self.function_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Use existing RustCodeGenerator functionality
        let full_function = self.codegen.generate_function_generic(
            expr,
            function_name,
            std::any::type_name::<T>(),
        )?;

        // Convert to inline format - this is the only unique functionality
        let inline_function = self.format_as_inline_function(&full_function, function_name)?;

        // Cache the result
        self.function_cache
            .insert(cache_key, inline_function.clone());

        Ok(inline_function)
    }

    /// Generate a complete Rust module with multiple inline functions
    ///
    /// This creates a module that can be included directly in user code,
    /// providing zero-overhead evaluation for multiple expressions.
    pub fn generate_inline_module<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expressions: &[(String, ASTRepr<T>)],
        module_name: &str,
    ) -> Result<String> {
        // Use existing RustCodeGenerator module generation
        let full_module = self.codegen.generate_module_generic(
            expressions,
            module_name,
            std::any::type_name::<T>(),
        )?;

        // Convert to inline format
        self.format_as_inline_module(&full_module, module_name)
    }

    /// Generate a macro that creates the inline function at compile time
    ///
    /// This approach allows the function to be generated and inlined at the
    /// call site, providing maximum optimization opportunities.
    pub fn generate_inline_macro<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
        macro_name: &str,
    ) -> Result<String> {
        // Create registry for variable management
        let mut registry = VariableRegistry::new();
        let variables = crate::ast::ast_utils::collect_variable_indices(expr);

        let mut sorted_variables: Vec<usize> = variables.into_iter().collect();
        sorted_variables.sort_unstable();

        let max_var_index = sorted_variables.iter().max().copied().unwrap_or(0);
        for _ in 0..=max_var_index {
            let _var_idx = registry.register_variable();
        }

        // Use existing inline expression generation
        let expr_code = self.codegen.generate_inline_expression(expr, &registry)?;

        // Generate parameter names for the macro
        let param_names = (0..=max_var_index)
            .map(|i| format!("$var_{i}"))
            .collect::<Vec<_>>()
            .join(", ");

        // Generate the macro - this is unique functionality
        let inline_macro = format!(
            r#"/// Generated inline macro: {macro_name}
/// Zero overhead - expands to pure Rust expression at compile time
macro_rules! {macro_name} {{
    ({param_names}) => {{
        {expr_code}
    }};
}}"#,
            macro_name = macro_name,
            param_names = param_names,
            expr_code = expr_code.replace("var_", "$var_")
        );

        Ok(inline_macro)
    }

    /// Clear the function cache
    pub fn clear_cache(&mut self) {
        self.function_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.function_cache.len(), self.function_cache.capacity())
    }

    // ========================================================================
    // PRIVATE HELPER METHODS - The only unique functionality
    // ========================================================================

    /// Convert a full function to inline format
    fn format_as_inline_function(
        &self,
        full_function: &str,
        function_name: &str,
    ) -> Result<String> {
        // Extract the function body from the full function
        // This is a simple text transformation - the real work is done by RustCodeGenerator

        // Find the function signature and body
        if let Some(fn_start) = full_function.find(&format!("fn {function_name}("))
            && let Some(body_start) = full_function[fn_start..].find('{')
        {
            let body_start = fn_start + body_start;
            if let Some(body_end) = full_function.rfind('}') {
                let signature = &full_function[fn_start..body_start].trim();
                let body = &full_function[body_start + 1..body_end].trim();

                return Ok(format!(
                    r#"/// Generated inline function: {function_name}
/// Zero overhead - identical performance to hand-written Rust
#[inline]
{signature} {{
    {body}
}}"#
                ));
            }
        }

        // Fallback: just add #[inline] attribute
        Ok(format!(
            r#"/// Generated inline function: {function_name}
/// Zero overhead - identical performance to hand-written Rust
#[inline]
{full_function}"#
        ))
    }

    /// Convert a full module to inline format
    fn format_as_inline_module(&self, full_module: &str, module_name: &str) -> Result<String> {
        // Add inline attributes to all functions in the module
        let inline_module = full_module
            .lines()
            .map(|line| {
                if line.trim_start().starts_with("pub fn ") || line.trim_start().starts_with("fn ")
                {
                    format!("    #[inline]\n{line}")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(
            r#"//! Generated inline module: {module_name}
//! Zero overhead mathematical expressions - identical performance to hand-written Rust
//! 
//! All functions in this module are marked #[inline] and have no FFI overhead.
//! They can be called directly with native Rust types.

{inline_module}"#
        ))
    }
}

impl Default for StaticCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be statically compiled to inline Rust code
pub trait StaticCompilable<T: Scalar> {
    /// Generate inline Rust function for this expression
    fn to_inline_function(&self, function_name: &str) -> Result<String>;

    /// Generate inline Rust macro for this expression
    fn to_inline_macro(&self, macro_name: &str) -> Result<String>;
}

impl<T: Scalar + Float + Copy + 'static> StaticCompilable<T> for ASTRepr<T> {
    fn to_inline_function(&self, function_name: &str) -> Result<String> {
        let mut compiler = StaticCompiler::new();
        compiler.generate_inline_function(self, function_name)
    }

    fn to_inline_macro(&self, macro_name: &str) -> Result<String> {
        let mut compiler = StaticCompiler::new();
        compiler.generate_inline_macro(self, macro_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_inline_function_generation() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let expr = &x * &x + 2.0 * &x + 1.0;
        let ast = ctx.to_ast(&expr);

        let mut compiler = StaticCompiler::new();
        let inline_code = compiler
            .generate_inline_function(&ast, "test_poly")
            .unwrap();

        assert!(inline_code.contains("#[inline]"));
        assert!(inline_code.contains("fn test_poly"));
        assert!(inline_code.contains("var_0: f64"));
        assert!(inline_code.contains("-> f64"));

        println!("Generated inline function:\n{}", inline_code);
    }

    #[test]
    fn test_inline_macro_generation() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let y = ctx.var();
        let expr = &x * &y + 1.0;
        let ast = ctx.to_ast(&expr);

        let mut compiler = StaticCompiler::new();
        let macro_code = compiler.generate_inline_macro(&ast, "test_macro").unwrap();

        assert!(macro_code.contains("macro_rules! test_macro"));
        assert!(macro_code.contains("$var_0"));
        assert!(macro_code.contains("$var_1"));

        println!("Generated inline macro:\n{}", macro_code);
    }

    #[test]
    fn test_cache_functionality() {
        let mut ctx = DynamicContext::<f64>::new();
        let x = ctx.var();
        let expr = &x + 1.0;
        let ast = ctx.to_ast(&expr);

        let mut compiler = StaticCompiler::new();

        // First call should generate
        let code1 = compiler
            .generate_inline_function(&ast, "cached_func")
            .unwrap();
        let (cache_size_1, _) = compiler.cache_stats();

        // Second call should use cache
        let code2 = compiler
            .generate_inline_function(&ast, "cached_func")
            .unwrap();
        let (cache_size_2, _) = compiler.cache_stats();

        assert_eq!(code1, code2);
        assert_eq!(cache_size_1, 1);
        assert_eq!(cache_size_2, 1); // Should not increase

        // Clear cache
        compiler.clear_cache();
        let (cache_size_3, _) = compiler.cache_stats();
        assert_eq!(cache_size_3, 0);
    }
}
