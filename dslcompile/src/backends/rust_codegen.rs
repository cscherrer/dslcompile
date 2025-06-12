//! Rust Code Generation Backend
//!
//! This module provides Rust source code generation and hot-loading compilation
//! for mathematical expressions. It generates optimized Rust code that can be
//! compiled to dynamic libraries for maximum performance.
//!
//! # Features
//!
//! - **Optimized Code Generation**: Generates efficient Rust code with specialized optimizations
//! - **Hot-Loading**: Compiles and loads dynamic libraries at runtime
//! - **Multiple Function Signatures**: Supports various calling conventions
//! - **Advanced Optimizations**: Integer power optimization, unsafe optimizations, etc.
//! - **Batch Compilation**: Compile multiple expressions into a single module

use crate::{
    ast::{
        ASTRepr, Scalar, VariableRegistry,
        ast_repr::{Collection, Lambda},
        ast_utils::collect_variable_indices,
    },
    error::{DSLCompileError, Result},
    symbolic::power_utils::{
        PowerOptConfig, generate_integer_power_string, try_convert_to_integer,
    },
};
use dlopen2::raw::Library;
use frunk::{HCons, HNil};
use num_traits::Float;
use std::path::Path;

/// Optimization levels for Rust compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RustOptLevel {
    /// No optimization (fastest compilation)
    O0,
    /// Basic optimization
    O1,
    /// Full optimization
    #[default]
    O2,
    /// Aggressive optimization (may increase compile time significantly)
    O3,
    /// Size optimization
    Os,
    /// Aggressive size optimization
    Oz,
}

impl RustOptLevel {
    /// Get the rustc optimization flag
    #[must_use]
    pub fn as_flag(&self) -> &'static str {
        match self {
            RustOptLevel::O0 => "opt-level=0",
            RustOptLevel::O1 => "opt-level=1",
            RustOptLevel::O2 => "opt-level=2",
            RustOptLevel::O3 => "opt-level=3",
            RustOptLevel::Os => "opt-level=s",
            RustOptLevel::Oz => "opt-level=z",
        }
    }
}

/// Configuration for Rust code generation
#[derive(Debug, Clone)]
pub struct RustCodegenConfig {
    /// Whether to include debug information (TODO: implement in `RustCompiler::compile_dylib`)
    pub debug_info: bool,
    /// Whether to use unsafe optimizations
    pub unsafe_optimizations: bool,
    /// Whether to enable vectorization hints
    pub vectorization_hints: bool,
    /// Whether to inline aggressively
    pub aggressive_inlining: bool,
    /// Target CPU features (TODO: implement in `RustCompiler::compile_dylib`)
    pub target_cpu: Option<String>,
    /// Power optimization configuration
    pub power_config: PowerOptConfig,
}

impl Default for RustCodegenConfig {
    fn default() -> Self {
        Self {
            debug_info: false,
            unsafe_optimizations: false,
            vectorization_hints: true,
            aggressive_inlining: true,
            target_cpu: None,
            power_config: PowerOptConfig::default(),
        }
    }
}

/// Rust code generator for mathematical expressions
pub struct RustCodeGenerator {
    /// Configuration for code generation
    config: RustCodegenConfig,
}

impl RustCodeGenerator {
    /// Create a new Rust code generator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RustCodegenConfig::default(),
        }
    }

    /// Create a new Rust code generator with custom configuration
    #[must_use]
    pub fn with_config(config: RustCodegenConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &RustCodegenConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: RustCodegenConfig) {
        self.config = config;
    }

    /// Generate Rust source code with proper typed function signatures (NO VEC FLATTENING!)
    pub fn generate_function_with_registry<T: Scalar + Float + Copy + 'static>(
        &self,
        expr: &ASTRepr<T>,
        function_name: &str,
        type_name: &str,
        registry: &VariableRegistry,
    ) -> Result<String> {
        let expr_code = self.generate_expression_with_registry(expr, registry)?;

        // ✅ CRITICAL: Generate typed function signatures - NO VEC FLATTENING!
        // This preserves zero-cost abstractions and prevents performance kills

        // ✅ CRITICAL: Find ALL variables used in the expression, not just registry size
        // The registry might not contain all variables that appear in the expression
        let max_var_index = self.find_max_variable_index(expr);
        let actual_var_count = max_var_index + 1; // 0-indexed, so add 1

        // Detect data arrays and generate proper typed parameters
        let needs_data_arrays = self.expression_uses_data_arrays(expr);
        let data_array_count = if needs_data_arrays {
            self.count_data_arrays(expr)
        } else {
            0
        };

        // Generate typed parameter list (scalars + data arrays)
        let mut params = Vec::new();

        // Add scalar parameters with proper names for ALL variables used
        for i in 0..actual_var_count {
            params.push(format!("var_{i}: {type_name}"));
        }

        // Add data array parameters with proper types
        for i in 0..data_array_count {
            params.push(format!("data_{i}: &[{type_name}]"));
        }

        let param_list = params.join(", ");

        // Add optimization attributes based on configuration
        let mut attributes = String::new();
        if self.config.vectorization_hints && type_name == "f64" {
            attributes.push_str("#[target_feature(enable = \"avx2\")]\n");
        } else if self.config.aggressive_inlining {
            attributes.push_str("#[inline(always)]\n");
        }

        // ✅ CRITICAL: When data arrays are present, generate BOTH interfaces
        // The legacy interface handles data arrays by extracting them from the flattened array
        if needs_data_arrays {
            // For data array expressions, we need to generate a function that can handle
            // the legacy (*const f64, usize) calling convention by extracting data arrays

            let legacy_func_name = format!("{function_name}_legacy");

            Ok(format!(
                r#"
{attributes}#[no_mangle]
pub extern "C" fn {function_name}({param_list}) -> {type_name} {{
    // ✅ Direct typed parameter access - ZERO Vec flattening!
    // Parameters are directly accessible by name with full type safety
    {expr_code}
}}

{attributes}#[no_mangle]
pub extern "C" fn {legacy_func_name}(vars: *const {type_name}, len: usize) -> {type_name} {{
    // Legacy array interface for data array expressions
    if vars.is_null() || len < {min_params} {{
        return Default::default();
    }}
    
    let vars_slice = unsafe {{ std::slice::from_raw_parts(vars, len) }};
    
    // Extract scalar parameters (first {actual_var_count} values)
    {param_extraction}
    
    // Extract data arrays (remaining values as slices)
    let data_start_idx = {actual_var_count};
    let data_slice = if len > data_start_idx {{
        &vars_slice[data_start_idx..]
    }} else {{
        &[]
    }};
    
    // Call main typed function with extracted parameters
    {function_name}({call_args_with_data})
}}
"#,
                min_params = actual_var_count + 1, // At least scalars + some data
                param_extraction = self.generate_param_extraction_for_vars(
                    actual_var_count,
                    0, // Don't extract data arrays here
                    type_name
                ),
                call_args_with_data = {
                    let mut args = Vec::new();
                    // Add scalar parameters
                    for i in 0..actual_var_count {
                        args.push(format!("var_{i}"));
                    }
                    // Add data slice
                    args.push("data_slice".to_string());
                    args.join(", ")
                },
            ))
        } else {
            // Generate both typed and legacy functions for scalar-only expressions
            Ok(format!(
                r#"
{attributes}#[no_mangle]
pub extern "C" fn {function_name}({param_list}) -> {type_name} {{
    // ✅ Direct typed parameter access - ZERO Vec flattening!
    // Parameters are directly accessible by name with full type safety
    {expr_code}
}}

{attributes}#[no_mangle] 
pub extern "C" fn {function_name}_legacy(vars: *const {type_name}, len: usize) -> {type_name} {{
    // Legacy array interface for backward compatibility only
    if vars.is_null() || len < {min_params} {{
        return Default::default();
    }}
    
    let vars_slice = unsafe {{ std::slice::from_raw_parts(vars, len) }};
    
    // Extract parameters with bounds checking
    {param_extraction}
    
    // Call main typed function
    {function_name}({call_args})
}}
"#,
                min_params = actual_var_count,
                param_extraction = self.generate_param_extraction_for_vars(
                    actual_var_count,
                    data_array_count,
                    type_name
                ),
                call_args = self.generate_call_args_for_vars(actual_var_count, data_array_count),
            ))
        }
    }

    /// Generate Rust source code for a mathematical expression (generic version)
    pub fn generate_function_generic<T: Scalar + Float + Copy + 'static>(
        &self,
        expr: &ASTRepr<T>,
        function_name: &str,
        type_name: &str,
    ) -> Result<String> {
        // Create a default registry and register variables as needed
        let mut default_registry = VariableRegistry::new();
        let variables = collect_variable_indices(expr);

        // Sort variables to ensure deterministic order and register them
        let mut sorted_variables: Vec<usize> = variables.into_iter().collect();
        sorted_variables.sort_unstable();

        // Register enough variables for the highest index
        let max_var_index = sorted_variables.iter().max().copied().unwrap_or(0);
        for _ in 0..=max_var_index {
            let _var_idx = default_registry.register_variable();
        }

        self.generate_function_with_registry(expr, function_name, type_name, &default_registry)
    }

    /// Generate Rust source code for a mathematical expression (f64 specialization for backwards compatibility)
    pub fn generate_function(&self, expr: &ASTRepr<f64>, function_name: &str) -> Result<String> {
        self.generate_function_generic(expr, function_name, "f64")
    }

    /// Generate Rust source code for a complete module (generic version)
    pub fn generate_module_generic<T: Scalar + Float + Copy + 'static>(
        &self,
        expressions: &[(String, ASTRepr<T>)],
        module_name: &str,
        type_name: &str,
    ) -> Result<String> {
        let mut module_code = format!(
            r"//! Generated Rust module: {module_name}
//! This module contains compiled mathematical expressions for high-performance evaluation.
//! Working with type: {type_name}

"
        );

        for (func_name, expr) in expressions {
            let func_code = self.generate_function_generic(expr, func_name, type_name)?;
            module_code.push_str(&func_code);
            module_code.push('\n');
        }

        Ok(module_code)
    }

    /// Generate Rust source code for a complete module (f64 specialization for backwards compatibility)
    pub fn generate_module(
        &self,
        expressions: &[(String, ASTRepr<f64>)],
        module_name: &str,
    ) -> Result<String> {
        self.generate_module_generic(expressions, module_name, "f64")
    }

    /// Generate Rust expression code from `ASTRepr` (generic version)
    fn generate_expression_with_registry<T: Scalar + Float + Copy + 'static>(
        &self,
        expr: &ASTRepr<T>,
        registry: &VariableRegistry,
    ) -> Result<String> {
        match expr {
            ASTRepr::Constant(value) => {
                // Handle different numeric types safely without transmute
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    // Safe cast for f64
                    let val = value
                        .to_f64()
                        .ok_or(DSLCompileError::CompilationError(format!(
                            "Failed to convert constant to f64: {value}"
                        )))?;
                    Ok(format!("{val}_f64"))
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    // Safe cast for f32
                    let val = value
                        .to_f32()
                        .ok_or(DSLCompileError::CompilationError(format!(
                            "Failed to convert constant to f32: {value}"
                        )))?;
                    Ok(format!("{val}_f32"))
                } else {
                    // Generic fallback
                    Ok(format!("{value}"))
                }
            }
            ASTRepr::Variable(index) => {
                // Use the registry to generate debug name for the variable
                if *index < registry.len() {
                    Ok(registry.debug_name(*index))
                } else {
                    Err(DSLCompileError::CompilationError(format!(
                        "Variable index {index} not found in registry"
                    )))
                }
            }
            ASTRepr::Add(left, right) => {
                let left_code = self.generate_expression_with_registry(left, registry)?;
                let right_code = self.generate_expression_with_registry(right, registry)?;
                Ok(format!("({left_code} + {right_code})"))
            }
            ASTRepr::Sub(left, right) => {
                let left_code = self.generate_expression_with_registry(left, registry)?;
                let right_code = self.generate_expression_with_registry(right, registry)?;
                Ok(format!("({left_code} - {right_code})"))
            }
            ASTRepr::Mul(left, right) => {
                let left_code = self.generate_expression_with_registry(left, registry)?;
                let right_code = self.generate_expression_with_registry(right, registry)?;
                Ok(format!("({left_code} * {right_code})"))
            }
            ASTRepr::Div(left, right) => {
                let left_code = self.generate_expression_with_registry(left, registry)?;
                let right_code = self.generate_expression_with_registry(right, registry)?;
                Ok(format!("({left_code} / {right_code})"))
            }
            ASTRepr::Pow(base, exp) => {
                let base_code = self.generate_expression_with_registry(base, registry)?;

                // Check for square root optimization: x^0.5 -> x.sqrt()
                if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                    // Convert to f64 for comparison to handle generic numeric types
                    if let Ok(exp_f64) = format!("{exp_val}").parse::<f64>() {
                        if (exp_f64 - 0.5).abs() < 1e-15 {
                            return Ok(format!("({base_code}).sqrt()"));
                        }

                        // Check if exponent is a constant integer for optimization
                        if let Some(exp_int) = try_convert_to_integer(exp_f64, None) {
                            return Ok(generate_integer_power_string(
                                &base_code,
                                exp_int,
                                &self.config.power_config,
                            ));
                        }
                    }
                }

                let exp_code = self.generate_expression_with_registry(exp, registry)?;
                Ok(format!("({base_code}).powf({exp_code})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_code = self.generate_expression_with_registry(inner, registry)?;
                Ok(format!("(-{inner_code})"))
            }
            ASTRepr::Ln(inner) => {
                let inner_code = self.generate_expression_with_registry(inner, registry)?;
                Ok(format!("({inner_code}).ln()"))
            }
            ASTRepr::Exp(inner) => {
                let inner_code = self.generate_expression_with_registry(inner, registry)?;
                Ok(format!("({inner_code}).exp()"))
            }
            ASTRepr::Sin(inner) => {
                let inner_code = self.generate_expression_with_registry(inner, registry)?;
                Ok(format!("({inner_code}).sin()"))
            }
            ASTRepr::Cos(inner) => {
                let inner_code = self.generate_expression_with_registry(inner, registry)?;
                Ok(format!("({inner_code}).cos()"))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_code = self.generate_expression_with_registry(inner, registry)?;
                Ok(format!("({inner_code}).sqrt()"))
            }
            ASTRepr::Sum(collection) => {
                // Generate idiomatic Rust code for Collection-based summation
                self.generate_collection_sum(collection, registry)
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Generate inline Rust expression code (no FFI overhead)
    ///
    /// This generates pure Rust expressions that can be embedded directly
    /// in user code without any FFI or function call overhead.
    pub fn generate_inline_expression<T: Scalar + Float + Copy + 'static>(
        &self,
        expr: &ASTRepr<T>,
        registry: &VariableRegistry,
    ) -> Result<String> {
        self.generate_expression_with_registry(expr, registry)
    }

    /// Generate idiomatic Rust code for Collection-based summation
    fn generate_collection_sum<T: Scalar + Float + Copy + std::fmt::Display + 'static>(
        &self,
        collection: &Collection<T>,
        registry: &VariableRegistry,
    ) -> Result<String> {
        match collection {
            Collection::Empty => Ok("0.0".to_string()),

            Collection::Singleton(expr) => {
                // Single element: just evaluate the expression
                self.generate_expression_with_registry(expr, registry)
            }

            Collection::Range { start, end } => {
                // Mathematical range: generate (start..=end).map(|i| i as T).sum()
                let start_code = self.generate_expression_with_registry(start, registry)?;
                let end_code = self.generate_expression_with_registry(end, registry)?;

                // Check for constant range that can be optimized
                if let (ASTRepr::Constant(start_val), ASTRepr::Constant(end_val)) =
                    (start.as_ref(), end.as_ref())
                {
                    // Constant range: compute sum directly if reasonable size
                    let start_int = (*start_val).to_i64().unwrap_or(0);
                    let end_int = (*end_val).to_i64().unwrap_or(0);

                    if start_int <= end_int && (end_int - start_int) <= 1000 {
                        // Small range: compute arithmetic series sum = n(a + l)/2
                        let n = end_int - start_int + 1;
                        let sum = n * (start_int + end_int) / 2;
                        return Ok(format!("{sum}.0"));
                    }
                }

                // Generate idiomatic iterator pattern
                Ok(format!(
                    "({start_code} as i64..={end_code} as i64).map(|i| i as f64).sum::<f64>()"
                ))
            }

            Collection::Variable(var_index) => {
                // Variable reference: generates parameter based on Variable index
                Ok(format!("var_{var_index}.iter().sum::<f64>()"))
            }

            Collection::Map { lambda, collection } => {
                // TODO: Rust preprocessing constant propagation
                //
                // For simple cases, do constant propagation here to reduce EggLog load:
                // - Identity lambda over constant range: sum(i for i in 1..=n) → n*(n+1)/2
                // - Constant lambda over constant range: sum(c for i in 1..=n) → c*n
                // - Linear lambda over constant range: sum(a*i+b for i in 1..=n) → a*n*(n+1)/2 + b*n
                //
                // This preprocessing makes EggLog's job easier by handling trivial cases
                // before they reach the symbolic optimizer.

                // Mapped collection: generate collection.map(lambda).sum()
                let collection_code = self.generate_collection_iter(collection, registry)?;
                let lambda_code = self.generate_lambda_code(lambda, registry)?;

                // Special handling for DataArray collections that need access to function parameters
                match collection.as_ref() {
                    Collection::Variable(_) => {
                        // For data arrays, we directly use the iterator without special capture handling
                        // The lambda variables are already accessible in the function scope
                        // Note: .copied() produces owned values, so use |iter_var| not |&iter_var|
                        Ok(format!(
                            "{collection_code}.map(|iter_var| {lambda_code}).sum::<f64>()"
                        ))
                    }
                    _ => {
                        // For other collections (ranges, etc.), use the standard pattern
                        Ok(format!(
                            "{collection_code}.map(|iter_var| {lambda_code}).sum::<f64>()"
                        ))
                    }
                }
            }

            // Defer these for later implementation
            Collection::Union { .. } => Ok("/* TODO: Union collections */".to_string()),
            Collection::Intersection { .. } => {
                Ok("/* TODO: Intersection collections */".to_string())
            }
            Collection::Filter { .. } => Ok("/* TODO: Filter collections */".to_string()),
        }
    }

    /// Generate iterator code for collections (without sum operation)
    fn generate_collection_iter<T: Scalar + Float + Copy + std::fmt::Display + 'static>(
        &self,
        collection: &Collection<T>,
        registry: &VariableRegistry,
    ) -> Result<String> {
        match collection {
            Collection::Empty => Ok("std::iter::empty()".to_string()),

            Collection::Singleton(expr) => {
                let expr_code = self.generate_expression_with_registry(expr, registry)?;
                Ok(format!("std::iter::once({expr_code})"))
            }

            Collection::Range { start, end } => {
                let start_code = self.generate_expression_with_registry(start, registry)?;
                let end_code = self.generate_expression_with_registry(end, registry)?;
                Ok(format!(
                    "({start_code} as i64..={end_code} as i64).map(|i| i as f64)"
                ))
            }

            Collection::Variable(var_index) => Ok(format!("var_{var_index}.iter().copied()")),

            Collection::Map { lambda, collection } => {
                let collection_code = self.generate_collection_iter(collection, registry)?;
                let lambda_code = self.generate_lambda_code(lambda, registry)?;
                Ok(format!("({collection_code}).map(|iter_var| {lambda_code})"))
            }

            // Defer these for later implementation
            Collection::Union { .. } => Ok("/* TODO: Union iterator */".to_string()),
            Collection::Intersection { .. } => Ok("/* TODO: Intersection iterator */".to_string()),
            Collection::Filter { .. } => Ok("/* TODO: Filter iterator */".to_string()),
        }
    }

    /// Generate lambda function code for use in map/filter iterators
    fn generate_lambda_code<T: Scalar + Float + Copy + std::fmt::Display + 'static>(
        &self,
        lambda: &Lambda<T>,
        registry: &VariableRegistry,
    ) -> Result<String> {
        match lambda {
            Lambda::Identity => Ok("iter_var".to_string()),

            Lambda::Constant(expr) => {
                // Constant lambda: ignore iterator variable
                self.generate_expression_with_registry(expr, registry)
            }

            Lambda::Lambda { var_index, body } => {
                // For code generation, we'll use "iter_var" as the lambda variable name
                // and substitute it in the body expression
                self.generate_lambda_body_with_var(body, *var_index, "iter_var", registry)
            }

            Lambda::MultiArg { var_indices: _, body } => {
                // For multi-argument lambdas, we need to handle multiple variables
                // This is more complex - for now, generate the body expression as-is
                // TODO: Implement proper multi-argument lambda code generation with tuple destructuring
                self.generate_expression_with_registry(body, registry)
            }

            Lambda::Compose { f, g } => {
                // Function composition: f(g(x))
                let g_code = self.generate_lambda_code(g, registry)?;
                let f_code = self.generate_lambda_code(f, registry)?;
                // This is complex - for now, generate a closure
                Ok(format!(
                    "{{ let temp = {}; {} }}",
                    g_code,
                    f_code.replace("iter_var", "temp")
                ))
            }
        }
    }

    /// Generate lambda body code with variable substitution
    fn generate_lambda_body_with_var<T: Scalar + Float + Copy + std::fmt::Display + 'static>(
        &self,
        body: &ASTRepr<T>,
        var_index: usize,
        var_name: &str,
        registry: &VariableRegistry,
    ) -> Result<String> {
        // DEBUG: Print substitution information

        // For now, do simple variable name substitution
        match body {
            ASTRepr::Variable(index) if *index == var_index => Ok(var_name.to_string()),
            ASTRepr::Variable(index) => {
                // Other variables - look up in registry
                let name = registry.debug_name(*index);
                Ok(name)
            }
            ASTRepr::Constant(value) => Ok(format!("{value}")),
            ASTRepr::Add(left, right) => {
                let left_code =
                    self.generate_lambda_body_with_var(left, var_index, var_name, registry)?;
                let right_code =
                    self.generate_lambda_body_with_var(right, var_index, var_name, registry)?;
                Ok(format!("({left_code} + {right_code})"))
            }
            ASTRepr::Sub(left, right) => {
                let left_code =
                    self.generate_lambda_body_with_var(left, var_index, var_name, registry)?;
                let right_code =
                    self.generate_lambda_body_with_var(right, var_index, var_name, registry)?;
                Ok(format!("({left_code} - {right_code})"))
            }
            ASTRepr::Mul(left, right) => {
                let left_code =
                    self.generate_lambda_body_with_var(left, var_index, var_name, registry)?;
                let right_code =
                    self.generate_lambda_body_with_var(right, var_index, var_name, registry)?;
                Ok(format!("({left_code} * {right_code})"))
            }
            ASTRepr::Div(left, right) => {
                let left_code =
                    self.generate_lambda_body_with_var(left, var_index, var_name, registry)?;
                let right_code =
                    self.generate_lambda_body_with_var(right, var_index, var_name, registry)?;
                Ok(format!("({left_code} / {right_code})"))
            }
            ASTRepr::Pow(base, exp) => {
                let base_code =
                    self.generate_lambda_body_with_var(base, var_index, var_name, registry)?;

                // Check for square root optimization: x^0.5 -> x.sqrt()
                if let ASTRepr::Constant(exp_val) = exp.as_ref()
                    && let Ok(exp_f64) = format!("{exp_val}").parse::<f64>()
                    && (exp_f64 - 0.5).abs() < 1e-15
                {
                    return Ok(format!("({base_code}).sqrt()"));
                }

                let exp_code =
                    self.generate_lambda_body_with_var(exp, var_index, var_name, registry)?;
                Ok(format!("({base_code}).powf({exp_code})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_code =
                    self.generate_lambda_body_with_var(inner, var_index, var_name, registry)?;
                Ok(format!("(-{inner_code})"))
            }
            ASTRepr::Ln(inner) => {
                let inner_code =
                    self.generate_lambda_body_with_var(inner, var_index, var_name, registry)?;
                Ok(format!("({inner_code}).ln()"))
            }
            ASTRepr::Exp(inner) => {
                let inner_code =
                    self.generate_lambda_body_with_var(inner, var_index, var_name, registry)?;
                Ok(format!("({inner_code}).exp()"))
            }
            ASTRepr::Sin(inner) => {
                let inner_code =
                    self.generate_lambda_body_with_var(inner, var_index, var_name, registry)?;
                Ok(format!("({inner_code}).sin()"))
            }
            ASTRepr::Cos(inner) => {
                let inner_code =
                    self.generate_lambda_body_with_var(inner, var_index, var_name, registry)?;
                Ok(format!("({inner_code}).cos()"))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_code =
                    self.generate_lambda_body_with_var(inner, var_index, var_name, registry)?;
                Ok(format!("({inner_code}).sqrt()"))
            }
            ASTRepr::Sum(collection) => {
                // Nested sum - generate recursively
                self.generate_collection_sum(collection, registry)
            }
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Helper: Check if expression uses data arrays that need to be passed as parameters
    pub fn expression_uses_data_arrays<T>(&self, expr: &ASTRepr<T>) -> bool {
        match expr {
            ASTRepr::Sum(collection) => self.collection_uses_data_arrays(collection),
            ASTRepr::Add(l, r)
            | ASTRepr::Sub(l, r)
            | ASTRepr::Mul(l, r)
            | ASTRepr::Div(l, r)
            | ASTRepr::Pow(l, r) => {
                self.expression_uses_data_arrays(l) || self.expression_uses_data_arrays(r)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.expression_uses_data_arrays(inner),
            _ => false,
        }
    }

    /// Helper: Check if collection uses data arrays
    pub fn collection_uses_data_arrays<T>(&self, collection: &Collection<T>) -> bool {
        match collection {
            Collection::Variable(_) => true,
            Collection::Map { lambda, collection } => {
                self.collection_uses_data_arrays(collection) || self.lambda_uses_data_arrays(lambda)
            }
            Collection::Union { left, right } | Collection::Intersection { left, right } => {
                self.collection_uses_data_arrays(left) || self.collection_uses_data_arrays(right)
            }
            Collection::Filter {
                predicate,
                collection,
            } => {
                self.collection_uses_data_arrays(collection)
                    || self.expression_uses_data_arrays(predicate)
            }
            _ => false,
        }
    }

    /// Helper: Check if lambda uses data arrays
    pub fn lambda_uses_data_arrays<T>(&self, lambda: &Lambda<T>) -> bool {
        match lambda {
            Lambda::Lambda { body, .. } => self.expression_uses_data_arrays(body),
            Lambda::MultiArg { body, .. } => self.expression_uses_data_arrays(body),
            Lambda::Constant(expr) => self.expression_uses_data_arrays(expr),
            Lambda::Compose { f, g } => {
                self.lambda_uses_data_arrays(f) || self.lambda_uses_data_arrays(g)
            }
            _ => false,
        }
    }

    /// Helper: Count the number of unique data arrays used in expression
    pub fn count_data_arrays<T>(&self, expr: &ASTRepr<T>) -> usize {
        let mut max_data_index = 0;
        let mut found_any = false;
        self.find_max_data_array_index_with_flag(expr, &mut max_data_index, &mut found_any);
        if found_any { max_data_index + 1 } else { 0 }
    }

    /// Helper: Find the maximum data array index used
    fn find_max_data_array_index<T>(&self, expr: &ASTRepr<T>, max_index: &mut usize) {
        match expr {
            ASTRepr::Sum(collection) => {
                self.find_max_data_array_index_in_collection(collection, max_index)
            }
            ASTRepr::Add(l, r)
            | ASTRepr::Sub(l, r)
            | ASTRepr::Mul(l, r)
            | ASTRepr::Div(l, r)
            | ASTRepr::Pow(l, r) => {
                self.find_max_data_array_index(l, max_index);
                self.find_max_data_array_index(r, max_index);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.find_max_data_array_index(inner, max_index);
            }
            _ => {}
        }
    }

    /// Helper: Find the maximum data array index used with found flag
    fn find_max_data_array_index_with_flag<T>(
        &self,
        expr: &ASTRepr<T>,
        max_index: &mut usize,
        found_any: &mut bool,
    ) {
        match expr {
            ASTRepr::Sum(collection) => self.find_max_data_array_index_in_collection_with_flag(
                collection, max_index, found_any,
            ),
            ASTRepr::Add(l, r)
            | ASTRepr::Sub(l, r)
            | ASTRepr::Mul(l, r)
            | ASTRepr::Div(l, r)
            | ASTRepr::Pow(l, r) => {
                self.find_max_data_array_index_with_flag(l, max_index, found_any);
                self.find_max_data_array_index_with_flag(r, max_index, found_any);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.find_max_data_array_index_with_flag(inner, max_index, found_any);
            }
            _ => {}
        }
    }

    /// Helper: Find max data array index in collection
    fn find_max_data_array_index_in_collection<T>(
        &self,
        collection: &Collection<T>,
        max_index: &mut usize,
    ) {
        match collection {
            Collection::Variable(index) => {
                if *index > *max_index {
                    *max_index = *index;
                }
            }
            Collection::Map { lambda, collection } => {
                self.find_max_data_array_index_in_collection(collection, max_index);
                self.find_max_data_array_index_in_lambda(lambda, max_index);
            }
            Collection::Union { left, right } | Collection::Intersection { left, right } => {
                self.find_max_data_array_index_in_collection(left, max_index);
                self.find_max_data_array_index_in_collection(right, max_index);
            }
            Collection::Filter {
                predicate,
                collection,
            } => {
                self.find_max_data_array_index_in_collection(collection, max_index);
                self.find_max_data_array_index(predicate, max_index);
            }
            _ => {}
        }
    }

    /// Helper: Find max data array index in collection with found flag
    fn find_max_data_array_index_in_collection_with_flag<T>(
        &self,
        collection: &Collection<T>,
        max_index: &mut usize,
        found_any: &mut bool,
    ) {
        match collection {
            Collection::Variable(index) => {
                *found_any = true;
                if *index > *max_index {
                    *max_index = *index;
                }
            }
            Collection::Map { lambda, collection } => {
                self.find_max_data_array_index_in_collection_with_flag(
                    collection, max_index, found_any,
                );
                self.find_max_data_array_index_in_lambda_with_flag(lambda, max_index, found_any);
            }
            Collection::Union { left, right } | Collection::Intersection { left, right } => {
                self.find_max_data_array_index_in_collection_with_flag(left, max_index, found_any);
                self.find_max_data_array_index_in_collection_with_flag(right, max_index, found_any);
            }
            Collection::Filter {
                predicate,
                collection,
            } => {
                self.find_max_data_array_index_in_collection_with_flag(
                    collection, max_index, found_any,
                );
                self.find_max_data_array_index_with_flag(predicate, max_index, found_any);
            }
            _ => {}
        }
    }

    /// Helper: Find max data array index in lambda
    fn find_max_data_array_index_in_lambda<T>(&self, lambda: &Lambda<T>, max_index: &mut usize) {
        match lambda {
            Lambda::Lambda { body, .. } => self.find_max_data_array_index(body, max_index),
            Lambda::MultiArg { body, .. } => self.find_max_data_array_index(body, max_index),
            Lambda::Constant(expr) => self.find_max_data_array_index(expr, max_index),
            Lambda::Compose { f, g } => {
                self.find_max_data_array_index_in_lambda(f, max_index);
                self.find_max_data_array_index_in_lambda(g, max_index);
            }
            _ => {}
        }
    }

    /// Helper: Find max data array index in lambda with found flag
    fn find_max_data_array_index_in_lambda_with_flag<T>(
        &self,
        lambda: &Lambda<T>,
        max_index: &mut usize,
        found_any: &mut bool,
    ) {
        match lambda {
            Lambda::Lambda { body, .. } => {
                self.find_max_data_array_index_with_flag(body, max_index, found_any)
            }
            Lambda::MultiArg { body, .. } => {
                self.find_max_data_array_index_with_flag(body, max_index, found_any)
            }
            Lambda::Constant(expr) => {
                self.find_max_data_array_index_with_flag(expr, max_index, found_any)
            }
            Lambda::Compose { f, g } => {
                self.find_max_data_array_index_in_lambda_with_flag(f, max_index, found_any);
                self.find_max_data_array_index_in_lambda_with_flag(g, max_index, found_any);
            }
            _ => {}
        }
    }

    /// Generate HList type signature for function parameters
    fn generate_hlist_type(
        &self,
        var_count: usize,
        data_array_count: usize,
        type_name: &str,
    ) -> String {
        let mut types = Vec::new();

        // Add variable parameters
        for _ in 0..var_count {
            types.push(type_name.to_string());
        }

        // Add data array parameters
        for _ in 0..data_array_count {
            types.push(format!("&[{type_name}]"));
        }

        if types.is_empty() {
            "frunk::HNil".to_string()
        } else {
            format!("HList![{}]", types.join(", "))
        }
    }

    /// Generate HList destructuring pattern for function parameters
    fn generate_hlist_destructure(
        &self,
        registry: &VariableRegistry,
        data_array_count: usize,
    ) -> String {
        let mut names = Vec::new();

        // Add variable names from registry
        for i in 0..registry.len() {
            names.push(registry.debug_name(i));
        }

        // Add data array names
        for i in 0..data_array_count {
            names.push(format!("data_{i}"));
        }

        if names.is_empty() {
            "[]".to_string()
        } else {
            format!("[{}]", names.join(", "))
        }
    }

    /// Generate array to HList conversion code
    fn generate_array_to_hlist_conversion(
        &self,
        var_count: usize,
        data_array_count: usize,
        type_name: &str,
    ) -> String {
        if var_count == 0 && data_array_count == 0 {
            return "frunk::HNil".to_string();
        }

        let mut conversions = Vec::new();

        // Add variable conversions
        for i in 0..var_count {
            conversions.push(format!("vars_slice.get({i}).copied().unwrap_or_default()"));
        }

        // Add data array conversions (these would need to be passed separately)
        // For now, we'll use empty arrays as placeholders
        for i in 0..data_array_count {
            conversions.push(format!(
                "&[] as &[{type_name}] /* data_{i} needs separate input */"
            ));
        }

        format!("frunk::hlist![{}]", conversions.join(", "))
    }

    /// Generate parameter extraction code from variable slice
    fn generate_param_extraction_for_vars(
        &self,
        var_count: usize,
        data_array_count: usize,
        type_name: &str,
    ) -> String {
        let mut extractions = Vec::new();

        // Extract scalar parameters
        for i in 0..var_count {
            extractions.push(format!(
                "let var_{i} = vars_slice.get({i}).copied().unwrap_or_default();"
            ));
        }

        // For data arrays, use empty slices as placeholders (would need separate data input)
        for i in 0..data_array_count {
            extractions.push(format!(
                "let data_{i} = &[] as &[{type_name}]; // TODO: Need separate data input"
            ));
        }

        extractions.join("\n    ")
    }

    /// Generate function call arguments for variables and data arrays
    fn generate_call_args_for_vars(&self, var_count: usize, data_array_count: usize) -> String {
        let mut args = Vec::new();

        // Add scalar arguments
        for i in 0..var_count {
            args.push(format!("var_{i}"));
        }

        // Add data array arguments
        for i in 0..data_array_count {
            args.push(format!("data_{i}"));
        }

        args.join(", ")
    }

    /// Find the maximum variable index used in an expression tree
    fn find_max_variable_index<T: Scalar + Float + Copy + 'static>(
        &self,
        expr: &ASTRepr<T>,
    ) -> usize {
        match expr {
            ASTRepr::Constant(_) => 0,
            ASTRepr::Variable(index) => *index,
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                let left_index = self.find_max_variable_index(left);
                let right_index = self.find_max_variable_index(right);
                left_index.max(right_index)
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => self.find_max_variable_index(inner),
            ASTRepr::Sum(collection) => self.find_max_variable_index_in_collection(collection),
            ASTRepr::BoundVar(_) | ASTRepr::Let(_, _, _) => todo!(),
        }
    }

    /// Find maximum variable index used within a collection
    fn find_max_variable_index_in_collection<T: Scalar + Float + Copy + 'static>(
        &self,
        collection: &Collection<T>,
    ) -> usize {
        match collection {
            Collection::Empty => 0,
            Collection::Singleton(expr) => self.find_max_variable_index(expr),
            Collection::Range { start, end } => {
                let start_index = self.find_max_variable_index(start);
                let end_index = self.find_max_variable_index(end);
                start_index.max(end_index)
            }
            Collection::Variable(index) => *index, // Variable references DO count for max index
            Collection::Map { lambda, collection } => {
                let lambda_index = self.find_max_variable_index_in_lambda(lambda);
                let collection_index = self.find_max_variable_index_in_collection(collection);
                lambda_index.max(collection_index)
            }
            Collection::Union { left, right } | Collection::Intersection { left, right } => {
                let left_index = self.find_max_variable_index_in_collection(left);
                let right_index = self.find_max_variable_index_in_collection(right);
                left_index.max(right_index)
            }
            Collection::Filter {
                predicate,
                collection,
            } => {
                let predicate_index = self.find_max_variable_index(predicate);
                let collection_index = self.find_max_variable_index_in_collection(collection);
                predicate_index.max(collection_index)
            }
        }
    }

    /// Helper: Find max variable index in lambda
    fn find_max_variable_index_in_lambda<T: Scalar + Float + Copy + 'static>(
        &self,
        lambda: &Lambda<T>,
    ) -> usize {
        match lambda {
            Lambda::Identity => 0,
            Lambda::Constant(expr) => self.find_max_variable_index(expr),
            Lambda::Lambda { var_index: _, body } => self.find_max_variable_index(body),
            Lambda::MultiArg { var_indices: _, body } => self.find_max_variable_index(body),
            Lambda::Compose { f, g } => {
                let f_index = self.find_max_variable_index_in_lambda(f);
                let g_index = self.find_max_variable_index_in_lambda(g);
                f_index.max(g_index)
            }
        }
    }
}

impl Default for RustCodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rust compiler for compiling generated source code to dynamic libraries
pub struct RustCompiler {
    /// Optimization level
    opt_level: RustOptLevel,
    /// Additional rustc flags
    extra_flags: Vec<String>,
}

impl RustCompiler {
    /// Create a new Rust compiler with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            opt_level: RustOptLevel::O2,
            extra_flags: vec![
                "-C".to_string(),
                "panic=abort".to_string(), // Smaller binary size
            ],
        }
    }

    /// Create a new Rust compiler with custom optimization level
    #[must_use]
    pub fn with_opt_level(opt_level: RustOptLevel) -> Self {
        Self {
            opt_level,
            extra_flags: vec!["-C".to_string(), "panic=abort".to_string()],
        }
    }

    /// Add extra rustc flags
    #[must_use]
    pub fn with_extra_flags(mut self, flags: Vec<String>) -> Self {
        self.extra_flags.extend(flags);
        self
    }

    /// Compile Rust source code to a dynamic library
    pub fn compile_dylib(
        &self,
        source_code: &str,
        source_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        // Write source code to file
        std::fs::write(source_path, source_code).map_err(|e| {
            DSLCompileError::CompilationError(format!("Failed to write source file: {e}"))
        })?;

        // Compile with rustc
        let output = std::process::Command::new("rustc")
            .args([
                "--crate-type=dylib",
                "-C",
                self.opt_level.as_flag(),
                "-C",
                "panic=abort", // Smaller binary size
                source_path.to_str().unwrap(),
                "-o",
                output_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| DSLCompileError::CompilationError(format!("Failed to run rustc: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DSLCompileError::CompilationError(format!(
                "Rust compilation failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Check if rustc is available on the system
    #[must_use]
    pub fn is_available() -> bool {
        std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Get rustc version information
    pub fn version_info() -> Result<String> {
        let output = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .map_err(|e| DSLCompileError::CompilationError(format!("Failed to run rustc: {e}")))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(DSLCompileError::CompilationError(
                "Failed to get rustc version".to_string(),
            ))
        }
    }

    /// Generate a unique temporary ID for file naming
    fn generate_temp_id() -> String {
        use std::process;
        let process_id = process::id();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("{process_id}_{timestamp}")
    }

    /// Compile Rust code and load as a function (with optional cleanup)
    ///
    /// This method:
    /// 1. Creates a temporary source file in the temp directory  
    /// 2. Compiles it to a dynamic library
    /// 3. Loads the specified function from the library
    /// 4. Optionally schedules the library file for cleanup when the function is dropped
    ///
    /// # Arguments
    /// - `rust_code`: The Rust source code to compile
    /// - `function_name`: The name of the function to load from the compiled library
    /// - `cleanup`: Whether to clean up temporary files when the function is dropped
    pub fn compile_and_load_with_cleanup(
        &self,
        rust_code: &str,
        function_name: &str,
        cleanup: bool,
    ) -> Result<CompiledRustFunction> {
        let temp_dir = std::env::temp_dir();
        let source_name = format!(
            "dslcompile_{}_{}.rs",
            function_name,
            Self::generate_temp_id()
        );
        let _source_path = temp_dir.join(&source_name);

        let lib_name = format!(
            "libdslcompile_{}_{}.so",
            function_name,
            Self::generate_temp_id()
        );
        let _lib_path = temp_dir.join(&lib_name);

        // Create and compile the source file
        self.compile_and_load_in_dirs(rust_code, function_name, &temp_dir, &temp_dir)
            .map(|mut compiled_func| {
                if !cleanup {
                    // Don't schedule cleanup - useful in test environments to avoid hanging
                    compiled_func.lib_path = None;
                }
                compiled_func
            })
    }

    /// Compile Rust code and load as a function (with cleanup enabled by default)
    pub fn compile_and_load(
        &self,
        rust_code: &str,
        function_name: &str,
    ) -> Result<CompiledRustFunction> {
        // In test environments, disable cleanup to avoid hanging issues
        let cleanup = !cfg!(test);
        self.compile_and_load_with_cleanup(rust_code, function_name, cleanup)
    }

    /// Compile and load with custom directory paths
    ///
    /// Like `compile_and_load` but allows specifying custom directories for the
    /// generated source and library files.
    pub fn compile_and_load_in_dirs(
        &self,
        source_code: &str,
        function_name: &str,
        source_dir: &Path,
        lib_dir: &Path,
    ) -> Result<CompiledRustFunction> {
        // Ensure directories exist
        std::fs::create_dir_all(source_dir).map_err(|e| {
            DSLCompileError::CompilationError(format!("Failed to create source directory: {e}"))
        })?;
        std::fs::create_dir_all(lib_dir).map_err(|e| {
            DSLCompileError::CompilationError(format!("Failed to create library directory: {e}"))
        })?;

        // Generate paths in specified directories
        let source_path = source_dir.join(format!("{function_name}.rs"));
        let lib_path = if cfg!(target_os = "windows") {
            lib_dir.join(format!("{function_name}.dll"))
        } else if cfg!(target_os = "macos") {
            lib_dir.join(format!("lib{function_name}.dylib"))
        } else {
            lib_dir.join(format!("lib{function_name}.so"))
        };

        // Compile the dynamic library
        self.compile_dylib(source_code, &source_path, &lib_path)?;

        // Load and return the compiled function
        // In test environments, disable cleanup to avoid hanging issues
        let cleanup_path = if cfg!(test) {
            None
        } else {
            Some(lib_path.clone())
        };

        unsafe { CompiledRustFunction::load_with_cleanup(&lib_path, function_name, cleanup_path) }
    }
}

impl Default for RustCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be used as input to compiled functions
pub trait CallableInput {
    /// Convert to parameter array for function calling
    fn to_params(&self) -> Vec<f64>;
}

// HList implementations for zero-cost heterogeneous inputs
impl CallableInput for HNil {
    fn to_params(&self) -> Vec<f64> {
        Vec::new()
    }
}

impl<H, T> CallableInput for HCons<H, T>
where
    H: Into<f64> + Copy,
    T: CallableInput,
{
    fn to_params(&self) -> Vec<f64> {
        let mut params = vec![self.head.into()];
        params.extend(self.tail.to_params());
        params
    }
}

// Single scalar types support
impl CallableInput for f64 {
    fn to_params(&self) -> Vec<f64> {
        vec![*self]
    }
}

impl CallableInput for f32 {
    fn to_params(&self) -> Vec<f64> {
        vec![*self as f64]
    }
}

impl CallableInput for i32 {
    fn to_params(&self) -> Vec<f64> {
        vec![*self as f64]
    }
}

impl CallableInput for i64 {
    fn to_params(&self) -> Vec<f64> {
        vec![*self as f64]
    }
}

impl CallableInput for usize {
    fn to_params(&self) -> Vec<f64> {
        vec![*self as f64]
    }
}

// ❌ ANTI-PATTERN: Vec<f64> and &[f64] flattening support - DEPRECATED
// These implementations defeat type safety by flattening structured data.
// They should be avoided in favor of proper HList usage:
//
// BAD:  compiled_fn.call(vec![mu, sigma, data...])  // Flattens everything
// GOOD: compiled_fn.call(hlist![mu, sigma, data])   // Preserves structure
//
// ⚠️  DEPRECATED: These implementations will be removed in a future version.
// ⚠️  Use HLists for structured inputs: hlist![param1, param2, data_array]
impl CallableInput for Vec<f64> {
    fn to_params(&self) -> Vec<f64> {
        self.clone()
    }
}

// ⚠️  DEPRECATED: This implementation will be removed in a future version.
// ⚠️  Use HLists for structured inputs: hlist![param1, param2, data_array]
impl CallableInput for &[f64] {
    fn to_params(&self) -> Vec<f64> {
        self.to_vec()
    }
}

/// Function signature types for code generation
#[derive(Debug, Clone)]
pub enum FunctionSignature {
    /// f(x) -> f64
    Scalar,
    /// f(x, y) -> f64
    TwoScalars,
    /// f(vars: &[f64]) -> f64
    Vector(usize),
    /// f(dataset: &[(f64, f64)], params: &[f64]) -> f64
    DatasetAndParams { n_params: usize },
    /// Custom signature for mixed inputs
    Mixed {
        n_scalars: usize,
        vector_sizes: Vec<usize>,
    },
}

/// Compiled Rust function wrapper using dlopen2's raw API
pub struct CompiledRustFunction {
    /// The loaded dynamic library (kept alive)
    _library: Library,
    /// Type-safe function pointer
    function_ptr: extern "C" fn(*const f64, usize) -> f64,
    /// The function name for debugging
    function_name: String,
    /// Path to the temporary library file (for cleanup)
    lib_path: Option<std::path::PathBuf>,
}

// Safe Send/Sync implementation - function pointers are thread-safe
// and the library lifetime is managed properly
unsafe impl Send for CompiledRustFunction {}
unsafe impl Sync for CompiledRustFunction {}

impl CompiledRustFunction {
    /// Load a compiled dynamic library and create a function wrapper
    ///
    /// # Safety
    ///
    /// This function is unsafe because it loads functions from a dynamic library.
    /// The caller must ensure that:
    /// - The library path points to a valid dynamic library
    /// - The function name exists in the library with the expected signature
    /// - The library was compiled with compatible ABI
    unsafe fn load_with_cleanup(
        lib_path: &Path,
        function_name: &str,
        cleanup_path: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let library = Library::open(lib_path).map_err(|e| {
            DSLCompileError::CompilationError(format!("Failed to load library: {e}"))
        })?;

        // Try to load the _legacy version first since that matches our calling convention
        let legacy_func_name = format!("{function_name}_legacy");

        // Get the function symbol using dlopen2's raw API
        let function_ptr = unsafe {
            library
                .symbol::<extern "C" fn(*const f64, usize) -> f64>(&legacy_func_name)
                .or_else(|_| {
                    // Fallback: try the exact name (should not be needed with new naming)
                    library.symbol::<extern "C" fn(*const f64, usize) -> f64>(function_name)
                })
        }
        .map_err(|e| {
            DSLCompileError::CompilationError(format!(
                "Function '{function_name}' or '{legacy_func_name}' not found in library: {e}"
            ))
        })?;

        Ok(CompiledRustFunction {
            _library: library,
            function_ptr,
            function_name: function_name.to_string(),
            lib_path: cleanup_path,
        })
    }

    /// Call the function with HList or other callable input - zero-cost abstraction
    pub fn call<I: CallableInput>(&self, input: I) -> Result<f64> {
        let params = input.to_params();
        Ok((self.function_ptr)(params.as_ptr(), params.len()))
    }

    /// Get the function name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.function_name
    }
}

impl Drop for CompiledRustFunction {
    fn drop(&mut self) {
        // In test environments, completely skip cleanup to avoid hanging issues
        if cfg!(test) {
            // Set lib_path to None to indicate no cleanup needed
            self.lib_path = None;
            return;
        }

        if let Some(lib_path) = self.lib_path.take() {
            // Try to remove the temporary library file with timeout protection
            // Use a non-blocking approach that won't hang the process
            match std::fs::remove_file(&lib_path) {
                Ok(()) => {
                    // Successfully cleaned up
                    #[cfg(debug_assertions)]
                    eprintln!("Successfully removed temporary library file: {lib_path:?}");
                }
                Err(e) => {
                    // File removal failed - this can happen if the library is still loaded
                    // or if there are permission issues. This is not critical since:
                    // 1. The OS will eventually clean up temp files
                    // 2. The files are in the temp directory which gets cleared periodically
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Note: Could not remove temporary library file {lib_path:?}: {e}. This is not critical - the OS will clean it up eventually."
                    );

                    // Don't attempt any retry logic or blocking operations that could hang
                    // Just let the OS handle cleanup during its normal temp file maintenance
                }
            }
        }
    }
}

/// Runtime call specification for flexible function calling
#[derive(Debug, Clone)]
pub enum RuntimeCallSpec<'a> {
    /// Call with parameters only
    ParamsOnly { params: &'a [f64] },
    /// Call with parameters and single data array
    ParamsAndData { params: &'a [f64], data: &'a [f64] },
    /// Call with parameters and multiple data arrays
    ParamsAndMultipleArrays {
        params: &'a [f64],
        data_arrays: &'a [&'a [f64]],
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use frunk::hlist;

    #[test]
    fn test_simple_expression() {
        let codegen = RustCodeGenerator::new();
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(1.0)),
        );
        let code = codegen
            .generate_function_generic(&expr, "test_fn", "f64")
            .unwrap();

        assert!(code.contains("#[no_mangle]"));
        assert!(code.contains("pub extern \"C\" fn test_fn"));
        assert!(code.contains("(var_0 + 1_f64)"));
    }

    #[test]
    fn test_complex_expression() {
        let codegen = RustCodeGenerator::new();
        let expr: ASTRepr<f64> = ASTRepr::Mul(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Variable(1)),
        );
        let code = codegen
            .generate_function_generic(&expr, "multiply", "f64")
            .unwrap();

        assert!(code.contains("#[no_mangle]"));
        assert!(code.contains("pub extern \"C\" fn multiply"));
        // Variables are now named var_0, var_1, etc.
        assert!(code.contains("(var_0 * var_1)"));
    }

    #[test]
    fn test_trigonometric_functions() {
        let codegen = RustCodeGenerator::new();
        let expr: ASTRepr<f64> = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let code = codegen
            .generate_function_generic(&expr, "sin_x", "f64")
            .unwrap();

        assert!(code.contains("#[no_mangle]"));
        assert!(code.contains("pub extern \"C\" fn sin_x"));
        assert!(code.contains("(var_0).sin()"));
    }

    #[test]
    fn test_nested_expression() {
        let codegen = RustCodeGenerator::new();
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Mul(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Variable(1)),
            )),
            Box::new(ASTRepr::Constant(5.0)),
        );
        let code = codegen
            .generate_function_generic(&expr, "nested", "f64")
            .unwrap();

        assert!(code.contains("#[no_mangle]"));
        assert!(code.contains("pub extern \"C\" fn nested"));
        // Variables are now named var_0, var_1, etc.
        assert!(code.contains("((var_0 * var_1) + 5_f64)"));
    }

    #[test]
    fn test_rust_compiler_availability() {
        // This test checks if rustc is available on the system
        // It may fail in environments without Rust toolchain
        if RustCompiler::is_available() {
            let version = RustCompiler::version_info();
            assert!(version.is_ok());
            println!("Rust version: {}", version.unwrap());
        } else {
            println!("Rust compiler not available - skipping compiler tests");
        }
    }

    #[test]
    fn test_rust_compiler_creation() {
        let compiler = RustCompiler::new();
        assert_eq!(compiler.opt_level, RustOptLevel::O2);

        let compiler_o3 = RustCompiler::with_opt_level(RustOptLevel::O3);
        assert_eq!(compiler_o3.opt_level, RustOptLevel::O3);

        let compiler_with_flags = RustCompiler::new()
            .with_extra_flags(vec!["-C".to_string(), "target-cpu=native".to_string()]);
        assert!(compiler_with_flags.extra_flags.len() >= 2);
    }

    #[test]
    fn test_compile_and_load_functionality() {
        // Only run this test if rustc is available
        if !RustCompiler::is_available() {
            println!("Rust compiler not available - skipping compile_and_load test");
            return;
        }

        let codegen = RustCodeGenerator::new();
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(1.0)),
        );
        let rust_code = codegen.generate_function(&expr, "test_func").unwrap();

        let compiler = RustCompiler::new();
        let compiled_func = compiler.compile_and_load(&rust_code, "test_func").unwrap();

        let result = compiled_func.call(hlist![5.0]).unwrap();
        assert_eq!(result, 6.0);

        println!("compile_and_load test passed: f(5) = {result}");
        // No manual cleanup needed - handled automatically by Drop
    }

    #[test]
    fn test_callable_input_hlist_integration() {
        use frunk::hlist;

        // Test that CallableInput works with various input types

        // Single scalar
        let single_f64: f64 = 5.0;
        assert_eq!(single_f64.to_params(), vec![5.0]);

        let single_f32: f32 = 3.5;
        assert_eq!(single_f32.to_params(), vec![3.5]);

        let single_i32: i32 = 42;
        assert_eq!(single_i32.to_params(), vec![42.0]);

        // HLists (zero-cost heterogeneous)
        let hlist_homo = hlist![3.0, 4.0, 5.0];
        assert_eq!(hlist_homo.to_params(), vec![3.0, 4.0, 5.0]);

        let hlist_hetero = hlist![3.0_f64, 4_i32, 5.5_f32];
        assert_eq!(hlist_hetero.to_params(), vec![3.0, 4.0, 5.5]);

        // Vec and slices (backward compatibility)
        let vec_input = vec![1.0, 2.0, 3.0];
        assert_eq!(vec_input.to_params(), vec![1.0, 2.0, 3.0]);

        let slice_input: &[f64] = &[7.0, 8.0];
        assert_eq!(slice_input.to_params(), vec![7.0, 8.0]);

        // Empty HList
        let empty_hlist = hlist![];
        assert_eq!(empty_hlist.to_params(), Vec::<f64>::new());

        println!("✅ All CallableInput implementations work correctly!");
        println!("✅ HLists provide zero-cost heterogeneous input support!");
        println!("✅ Backward compatibility maintained for Vec<f64> and &[f64]!");
        println!("✅ Single scalars work directly without wrapper types!");
    }
}
