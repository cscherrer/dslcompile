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

use crate::ast::ast_utils::collect_variable_indices;
use crate::error::{MathCompileError, Result};
use crate::final_tagless::{ASTRepr, NumericType, VariableRegistry};
use crate::symbolic::power_utils::{
    PowerOptConfig, generate_integer_power_string, try_convert_to_integer,
};
use dlopen2::raw::Library;
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

    /// Create a new Rust code generator with custom settings (deprecated, use `with_config`)
    #[deprecated(since = "0.1.0", note = "Use with_config instead")]
    #[must_use]
    pub fn with_settings(debug_info: bool, unsafe_optimizations: bool) -> Self {
        Self {
            config: RustCodegenConfig {
                debug_info,
                unsafe_optimizations,
                ..Default::default()
            },
        }
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

    /// Generate Rust source code for a mathematical expression with variable registry (generic version)
    pub fn generate_function_with_registry<T: NumericType + Float + Copy>(
        &self,
        expr: &ASTRepr<T>,
        function_name: &str,
        type_name: &str,
        registry: &VariableRegistry,
    ) -> Result<String> {
        let expr_code = self.generate_expression_with_registry(expr, registry)?;

        // Generate function signature based on variables used
        let param_list: Vec<String> = registry
            .get_all_names()
            .iter()
            .map(|name| format!("{name}: {type_name}"))
            .collect();
        let params = param_list.join(", ");

        // Add optimization attributes based on configuration
        let mut attributes = String::new();
        if self.config.vectorization_hints && type_name == "f64" {
            // Vectorization hints only make sense for f64
            attributes.push_str("#[target_feature(enable = \"avx2\")]\n");
        } else if self.config.aggressive_inlining {
            attributes.push_str("#[inline(always)]\n");
        }

        Ok(format!(
            r#"
{attributes}#[no_mangle]
pub extern "C" fn {function_name}({params}) -> {type_name} {{
    return {expr_code};
}}

{attributes}#[no_mangle]
pub extern "C" fn {function_name}_multi_vars(vars: *const {type_name}, count: usize) -> {type_name} {{
    if vars.is_null() || count == 0 {{
        return Default::default();
    }}
    
    // Extract variables from array based on registry order
    let mut extracted_vars = Vec::new();
    for i in 0..{var_count} {{
        if i < count {{
            extracted_vars.push(unsafe {{ *vars.add(i) }});
        }} else {{
            extracted_vars.push(Default::default());
        }}
    }}
    
    // Call the main function with extracted variables
    {function_name}({extracted_call_params})
}}
"#,
            var_count = registry.len(),
            extracted_call_params = registry
                .get_all_names()
                .iter()
                .enumerate()
                .map(|(i, _)| format!("extracted_vars[{i}]"))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    /// Generate Rust source code for a mathematical expression (generic version)
    pub fn generate_function_generic<T: NumericType + Float + Copy>(
        &self,
        expr: &ASTRepr<T>,
        function_name: &str,
        type_name: &str,
    ) -> Result<String> {
        // Create a default registry with variable indices as names
        let mut default_registry = VariableRegistry::new();
        let variables = collect_variable_indices(expr);

        // Sort variables to ensure deterministic order
        let mut sorted_variables: Vec<usize> = variables.into_iter().collect();
        sorted_variables.sort_unstable();

        // Register variables using their indices as names
        for &var_index in &sorted_variables {
            let var_name = format!("var_{var_index}");
            default_registry.register_variable(&var_name);
        }

        self.generate_function_with_registry(expr, function_name, type_name, &default_registry)
    }

    /// Generate Rust source code for a mathematical expression (f64 specialization for backwards compatibility)
    pub fn generate_function(&self, expr: &ASTRepr<f64>, function_name: &str) -> Result<String> {
        self.generate_function_generic(expr, function_name, "f64")
    }

    /// Generate Rust source code for a complete module (generic version)
    pub fn generate_module_generic<T: NumericType + Float + Copy>(
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
    fn generate_expression_with_registry<T: NumericType + Float + Copy>(
        &self,
        expr: &ASTRepr<T>,
        registry: &VariableRegistry,
    ) -> Result<String> {
        match expr {
            ASTRepr::Constant(value) => {
                // Handle different numeric types safely without transmute
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    // Safe cast for f64
                    let val = value.to_f64().ok_or(MathCompileError::CompilationError(format!("Failed to convert constant to f64: {value}")))?;
                    Ok(format!("{val}_f64"))
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    // Safe cast for f32
                    let val = value.to_f32().ok_or(MathCompileError::CompilationError(format!("Failed to convert constant to f32: {value}")))?;
                    Ok(format!("{val}_f32"))
                } else {
                    // Generic fallback
                    Ok(format!("{value}"))
                }
            }
            ASTRepr::Variable(index) => {
                // Use the registry to map index to variable name
                if let Some(var_name) = registry.get_name(*index) {
                    Ok(var_name.to_string())
                } else {
                    Err(MathCompileError::CompilationError(format!(
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
                let exp_code = self.generate_expression_with_registry(exp, registry)?;

                // Check if exponent is a constant integer for optimization
                if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                    if let Some(exp_int) = try_convert_to_integer(*exp_val, None) {
                        return Ok(generate_integer_power_string(
                            &base_code,
                            exp_int,
                            &self.config.power_config,
                        ));
                    }
                }

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
        }
    }

    /// Generate a function with runtime data binding support
    pub fn generate_runtime_data_function(
        &self,
        expr: &ASTRepr<f64>,
        function_name: &str,
        data_spec: &RuntimeDataSpec,
        partial_eval_context: Option<&PartialEvalContext>,
    ) -> Result<String> {
        // Step 1: Apply partial evaluation if context is provided
        let optimized_expr = if let Some(context) = partial_eval_context {
            self.apply_partial_evaluation(expr, context)?
        } else {
            expr.clone()
        };

        // Step 2: Generate function signature based on data spec
        let signature = self.generate_runtime_signature(function_name, data_spec)?;

        // Step 3: Generate function body with runtime data access
        let body = self.generate_runtime_body(&optimized_expr, data_spec)?;

        // Step 4: Add optimization attributes
        let mut attributes = String::new();
        if self.config.aggressive_inlining {
            attributes.push_str("#[inline(always)]\n");
        }

        Ok(format!(
            r"
{attributes}#[no_mangle]
{signature} {{
{body}
}}
"
        ))
    }

    /// Apply partial evaluation using static values and abstract interpretation
    fn apply_partial_evaluation(
        &self,
        expr: &ASTRepr<f64>,
        context: &PartialEvalContext,
    ) -> Result<ASTRepr<f64>> {
        // This is where we'd integrate with the abstract interpretation infrastructure
        // For now, we'll do basic constant folding with static values
        self.fold_static_constants(expr, &context.static_values)
    }

    /// Fold static constants in the expression
    fn fold_static_constants(
        &self,
        expr: &ASTRepr<f64>,
        static_values: &std::collections::HashMap<String, f64>,
    ) -> Result<ASTRepr<f64>> {
        match expr {
            ASTRepr::Constant(value) => Ok(ASTRepr::Constant(*value)),

            ASTRepr::Variable(index) => {
                // For now, we'll keep variables as-is
                // In a full implementation, we'd map variable names to static values
                Ok(ASTRepr::Variable(*index))
            }

            ASTRepr::Add(left, right) => {
                let left_folded = self.fold_static_constants(left, static_values)?;
                let right_folded = self.fold_static_constants(right, static_values)?;

                match (&left_folded, &right_folded) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a + b)),
                    (ASTRepr::Constant(a), _) if *a == 0.0 => Ok(right_folded),
                    (_, ASTRepr::Constant(b)) if *b == 0.0 => Ok(left_folded),
                    _ => Ok(ASTRepr::Add(Box::new(left_folded), Box::new(right_folded))),
                }
            }

            ASTRepr::Mul(left, right) => {
                let left_folded = self.fold_static_constants(left, static_values)?;
                let right_folded = self.fold_static_constants(right, static_values)?;

                match (&left_folded, &right_folded) {
                    (ASTRepr::Constant(a), ASTRepr::Constant(b)) => Ok(ASTRepr::Constant(a * b)),
                    (ASTRepr::Constant(a), _) if *a == 0.0 => Ok(ASTRepr::Constant(0.0)),
                    (_, ASTRepr::Constant(b)) if *b == 0.0 => Ok(ASTRepr::Constant(0.0)),
                    (ASTRepr::Constant(a), _) if *a == 1.0 => Ok(right_folded),
                    (_, ASTRepr::Constant(b)) if *b == 1.0 => Ok(left_folded),
                    _ => Ok(ASTRepr::Mul(Box::new(left_folded), Box::new(right_folded))),
                }
            }

            // Add other operations as needed
            _ => Ok(expr.clone()),
        }
    }

    /// Generate function signature for runtime data binding
    fn generate_runtime_signature(
        &self,
        function_name: &str,
        data_spec: &RuntimeDataSpec,
    ) -> Result<String> {
        match &data_spec.signature_pattern {
            RuntimeSignature::ParamsOnly { n_params } => Ok(format!(
                "pub extern \"C\" fn {function_name}(params: *const f64, n_params: usize) -> f64"
            )),

            RuntimeSignature::ParamsAndData { n_params } => Ok(format!(
                "pub extern \"C\" fn {function_name}(params: *const f64, n_params: usize, data: *const f64, n_data: usize) -> f64"
            )),

            RuntimeSignature::ParamsAndMultipleArrays { n_params, n_arrays } => Ok(format!(
                "pub extern \"C\" fn {function_name}(params: *const f64, n_params: usize, data_arrays: *const *const f64, data_sizes: *const usize, n_arrays: usize) -> f64"
            )),

            _ => {
                // For now, default to params + data
                Ok(format!(
                    "pub extern \"C\" fn {function_name}(params: *const f64, n_params: usize, data: *const f64, n_data: usize) -> f64"
                ))
            }
        }
    }

    /// Generate function body with runtime data access
    fn generate_runtime_body(
        &self,
        expr: &ASTRepr<f64>,
        data_spec: &RuntimeDataSpec,
    ) -> Result<String> {
        let mut body = String::new();

        // Add safety checks
        body.push_str("    if params.is_null() { return 0.0; }\n");

        // Extract parameters
        for (i, param) in data_spec.runtime_params.iter().enumerate() {
            if let DataBinding::RuntimeScalar { name, param_index } = param {
                body.push_str(&format!(
                    "    let {name} = if {param_index} < n_params {{ unsafe {{ *params.add({param_index}) }} }} else {{ 0.0 }};\n"
                ));
            }
        }

        // Add data array access helpers if needed
        if !data_spec.runtime_data.is_empty() {
            body.push_str("    // Runtime data access implementation needed\n");
        }

        // Generate the main expression evaluation
        let expr_code = self.generate_runtime_expression(expr, data_spec)?;
        body.push_str(&format!("    {expr_code}\n"));

        Ok(body)
    }

    /// Generate expression code with runtime data access
    fn generate_runtime_expression(
        &self,
        expr: &ASTRepr<f64>,
        data_spec: &RuntimeDataSpec,
    ) -> Result<String> {
        match expr {
            ASTRepr::Constant(value) => Ok(format!("{value}_f64")),
            ASTRepr::Variable(index) => {
                // Map variable indices to parameter names from the data spec
                if let Some(param) = data_spec.runtime_params.get(*index) {
                    if let DataBinding::RuntimeScalar { name, .. } = param {
                        Ok(name.clone())
                    } else {
                        Ok(format!("param_{index}"))
                    }
                } else {
                    Ok(format!("param_{index}"))
                }
            }
            ASTRepr::Add(left, right) => {
                let left_code = self.generate_runtime_expression(left, data_spec)?;
                let right_code = self.generate_runtime_expression(right, data_spec)?;
                Ok(format!("({left_code} + {right_code})"))
            }
            ASTRepr::Mul(left, right) => {
                let left_code = self.generate_runtime_expression(left, data_spec)?;
                let right_code = self.generate_runtime_expression(right, data_spec)?;
                Ok(format!("({left_code} * {right_code})"))
            }
            ASTRepr::Sub(left, right) => {
                let left_code = self.generate_runtime_expression(left, data_spec)?;
                let right_code = self.generate_runtime_expression(right, data_spec)?;
                Ok(format!("({left_code} - {right_code})"))
            }
            ASTRepr::Div(left, right) => {
                let left_code = self.generate_runtime_expression(left, data_spec)?;
                let right_code = self.generate_runtime_expression(right, data_spec)?;
                Ok(format!("({left_code} / {right_code})"))
            }
            ASTRepr::Pow(base, exp) => {
                let base_code = self.generate_runtime_expression(base, data_spec)?;
                let exp_code = self.generate_runtime_expression(exp, data_spec)?;
                Ok(format!("{base_code}.powf({exp_code})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_code = self.generate_runtime_expression(inner, data_spec)?;
                Ok(format!("(-{inner_code})"))
            }
            ASTRepr::Ln(inner) => {
                let inner_code = self.generate_runtime_expression(inner, data_spec)?;
                Ok(format!("{inner_code}.ln()"))
            }
            ASTRepr::Exp(inner) => {
                let inner_code = self.generate_runtime_expression(inner, data_spec)?;
                Ok(format!("{inner_code}.exp()"))
            }
            ASTRepr::Sin(inner) => {
                let inner_code = self.generate_runtime_expression(inner, data_spec)?;
                Ok(format!("{inner_code}.sin()"))
            }
            ASTRepr::Cos(inner) => {
                let inner_code = self.generate_runtime_expression(inner, data_spec)?;
                Ok(format!("{inner_code}.cos()"))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_code = self.generate_runtime_expression(inner, data_spec)?;
                Ok(format!("{inner_code}.sqrt()"))
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
            MathCompileError::CompilationError(format!("Failed to write source file: {e}"))
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
            .map_err(|e| MathCompileError::CompilationError(format!("Failed to run rustc: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MathCompileError::CompilationError(format!(
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
            .map_err(|e| MathCompileError::CompilationError(format!("Failed to run rustc: {e}")))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(MathCompileError::CompilationError(
                "Failed to get rustc version".to_string(),
            ))
        }
    }

    /// Compile Rust source code and load it as a dynamic library with auto-generated paths
    ///
    /// This is a convenience method that:
    /// 1. Auto-generates source and library paths from the function name in a temp directory
    /// 2. Compiles the Rust code to a dynamic library
    /// 3. Loads the library and returns a convenient wrapper
    ///
    /// # Arguments
    ///
    /// * `rust_code` - The Rust source code to compile
    /// * `function_name` - The name of the function (used for file naming)
    ///
    /// # Returns
    ///
    /// A `CompiledRustFunction` that can be called directly
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mathcompile::backends::RustCompiler;
    ///
    /// let compiler = RustCompiler::new();
    /// let rust_code = "pub extern \"C\" fn my_func(x: f64) -> f64 { x * 2.0 }";
    /// let compiled = compiler.compile_and_load(rust_code, "my_func")?;
    /// let result = compiled.call(5.0)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compile_and_load(
        &self,
        rust_code: &str,
        function_name: &str,
    ) -> Result<CompiledRustFunction> {
        use std::env;
        use std::process;

        // Create a unique temporary directory for this compilation
        let temp_dir = env::temp_dir();
        let process_id = process::id();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let unique_suffix = format!("{process_id}_{timestamp}");
        let source_filename = format!("{function_name}_{unique_suffix}.rs");
        let lib_name = format!("lib{function_name}_{unique_suffix}");

        let source_path = temp_dir.join(&source_filename);

        // Determine the correct library extension for the platform
        let lib_extension = if cfg!(target_os = "windows") {
            "dll"
        } else if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };

        let lib_filename = format!("{lib_name}.{lib_extension}");
        let lib_path = temp_dir.join(&lib_filename);

        // Compile the code
        self.compile_dylib(rust_code, &source_path, &lib_path)?;

        // Load the library
        let compiled_func = unsafe {
            CompiledRustFunction::load_with_cleanup(
                &lib_path,
                function_name,
                Some(lib_path.clone()),
            )?
        };

        // Clean up source file (keep the library file until the function is dropped)
        let _ = std::fs::remove_file(&source_path);

        Ok(compiled_func)
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
            MathCompileError::CompilationError(format!("Failed to create source directory: {e}"))
        })?;
        std::fs::create_dir_all(lib_dir).map_err(|e| {
            MathCompileError::CompilationError(format!("Failed to create library directory: {e}"))
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
        unsafe {
            CompiledRustFunction::load_with_cleanup(
                &lib_path,
                function_name,
                Some(lib_path.clone()),
            )
        }
    }
}

impl Default for RustCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for compiled functions with flexible input types
pub trait CompiledFunction<Input> {
    type Output;

    /// Call the compiled function with the given input
    fn call(&self, input: Input) -> Result<Self::Output>;

    /// Get the function name for debugging
    fn name(&self) -> &str;
}

/// Trait for describing function input patterns
pub trait InputSpec {
    /// Get a description of this input pattern
    fn description(&self) -> String;

    /// Get the total number of scalar values needed
    fn scalar_count(&self) -> usize;

    /// Get the number of array inputs
    fn array_count(&self) -> usize;

    /// Validate that the given input matches this spec
    fn validate(&self, input: &FunctionInput) -> Result<()>;
}

/// Simple scalar-only input specification
#[derive(Debug, Clone)]
pub struct ScalarInputSpec {
    pub count: usize,
}

impl InputSpec for ScalarInputSpec {
    fn description(&self) -> String {
        format!("ScalarInputSpec({})", self.count)
    }

    fn scalar_count(&self) -> usize {
        self.count
    }

    fn array_count(&self) -> usize {
        0
    }

    fn validate(&self, input: &FunctionInput) -> Result<()> {
        match input {
            FunctionInput::Scalars(scalars) => {
                if scalars.len() == self.count {
                    Ok(())
                } else {
                    Err(MathCompileError::InvalidInput(format!(
                        "Expected {} scalars, got {}",
                        self.count,
                        scalars.len()
                    )))
                }
            }
            _ => Err(MathCompileError::InvalidInput(
                "Expected scalar input".to_string(),
            )),
        }
    }
}

/// Mixed scalar and array input specification
#[derive(Debug, Clone)]
pub struct MixedInputSpec {
    pub scalars: usize,
    pub arrays: Vec<Option<usize>>, // None = dynamic size
}

impl InputSpec for MixedInputSpec {
    fn description(&self) -> String {
        format!(
            "MixedInputSpec(scalars: {}, arrays: {:?})",
            self.scalars, self.arrays
        )
    }

    fn scalar_count(&self) -> usize {
        self.scalars
    }

    fn array_count(&self) -> usize {
        self.arrays.len()
    }

    fn validate(&self, input: &FunctionInput) -> Result<()> {
        match input {
            FunctionInput::Mixed { scalars, arrays } => {
                if scalars.len() != self.scalars {
                    return Err(MathCompileError::InvalidInput(format!(
                        "Expected {} scalars, got {}",
                        self.scalars,
                        scalars.len()
                    )));
                }
                if arrays.len() != self.arrays.len() {
                    return Err(MathCompileError::InvalidInput(format!(
                        "Expected {} arrays, got {}",
                        self.arrays.len(),
                        arrays.len()
                    )));
                }
                // Could add size validation for fixed-size arrays here
                Ok(())
            }
            _ => Err(MathCompileError::InvalidInput(
                "Expected mixed input".to_string(),
            )),
        }
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

/// Runtime input for compiled functions
#[derive(Debug, Clone)]
pub enum FunctionInput<'a> {
    /// Pure scalar inputs
    Scalars(Vec<f64>),
    /// Mixed scalars and arrays
    Mixed {
        scalars: &'a [f64],
        arrays: &'a [&'a [f64]],
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
            MathCompileError::CompilationError(format!("Failed to load library: {e}"))
        })?;

        // Try to load the _multi_vars version first since that's our standard signature
        let multi_var_func_name = format!("{function_name}_multi_vars");

        // Get the function symbol using dlopen2's raw API
        let function_ptr = unsafe {
            library
                .symbol::<extern "C" fn(*const f64, usize) -> f64>(&multi_var_func_name)
                .or_else(|_| {
                    // Fallback: try the exact name
                    library.symbol::<extern "C" fn(*const f64, usize) -> f64>(function_name)
                })
        }
        .map_err(|e| {
            MathCompileError::CompilationError(format!(
                "Function '{function_name}' or '{multi_var_func_name}' not found in library: {e}"
            ))
        })?;

        Ok(CompiledRustFunction {
            _library: library,
            function_ptr,
            function_name: function_name.to_string(),
            lib_path: cleanup_path,
        })
    }

    /// Call the function with flexible input - now type-safe
    pub fn call_with_spec(&self, input: &FunctionInput) -> Result<f64> {
        match input {
            FunctionInput::Scalars(scalars) => {
                // Direct call using the function pointer
                Ok((self.function_ptr)(scalars.as_ptr(), scalars.len()))
            }
            FunctionInput::Mixed { scalars, arrays } => {
                // For mixed inputs, we'd need a more complex calling convention
                if arrays.is_empty() {
                    Ok((self.function_ptr)(scalars.as_ptr(), scalars.len()))
                } else {
                    Err(MathCompileError::CompilationError(
                        "Mixed input types not yet implemented".to_string(),
                    ))
                }
            }
        }
    }

    /// Backward compatibility: Call with single scalar value
    pub fn call(&self, x: f64) -> Result<f64> {
        self.call_with_spec(&FunctionInput::Scalars(vec![x]))
    }

    /// Backward compatibility: Call with two scalar values
    pub fn call_two_vars(&self, x: f64, y: f64) -> Result<f64> {
        self.call_with_spec(&FunctionInput::Scalars(vec![x, y]))
    }

    /// Backward compatibility: Call with multiple variables
    pub fn call_multi_vars(&self, vars: &[f64]) -> Result<f64> {
        self.call_with_spec(&FunctionInput::Scalars(vars.to_vec()))
    }

    /// Get the function name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.function_name
    }

    /// Call function with runtime data binding (params + single data array)
    ///
    /// Note: This requires the compiled function to have the appropriate signature
    pub fn call_with_data(&self, params: &[f64], data: &[f64]) -> Result<f64> {
        self.call_with_spec(&FunctionInput::Scalars(params.to_vec()))
    }

    /// Call function with runtime data binding (params + multiple data arrays)
    pub fn call_with_multiple_data(&self, params: &[f64], data_arrays: &[&[f64]]) -> Result<f64> {
        self.call_with_spec(&FunctionInput::Mixed {
            scalars: params,
            arrays: data_arrays,
        })
    }

    /// Call function with runtime data specification
    pub fn call_with_runtime_spec(&self, spec: &RuntimeCallSpec) -> Result<f64> {
        match spec {
            RuntimeCallSpec::ParamsOnly { params } => self.call_multi_vars(params),
            RuntimeCallSpec::ParamsAndData { params, data } => self.call_with_data(params, data),
            RuntimeCallSpec::ParamsAndMultipleArrays {
                params,
                data_arrays,
            } => self.call_with_multiple_data(params, data_arrays),
        }
    }
}

impl Drop for CompiledRustFunction {
    fn drop(&mut self) {
        if let Some(lib_path) = self.lib_path.take() {
            let _ = std::fs::remove_file(&lib_path);
        }
    }
}

/// Runtime data binding specification for partial evaluation
#[derive(Debug, Clone)]
pub enum DataBinding {
    /// Static parameter (known at compile time, can be partially evaluated)
    Static { name: String, value: f64 },
    /// Runtime scalar parameter (unknown at compile time)
    RuntimeScalar { name: String, param_index: usize },
    /// Runtime array data (unknown at compile time)
    RuntimeArray {
        name: String,
        data_index: usize,
        element_type: DataElementType,
        size_hint: Option<usize>, // None = dynamic size
    },
    /// Runtime matrix data (2D array)
    RuntimeMatrix {
        name: String,
        data_index: usize,
        rows_hint: Option<usize>,
        cols_hint: Option<usize>,
    },
}

/// Types of data elements for runtime arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataElementType {
    /// Single floating point values
    Scalar,
    /// Pairs of values (x, y)
    Pair,
    /// Triples of values (x, y, z)
    Triple,
    /// Custom tuple size
    Tuple(usize),
}

/// Specification for functions that use runtime data binding
#[derive(Debug, Clone)]
pub struct RuntimeDataSpec {
    /// Static parameters that can be partially evaluated
    pub static_params: Vec<DataBinding>,
    /// Runtime parameters (scalars)
    pub runtime_params: Vec<DataBinding>,
    /// Runtime data arrays
    pub runtime_data: Vec<DataBinding>,
    /// Expected function signature pattern
    pub signature_pattern: RuntimeSignature,
}

/// Runtime function signature patterns
#[derive(Debug, Clone)]
pub enum RuntimeSignature {
    /// f(params: &[f64]) -> f64
    ParamsOnly { n_params: usize },
    /// f(params: &[f64], data: &[f64]) -> f64  
    ParamsAndData { n_params: usize },
    /// f(params: &[f64], `data_x`: &[f64], `data_y`: &[f64]) -> f64
    ParamsAndPairedData { n_params: usize },
    /// f(params: &[f64], `data_arrays`: &[&[f64]]) -> f64
    ParamsAndMultipleArrays { n_params: usize, n_arrays: usize },
    /// Custom signature for complex cases
    Custom {
        param_types: Vec<RuntimeParamType>,
        return_type: RuntimeReturnType,
    },
}

/// Types of runtime parameters
#[derive(Debug, Clone)]
pub enum RuntimeParamType {
    Scalar,
    Array {
        element_type: DataElementType,
    },
    Matrix {
        rows: Option<usize>,
        cols: Option<usize>,
    },
}

/// Return types for runtime functions
#[derive(Debug, Clone)]
pub enum RuntimeReturnType {
    Scalar,
    Array {
        size: Option<usize>,
    },
    Matrix {
        rows: Option<usize>,
        cols: Option<usize>,
    },
}

impl RuntimeDataSpec {
    /// Create a simple params-only specification
    #[must_use]
    pub fn params_only(param_names: &[&str]) -> Self {
        let runtime_params = param_names
            .iter()
            .enumerate()
            .map(|(i, name)| DataBinding::RuntimeScalar {
                name: (*name).to_string(),
                param_index: i,
            })
            .collect();

        Self {
            static_params: Vec::new(),
            runtime_params,
            runtime_data: Vec::new(),
            signature_pattern: RuntimeSignature::ParamsOnly {
                n_params: param_names.len(),
            },
        }
    }

    /// Create a params-and-data specification (common for statistical models)
    #[must_use]
    pub fn params_and_data(param_names: &[&str], data_arrays: &[&str]) -> Self {
        let runtime_params = param_names
            .iter()
            .enumerate()
            .map(|(i, name)| DataBinding::RuntimeScalar {
                name: (*name).to_string(),
                param_index: i,
            })
            .collect();

        let runtime_data = data_arrays
            .iter()
            .enumerate()
            .map(|(i, name)| DataBinding::RuntimeArray {
                name: (*name).to_string(),
                data_index: i,
                element_type: DataElementType::Scalar,
                size_hint: None,
            })
            .collect();

        Self {
            static_params: Vec::new(),
            runtime_params,
            runtime_data,
            signature_pattern: RuntimeSignature::ParamsAndMultipleArrays {
                n_params: param_names.len(),
                n_arrays: data_arrays.len(),
            },
        }
    }

    /// Add static parameter for partial evaluation
    #[must_use]
    pub fn with_static_param(mut self, name: &str, value: f64) -> Self {
        self.static_params.push(DataBinding::Static {
            name: name.to_string(),
            value,
        });
        self
    }

    /// Get total number of runtime inputs
    #[must_use]
    pub fn total_runtime_inputs(&self) -> usize {
        self.runtime_params.len() + self.runtime_data.len()
    }
}

/// Partial evaluation context for abstract interpretation
#[derive(Debug, Clone)]
pub struct PartialEvalContext {
    /// Known static values
    pub static_values: std::collections::HashMap<String, f64>,
    /// Abstract domains for runtime parameters
    pub param_domains:
        std::collections::HashMap<String, crate::interval_domain::IntervalDomain<f64>>,
    /// Abstract domains for runtime data arrays
    pub data_domains:
        std::collections::HashMap<String, crate::interval_domain::IntervalDomain<f64>>,
}

impl Default for PartialEvalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEvalContext {
    /// Create a new partial evaluation context
    #[must_use]
    pub fn new() -> Self {
        Self {
            static_values: std::collections::HashMap::new(),
            param_domains: std::collections::HashMap::new(),
            data_domains: std::collections::HashMap::new(),
        }
    }

    /// Add a static value for partial evaluation
    pub fn add_static_value(&mut self, name: &str, value: f64) {
        self.static_values.insert(name.to_string(), value);
    }

    /// Add an abstract domain for a runtime parameter
    pub fn add_param_domain(
        &mut self,
        name: &str,
        domain: crate::interval_domain::IntervalDomain<f64>,
    ) {
        self.param_domains.insert(name.to_string(), domain);
    }

    /// Add an abstract domain for runtime data
    pub fn add_data_domain(
        &mut self,
        name: &str,
        domain: crate::interval_domain::IntervalDomain<f64>,
    ) {
        self.data_domains.insert(name.to_string(), domain);
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
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_simple_expression() {
        let codegen = RustCodeGenerator::new();
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
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
        let expr = ASTEval::mul(ASTEval::var(0), ASTEval::var(1));
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
        let expr = ASTEval::sin(ASTEval::var(0));
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
        let expr = ASTEval::add(
            ASTEval::mul(ASTEval::var(0), ASTEval::var(1)),
            ASTEval::constant(5.0),
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

        let result = compiled_func
            .call_with_spec(&FunctionInput::Scalars(vec![5.0]))
            .unwrap();
        assert_eq!(result, 6.0);

        println!("compile_and_load test passed: f(5) = {result}");
        // No manual cleanup needed - handled automatically by Drop
    }
}
