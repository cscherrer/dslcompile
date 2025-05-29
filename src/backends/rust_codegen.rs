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

use crate::ast_utils::collect_variable_indices;
use crate::error::{MathCompileError, Result};
use crate::final_tagless::{ASTRepr, NumericType, VariableRegistry};
use crate::power_utils::{PowerOptConfig, generate_integer_power_string, try_convert_to_integer};
use libloading;
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
    {expr_code}
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
                // Handle different numeric types appropriately
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let val = unsafe { std::mem::transmute_copy::<T, f64>(value) };
                    Ok(format!("{val}_f64"))
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    let val = unsafe { std::mem::transmute_copy::<T, f32>(value) };
                    Ok(format!("{val}_f32"))
                } else {
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

/// Compiled Rust function wrapper for convenient calling
pub struct CompiledRustFunction {
    /// The loaded dynamic library (kept alive)
    _library: libloading::Library,
    /// Function pointer for single variable functions
    single_var_func: Option<libloading::Symbol<'static, extern "C" fn(f64) -> f64>>,
    /// Function pointer for two variable functions  
    two_var_func: Option<libloading::Symbol<'static, extern "C" fn(f64, f64) -> f64>>,
    /// Function pointer for multi-variable functions (takes array)
    multi_var_func: Option<libloading::Symbol<'static, extern "C" fn(*const f64, usize) -> f64>>,
    /// The function name for debugging
    function_name: String,
    /// Path to the temporary library file (for cleanup)
    lib_path: Option<std::path::PathBuf>,
}

impl CompiledRustFunction {
    /// Load a compiled dynamic library and create a function wrapper
    ///
    /// # Safety
    ///
    /// This function is unsafe because it loads and calls functions from a dynamic library.
    /// The caller must ensure that:
    /// - The library path points to a valid dynamic library
    /// - The function name exists in the library
    /// - The function has the expected signature
    unsafe fn load(lib_path: &Path, function_name: &str) -> Result<Self> {
        unsafe { Self::load_with_cleanup(lib_path, function_name, None) }
    }

    /// Load a compiled dynamic library with optional cleanup path
    ///
    /// # Safety
    ///
    /// This function is unsafe because it loads and calls functions from a dynamic library.
    /// The caller must ensure that:
    /// - The library path points to a valid dynamic library
    /// - The function name exists in the library
    /// - The function has the expected signature
    unsafe fn load_with_cleanup(
        lib_path: &Path,
        function_name: &str,
        cleanup_path: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        unsafe {
            let library = libloading::Library::new(lib_path).map_err(|e| {
                MathCompileError::CompilationError(format!("Failed to load library: {e}"))
            })?;

            // Try to load the single variable function
            let single_var_func = library
                .get::<extern "C" fn(f64) -> f64>(function_name.as_bytes())
                .ok()
                .map(|symbol| std::mem::transmute(symbol));

            // Try to load the two variable function
            let two_var_func_name = format!("{function_name}_two_vars");
            let two_var_func = library
                .get::<extern "C" fn(f64, f64) -> f64>(two_var_func_name.as_bytes())
                .ok()
                .map(|symbol| std::mem::transmute(symbol));

            // Try to load the multi-variable function
            let multi_var_func_name = format!("{function_name}_multi_vars");
            let multi_var_func = library
                .get::<extern "C" fn(*const f64, usize) -> f64>(multi_var_func_name.as_bytes())
                .ok()
                .map(|symbol| std::mem::transmute(symbol));

            Ok(CompiledRustFunction {
                _library: library,
                single_var_func,
                two_var_func,
                multi_var_func,
                function_name: function_name.to_string(),
                lib_path: cleanup_path,
            })
        }
    }

    /// Call the function with a single variable
    pub fn call(&self, x: f64) -> Result<f64> {
        if let Some(func) = &self.single_var_func {
            Ok(func(x))
        } else {
            Err(MathCompileError::CompilationError(format!(
                "Single variable function '{}' not found in library",
                self.function_name
            )))
        }
    }

    /// Call the function with two variables
    pub fn call_two_vars(&self, x: f64, y: f64) -> Result<f64> {
        if let Some(func) = &self.two_var_func {
            Ok(func(x, y))
        } else {
            Err(MathCompileError::CompilationError(format!(
                "Two variable function '{}_two_vars' not found in library",
                self.function_name
            )))
        }
    }

    /// Call the function with multiple variables
    pub fn call_multi_vars(&self, vars: &[f64]) -> Result<f64> {
        if let Some(func) = &self.multi_var_func {
            Ok(func(vars.as_ptr(), vars.len()))
        } else {
            Err(MathCompileError::CompilationError(format!(
                "Multi variable function '{}_multi_vars' not found in library",
                self.function_name
            )))
        }
    }

    /// Get the function name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.function_name
    }
}

impl Drop for CompiledRustFunction {
    fn drop(&mut self) {
        if let Some(lib_path) = self.lib_path.take() {
            let _ = std::fs::remove_file(&lib_path);
        }
    }
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

        let result = compiled_func.call(5.0).unwrap();
        assert_eq!(result, 6.0);

        println!("compile_and_load test passed: f(5) = {result}");
        // No manual cleanup needed - handled automatically by Drop
    }
}
