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

use crate::error::{MathJITError, Result};
use crate::final_tagless::{ASTRepr, NumericType};
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
    /// Whether to include debug information
    pub debug_info: bool,
    /// Whether to use unsafe optimizations
    pub unsafe_optimizations: bool,
    /// Whether to enable vectorization hints
    pub vectorization_hints: bool,
    /// Whether to inline aggressively
    pub aggressive_inlining: bool,
    /// Target CPU features
    pub target_cpu: Option<String>,
}

impl Default for RustCodegenConfig {
    fn default() -> Self {
        Self {
            debug_info: false,
            unsafe_optimizations: false,
            vectorization_hints: true,
            aggressive_inlining: true,
            target_cpu: None,
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

    /// Generate Rust source code for a mathematical expression (generic version)
    pub fn generate_function_generic<T: NumericType + Float + Copy>(
        &self,
        expr: &ASTRepr<T>,
        function_name: &str,
        type_name: &str,
    ) -> Result<String> {
        let expr_code = self.generate_expression_generic(expr)?;
        let variables = self.find_variables_generic(expr);

        // Generate function signature based on variables used
        let (params, call_params, y_param, y_var) = if variables.contains("y") {
            (
                format!("x: {type_name}, y: {type_name}"),
                "x, y".to_string(),
                format!("y: {type_name}"),
                "y".to_string(),
            )
        } else {
            (
                format!("x: {type_name}"),
                "x".to_string(),
                format!("_y: {type_name}"),
                "_y".to_string(),
            )
        };

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
pub extern "C" fn {function_name}_two_vars(x: {type_name}, {y_param}) -> {type_name} {{
    {function_name}({call_params})
}}

{attributes}#[no_mangle]
pub extern "C" fn {function_name}_multi_vars(vars: *const {type_name}, count: usize) -> {type_name} {{
    if vars.is_null() || count == 0 {{
        return Default::default();
    }}
    
    let x = unsafe {{ *vars }};
    let {y_var} = if count > 1 {{ unsafe {{ *vars.add(1) }} }} else {{ Default::default() }};
    
    {function_name}({call_params})
}}
"#
        ))
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
    fn generate_expression_generic<T: NumericType + Float + Copy>(
        &self,
        expr: &ASTRepr<T>,
    ) -> Result<String> {
        match expr {
            ASTRepr::Constant(value) => {
                // For generic types, we use the Display implementation
                Ok(format!("{value}"))
            }
            ASTRepr::Variable(index) => {
                // Map variable indices to function parameters
                match *index {
                    0 => Ok("x".to_string()),
                    1 => Ok("y".to_string()),
                    _ => Ok(format!("var_{index}")), // Generic variable name for unknown indices
                }
            }
            ASTRepr::VariableByName(name) => Ok(name.clone()),
            ASTRepr::Add(left, right) => {
                let left_code = self.generate_expression_generic(left)?;
                let right_code = self.generate_expression_generic(right)?;
                Ok(format!("{left_code} + {right_code}"))
            }
            ASTRepr::Sub(left, right) => {
                let left_code = self.generate_expression_generic(left)?;
                let right_code = self.generate_expression_generic(right)?;
                Ok(format!("{left_code} - {right_code}"))
            }
            ASTRepr::Mul(left, right) => {
                let left_code = self.generate_expression_generic(left)?;
                let right_code = self.generate_expression_generic(right)?;
                Ok(format!("{left_code} * {right_code}"))
            }
            ASTRepr::Div(left, right) => {
                let left_code = self.generate_expression_generic(left)?;
                let right_code = self.generate_expression_generic(right)?;
                Ok(format!("{left_code} / {right_code}"))
            }
            ASTRepr::Pow(base, exp) => {
                let base_code = self.generate_expression_generic(base)?;
                let exp_code = self.generate_expression_generic(exp)?;

                // Check if the exponent is a constant for optimization
                if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                    // For integer powers, use specialized implementation
                    if let Some(exp_int) = self.try_convert_to_integer(*exp_val) {
                        if exp_int.abs() <= 10 {
                            return Ok(self.generate_integer_power_generic(&base_code, exp_int));
                        }
                    }
                }

                Ok(format!("{base_code}.powf({exp_code})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_code = self.generate_expression_generic(inner)?;
                Ok(format!("-{inner_code}"))
            }
            ASTRepr::Ln(inner) => {
                let inner_code = self.generate_expression_generic(inner)?;
                Ok(format!("{inner_code}.ln()"))
            }
            ASTRepr::Exp(inner) => {
                let inner_code = self.generate_expression_generic(inner)?;
                Ok(format!("{inner_code}.exp()"))
            }
            ASTRepr::Sin(inner) => {
                let inner_code = self.generate_expression_generic(inner)?;
                Ok(format!("{inner_code}.sin()"))
            }
            ASTRepr::Cos(inner) => {
                let inner_code = self.generate_expression_generic(inner)?;
                Ok(format!("{inner_code}.cos()"))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_code = self.generate_expression_generic(inner)?;
                Ok(format!("{inner_code}.sqrt()"))
            }
        }
    }

    /// Generate Rust expression code from `ASTRepr` (f64 specialization for backwards compatibility)
    fn generate_expression(&self, expr: &ASTRepr<f64>) -> Result<String> {
        self.generate_expression_generic(expr)
    }

    /// Try to convert a generic numeric value to an integer for optimization purposes
    fn try_convert_to_integer<T: NumericType + Float + Copy>(&self, value: T) -> Option<i32> {
        // Convert to f64 for the check
        let float_val = value.to_f64().unwrap_or(0.0);
        if float_val.fract() == 0.0 && float_val.abs() <= 100.0 {
            Some(float_val as i32)
        } else {
            None
        }
    }

    /// Generate optimized code for integer powers (generic version)
    fn generate_integer_power_generic(&self, base: &str, exp: i32) -> String {
        match exp {
            0 => "1.0".to_string(),
            1 => base.to_string(),
            -1 => format!("1.0 / {base}"),
            2 => format!("{base} * {base}"),
            -2 => format!("1.0 / ({base} * {base})"),
            3 => format!("{base} * {base} * {base}"),
            4 => {
                if self.config.unsafe_optimizations {
                    format!("{{ let temp = {base} * {base}; temp * temp }}")
                } else {
                    format!("{base} * {base} * {base} * {base}")
                }
            }
            5 => format!("{{ let temp = {base} * {base}; temp * temp * {base} }}"),
            6 => format!("{{ let temp = {base} * {base} * {base}; temp * temp }}"),
            8 => format!(
                "{{ let temp2 = {base} * {base}; let temp4 = temp2 * temp2; temp4 * temp4 }}"
            ),
            10 => format!(
                "{{ let temp5 = {base} * {base} * {base} * {base} * {base}; temp5 * temp5 }}"
            ),
            exp if exp < 0 => format!(
                "1.0 / ({})",
                self.generate_integer_power_generic(base, -exp)
            ),
            _ => format!("{base}.powi({exp})"),
        }
    }

    /// Generate optimized code for integer powers (f64 specialization for backwards compatibility)
    fn generate_integer_power(&self, base: &str, exp: i32) -> String {
        self.generate_integer_power_generic(base, exp)
    }

    /// Find variables in the expression (generic version)
    fn find_variables_generic<T: NumericType>(
        &self,
        expr: &ASTRepr<T>,
    ) -> std::collections::HashSet<String> {
        let mut variables = std::collections::HashSet::new();
        self.collect_variables_generic(expr, &mut variables);
        variables
    }

    /// Find variables in the expression (f64 specialization for backwards compatibility)
    fn find_variables(&self, expr: &ASTRepr<f64>) -> std::collections::HashSet<String> {
        self.find_variables_generic(expr)
    }

    /// Collect variables recursively (generic version)
    fn collect_variables_generic<T: NumericType>(
        &self,
        expr: &ASTRepr<T>,
        variables: &mut std::collections::HashSet<String>,
    ) {
        match expr {
            ASTRepr::Variable(index) => {
                // Map variable indices to generic names
                let var_name = match *index {
                    0 => "x".to_string(),
                    1 => "y".to_string(),
                    _ => format!("var_{index}"),
                };
                variables.insert(var_name);
            }
            ASTRepr::VariableByName(name) => {
                variables.insert(name.clone());
            }
            ASTRepr::Add(left, right)
            | ASTRepr::Sub(left, right)
            | ASTRepr::Mul(left, right)
            | ASTRepr::Div(left, right)
            | ASTRepr::Pow(left, right) => {
                self.collect_variables_generic(left, variables);
                self.collect_variables_generic(right, variables);
            }
            ASTRepr::Neg(inner)
            | ASTRepr::Ln(inner)
            | ASTRepr::Exp(inner)
            | ASTRepr::Sin(inner)
            | ASTRepr::Cos(inner)
            | ASTRepr::Sqrt(inner) => {
                self.collect_variables_generic(inner, variables);
            }
            ASTRepr::Constant(_) => {
                // Constants don't contribute variables
            }
        }
    }

    /// Collect variables recursively (f64 specialization for backwards compatibility)
    fn collect_variables(
        &self,
        expr: &ASTRepr<f64>,
        variables: &mut std::collections::HashSet<String>,
    ) {
        self.collect_variables_generic(expr, variables);
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
            MathJITError::CompilationError(format!("Failed to write source file: {e}"))
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
            .map_err(|e| MathJITError::CompilationError(format!("Failed to run rustc: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MathJITError::CompilationError(format!(
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
            .map_err(|e| MathJITError::CompilationError(format!("Failed to run rustc: {e}")))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(MathJITError::CompilationError(
                "Failed to get rustc version".to_string(),
            ))
        }
    }
}

impl Default for RustCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_rust_code_generation() {
        let codegen = RustCodeGenerator::new();

        // Simple expression: x + 1
        let expr = ASTEval::add(ASTEval::var_by_name("x"), ASTEval::constant(1.0));
        let rust_code = codegen.generate_function(&expr, "test_func").unwrap();

        assert!(rust_code.contains("#[no_mangle]"));
        assert!(rust_code.contains("pub extern \"C\" fn test_func"));
        assert!(rust_code.contains("x + 1")); // Updated to match cleaner output
    }

    #[test]
    fn test_complex_expression_generation() {
        let codegen = RustCodeGenerator::new();

        // Complex expression: x² + 2x + 1
        let expr = ASTEval::add(
            ASTEval::add(
                ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
                ASTEval::mul(ASTEval::constant(2.0), ASTEval::var_by_name("x")),
            ),
            ASTEval::constant(1.0),
        );

        let rust_code = codegen.generate_function(&expr, "quadratic").unwrap();

        assert!(rust_code.contains("#[no_mangle]"));
        assert!(rust_code.contains("quadratic"));
        assert!(rust_code.contains('x'));
    }

    #[test]
    fn test_integer_power_optimization() {
        let codegen = RustCodeGenerator::new();

        // Test x^3
        let expr = ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(3.0));
        let rust_code = codegen.generate_function(&expr, "cube").unwrap();

        // Should generate optimized multiplication instead of powf
        assert!(rust_code.contains("x * x * x"));
        assert!(!rust_code.contains("powf"));
    }

    #[test]
    fn test_transcendental_functions() {
        let codegen = RustCodeGenerator::new();

        // Test sin(x) + cos(x)
        let expr = ASTEval::add(
            ASTEval::sin(ASTEval::var_by_name("x")),
            ASTEval::cos(ASTEval::var_by_name("x")),
        );

        let rust_code = codegen.generate_function(&expr, "trig_func").unwrap();

        assert!(rust_code.contains("sin()"));
        assert!(rust_code.contains("cos()"));
        assert!(rust_code.contains('x'));
    }

    #[test]
    fn test_module_generation() {
        let codegen = RustCodeGenerator::new();

        let expressions = vec![
            (
                "linear".to_string(),
                ASTEval::add(ASTEval::var_by_name("x"), ASTEval::constant(1.0)),
            ),
            (
                "square".to_string(),
                ASTEval::pow(ASTEval::var_by_name("x"), ASTEval::constant(2.0)),
            ),
        ];

        let module_code = codegen
            .generate_module(&expressions, "test_module")
            .unwrap();

        assert!(module_code.contains("linear"));
        assert!(module_code.contains("square"));
        assert!(module_code.contains("#[no_mangle]"));

        // Should contain both functions
        let linear_count = module_code.matches("linear").count();
        let square_count = module_code.matches("square").count();
        assert!(linear_count >= 2); // Function name appears multiple times
        assert!(square_count >= 2);
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
}
