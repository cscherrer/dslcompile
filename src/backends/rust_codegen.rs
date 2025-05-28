//! Rust Code Generation Backend
//!
//! This module provides Rust source code generation and hot-loading compilation
//! for mathematical expressions. It generates optimized Rust code that can be
//! compiled to dynamic libraries for maximum performance.

use crate::error::{MathJITError, Result};
use crate::final_tagless::JITRepr;
use std::path::Path;

/// Rust compiler optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum RustOptLevel {
    /// No optimization (fastest compilation)
    O0,
    /// Basic optimization
    O1,
    /// Full optimization
    O2,
    /// Aggressive optimization
    O3,
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
        }
    }
}

/// Rust code generator for mathematical expressions
pub struct RustCodeGenerator {
    /// Whether to include debug information
    debug_info: bool,
    /// Whether to use unsafe optimizations
    unsafe_optimizations: bool,
}

impl RustCodeGenerator {
    /// Create a new Rust code generator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            debug_info: false,
            unsafe_optimizations: false,
        }
    }

    /// Create a new Rust code generator with custom settings
    #[must_use]
    pub fn with_settings(debug_info: bool, unsafe_optimizations: bool) -> Self {
        Self {
            debug_info,
            unsafe_optimizations,
        }
    }

    /// Generate Rust source code for a mathematical expression
    pub fn generate_function(&self, expr: &JITRepr<f64>, function_name: &str) -> Result<String> {
        let expr_code = self.generate_expression(expr)?;
        let variables = self.find_variables(expr);

        // Generate function signature based on variables used
        let (params, call_params) = if variables.contains("y") {
            ("x: f64, y: f64", "x, y")
        } else {
            ("x: f64", "x")
        };

        Ok(format!(
            r#"
#[no_mangle]
pub extern "C" fn {function_name}({params}) -> f64 {{
    {expr_code}
}}

#[no_mangle]
pub extern "C" fn {function_name}_two_vars(x: f64, y: f64) -> f64 {{
    {function_name}({call_params})
}}

#[no_mangle]
pub extern "C" fn {function_name}_multi_vars(vars: *const f64, count: usize) -> f64 {{
    if vars.is_null() || count == 0 {{
        return 0.0;
    }}
    
    let x = unsafe {{ *vars }};
    let y = if count > 1 {{ unsafe {{ *vars.add(1) }} }} else {{ 0.0 }};
    
    {function_name}({call_params})
}}
"#
        ))
    }

    /// Generate Rust source code for a complete module
    pub fn generate_module(
        &self,
        expressions: &[(String, JITRepr<f64>)],
        module_name: &str,
    ) -> Result<String> {
        let mut module_code = format!(
            r"//! Generated Rust module: {module_name}
//! This module contains compiled mathematical expressions for high-performance evaluation.

"
        );

        for (func_name, expr) in expressions {
            let func_code = self.generate_function(expr, func_name)?;
            module_code.push_str(&func_code);
            module_code.push('\n');
        }

        Ok(module_code)
    }

    /// Generate Rust expression code from `JITRepr`
    fn generate_expression(&self, expr: &JITRepr<f64>) -> Result<String> {
        match expr {
            JITRepr::Constant(value) => {
                // Ensure floating point literals have .0 suffix if they're whole numbers
                if value.fract() == 0.0 {
                    Ok(format!("{value}.0"))
                } else {
                    Ok(format!("{value}"))
                }
            }
            JITRepr::Variable(name) => {
                // Map variable names to function parameters
                match name.as_str() {
                    "x" => Ok("x".to_string()),
                    "y" => Ok("y".to_string()),
                    _ => Ok("x".to_string()), // Default to x for unknown variables
                }
            }
            JITRepr::Add(left, right) => {
                let left_code = self.generate_expression(left)?;
                let right_code = self.generate_expression(right)?;
                Ok(format!("({left_code} + {right_code})"))
            }
            JITRepr::Sub(left, right) => {
                let left_code = self.generate_expression(left)?;
                let right_code = self.generate_expression(right)?;
                Ok(format!("({left_code} - {right_code})"))
            }
            JITRepr::Mul(left, right) => {
                let left_code = self.generate_expression(left)?;
                let right_code = self.generate_expression(right)?;
                Ok(format!("({left_code} * {right_code})"))
            }
            JITRepr::Div(left, right) => {
                let left_code = self.generate_expression(left)?;
                let right_code = self.generate_expression(right)?;
                Ok(format!("({left_code} / {right_code})"))
            }
            JITRepr::Pow(base, exp) => {
                let base_code = self.generate_expression(base)?;
                let exp_code = self.generate_expression(exp)?;

                // Optimize for integer powers
                if let JITRepr::Constant(exp_val) = exp.as_ref() {
                    if exp_val.fract() == 0.0 && exp_val.abs() <= 10.0 {
                        let exp_int = *exp_val as i32;
                        return Ok(self.generate_integer_power(&base_code, exp_int));
                    }
                }

                Ok(format!("({base_code}).powf({exp_code})"))
            }
            JITRepr::Neg(inner) => {
                let inner_code = self.generate_expression(inner)?;
                Ok(format!("(-{inner_code})"))
            }
            JITRepr::Ln(inner) => {
                let inner_code = self.generate_expression(inner)?;
                Ok(format!("({inner_code}).ln()"))
            }
            JITRepr::Exp(inner) => {
                let inner_code = self.generate_expression(inner)?;
                Ok(format!("({inner_code}).exp()"))
            }
            JITRepr::Sin(inner) => {
                let inner_code = self.generate_expression(inner)?;
                Ok(format!("({inner_code}).sin()"))
            }
            JITRepr::Cos(inner) => {
                let inner_code = self.generate_expression(inner)?;
                Ok(format!("({inner_code}).cos()"))
            }
            JITRepr::Sqrt(inner) => {
                let inner_code = self.generate_expression(inner)?;
                Ok(format!("({inner_code}).sqrt()"))
            }
        }
    }

    /// Generate optimized code for integer powers
    #[allow(clippy::only_used_in_recursion)]
    fn generate_integer_power(&self, base: &str, exp: i32) -> String {
        match exp {
            0 => "1.0".to_string(),
            1 => base.to_string(),
            -1 => format!("(1.0 / {base})"),
            2 => format!("({base} * {base})"),
            3 => format!("({base} * {base} * {base})"),
            4 => {
                let base_squared = format!("({base} * {base})");
                format!("({base_squared} * {base_squared})")
            }
            -2 => {
                let base_squared = format!("({base} * {base})");
                format!("(1.0 / ({base_squared}))")
            }
            exp if exp > 0 && exp <= 10 => {
                // Use repeated multiplication for small positive powers
                let mut result = base.to_string();
                for _ in 1..exp {
                    result = format!("({result} * {base})");
                }
                result
            }
            exp if (-10..0).contains(&exp) => {
                // Handle negative powers
                let positive_power = self.generate_integer_power(base, -exp);
                format!("(1.0 / ({positive_power}))")
            }
            _ => {
                // Fallback to powf for large exponents
                format!("({base}).powf({exp}.0)")
            }
        }
    }

    /// Find all variables used in an expression
    fn find_variables(&self, expr: &JITRepr<f64>) -> std::collections::HashSet<String> {
        let mut variables = std::collections::HashSet::new();
        self.collect_variables(expr, &mut variables);
        variables
    }

    /// Recursively collect all variables in an expression
    fn collect_variables(&self, expr: &JITRepr<f64>, variables: &mut std::collections::HashSet<String>) {
        match expr {
            JITRepr::Variable(name) => {
                variables.insert(name.clone());
            }
            JITRepr::Add(left, right) | JITRepr::Sub(left, right) | JITRepr::Mul(left, right) | JITRepr::Div(left, right) | JITRepr::Pow(left, right) => {
                self.collect_variables(left, variables);
                self.collect_variables(right, variables);
            }
            JITRepr::Neg(inner) | JITRepr::Ln(inner) | JITRepr::Exp(inner) | JITRepr::Sin(inner) | JITRepr::Cos(inner) | JITRepr::Sqrt(inner) => {
                self.collect_variables(inner, variables);
            }
            JITRepr::Constant(_) => {
                // Constants don't contain variables
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

    /// Compile Rust source to a dynamic library
    pub fn compile_dylib(
        &self,
        source_code: &str,
        source_path: &Path,
        output_path: &Path,
    ) -> Result<()> {
        // Write source code to file
        std::fs::write(source_path, source_code)
            .map_err(|e| MathJITError::JITError(format!("Failed to write source file: {e}")))?;

        // Build rustc command
        let mut cmd = std::process::Command::new("rustc");
        cmd.args(["--crate-type=dylib", "-C", self.opt_level.as_flag()]);

        // Add extra flags
        for flag in &self.extra_flags {
            cmd.arg(flag);
        }

        // Add source and output paths
        cmd.arg(source_path.to_str().unwrap())
            .arg("-o")
            .arg(output_path.to_str().unwrap());

        // Execute compilation
        let output = cmd
            .output()
            .map_err(|e| MathJITError::JITError(format!("Failed to run rustc: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MathJITError::JITError(format!(
                "Rust compilation failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Check if rustc is available
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
            .map_err(|e| MathJITError::JITError(format!("Failed to run rustc: {e}")))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(MathJITError::JITError(
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
    use crate::final_tagless::{JITEval, JITMathExpr};

    #[test]
    fn test_rust_code_generation() {
        let codegen = RustCodeGenerator::new();

        // Simple expression: x + 1
        let expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));
        let rust_code = codegen.generate_function(&expr, "test_func").unwrap();

        assert!(rust_code.contains("#[no_mangle]"));
        assert!(rust_code.contains("pub extern \"C\" fn test_func"));
        assert!(rust_code.contains("(x + 1.0)"));
    }

    #[test]
    fn test_complex_expression_generation() {
        let codegen = RustCodeGenerator::new();

        // Complex expression: x^2 + 2*x + 1
        let expr = JITEval::add(
            JITEval::add(
                JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),
                JITEval::mul(JITEval::constant(2.0), JITEval::var("x")),
            ),
            JITEval::constant(1.0),
        );

        let rust_code = codegen.generate_function(&expr, "quadratic").unwrap();

        assert!(rust_code.contains("quadratic"));
        assert!(rust_code.contains("(x * x)")); // Should optimize x^2
        assert!(rust_code.contains("(2.0 * x)"));
    }

    #[test]
    fn test_integer_power_optimization() {
        let codegen = RustCodeGenerator::new();

        // Test x^3
        let expr = JITEval::pow(JITEval::var("x"), JITEval::constant(3.0));
        let rust_code = codegen.generate_function(&expr, "cube").unwrap();

        // Should generate optimized multiplication instead of powf
        assert!(rust_code.contains("(x * x * x)"));
        assert!(!rust_code.contains("powf"));
    }

    #[test]
    fn test_transcendental_functions() {
        let codegen = RustCodeGenerator::new();

        // Test sin(cos(x))
        let expr = JITEval::sin(JITEval::cos(JITEval::var("x")));
        let rust_code = codegen.generate_function(&expr, "trig_func").unwrap();

        assert!(rust_code.contains("sin"));
        assert!(rust_code.contains("cos"));
        assert!(rust_code.contains("trig_func"));
    }

    #[test]
    fn test_module_generation() {
        let codegen = RustCodeGenerator::new();

        let expressions = vec![
            (
                "linear".to_string(),
                JITEval::add(JITEval::var("x"), JITEval::constant(1.0)),
            ),
            (
                "quadratic".to_string(),
                JITEval::pow(JITEval::var("x"), JITEval::constant(2.0)),
            ),
        ];

        let module_code = codegen
            .generate_module(&expressions, "test_module")
            .unwrap();

        assert!(module_code.contains("test_module"));
        assert!(module_code.contains("linear"));
        assert!(module_code.contains("quadratic"));
        assert!(module_code.contains("#[no_mangle]"));
    }

    #[test]
    fn test_rust_compiler_availability() {
        // This test checks if rustc is available, but doesn't fail if it's not
        let available = RustCompiler::is_available();
        println!("Rustc available: {available}");

        if available {
            let version = RustCompiler::version_info();
            println!("Rustc version: {version:?}");
        }
    }

    #[test]
    fn test_rust_compiler_creation() {
        let compiler = RustCompiler::new();
        assert_eq!(compiler.opt_level, RustOptLevel::O2);

        let compiler_o3 = RustCompiler::with_opt_level(RustOptLevel::O3);
        assert_eq!(compiler_o3.opt_level, RustOptLevel::O3);
    }
}
