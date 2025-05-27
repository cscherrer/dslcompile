//! Symbolic optimization using egglog for algebraic simplification
//!
//! This module provides Layer 2 optimization in our three-layer optimization strategy:
//! 1. Hand-coded domain optimizations (in JIT layer)
//! 2. **Egglog symbolic optimization** (this module)
//! 3. Cranelift low-level optimization
//!
//! The symbolic optimizer handles algebraic identities, constant folding, and structural
//! optimizations that can be expressed as rewrite rules.

use crate::error::Result;
use crate::final_tagless::JITRepr;

/// Compilation strategy for mathematical expressions
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationStrategy {
    /// Fast JIT compilation using Cranelift (default)
    /// Best for: Simple expressions, rapid iteration, low latency
    CraneliftJIT,
    /// Hot-loading compiled Rust dylibs
    /// Best for: Complex expressions, maximum performance, production use
    HotLoadRust {
        /// Directory for generated Rust source files
        source_dir: std::path::PathBuf,
        /// Directory for compiled dylibs
        lib_dir: std::path::PathBuf,
        /// Optimization level for rustc
        opt_level: RustOptLevel,
    },
    /// Adaptive strategy: start with Cranelift, upgrade to Rust for hot expressions
    Adaptive {
        /// Threshold for call count before upgrading to Rust compilation
        call_threshold: usize,
        /// Threshold for expression complexity before upgrading
        complexity_threshold: usize,
    },
}

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

/// Compilation approach decision for a specific expression
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationApproach {
    /// Use Cranelift JIT compilation
    Cranelift,
    /// Use Rust hot-loading compilation
    RustHotLoad,
    /// Upgrade from Cranelift to Rust compilation
    UpgradeToRust,
}

impl Default for CompilationStrategy {
    fn default() -> Self {
        Self::CraneliftJIT
    }
}

/// Symbolic optimizer for expression simplification
///
/// This is a simplified implementation that will be enhanced with egglog integration.
/// For now, it implements basic algebraic simplifications directly.
pub struct SymbolicOptimizer {
    /// Configuration for optimization behavior
    config: OptimizationConfig,
    /// Compilation strategy
    compilation_strategy: CompilationStrategy,
    /// Statistics for adaptive compilation
    expression_stats: std::collections::HashMap<String, ExpressionStats>,
}

/// Statistics for tracking expression usage patterns
#[derive(Debug, Clone)]
pub struct ExpressionStats {
    /// Number of times this expression has been called
    pub call_count: usize,
    /// Complexity score of the expression
    pub complexity: usize,
    /// Average execution time in nanoseconds
    pub avg_execution_time_ns: f64,
    /// Whether this expression has been upgraded to Rust compilation
    pub rust_compiled: bool,
}

impl SymbolicOptimizer {
    /// Create a new symbolic optimizer with default configuration
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: OptimizationConfig::default(),
            compilation_strategy: CompilationStrategy::default(),
            expression_stats: std::collections::HashMap::new(),
        })
    }

    /// Create a new symbolic optimizer with custom configuration
    pub fn with_config(config: OptimizationConfig) -> Result<Self> {
        Ok(Self {
            config,
            compilation_strategy: CompilationStrategy::default(),
            expression_stats: std::collections::HashMap::new(),
        })
    }

    /// Create a new symbolic optimizer with custom compilation strategy
    pub fn with_strategy(strategy: CompilationStrategy) -> Result<Self> {
        Ok(Self {
            config: OptimizationConfig::default(),
            compilation_strategy: strategy,
            expression_stats: std::collections::HashMap::new(),
        })
    }

    /// Set the compilation strategy
    pub fn set_compilation_strategy(&mut self, strategy: CompilationStrategy) {
        self.compilation_strategy = strategy;
    }

    /// Get the current compilation strategy
    #[must_use]
    pub fn compilation_strategy(&self) -> &CompilationStrategy {
        &self.compilation_strategy
    }

    /// Determine the best compilation approach for a given expression
    pub fn choose_compilation_approach(
        &mut self,
        expr: &JITRepr<f64>,
        expr_id: &str,
    ) -> CompilationApproach {
        match &self.compilation_strategy {
            CompilationStrategy::CraneliftJIT => CompilationApproach::Cranelift,
            CompilationStrategy::HotLoadRust { .. } => CompilationApproach::RustHotLoad,
            CompilationStrategy::Adaptive {
                call_threshold,
                complexity_threshold,
            } => {
                let stats = self
                    .expression_stats
                    .entry(expr_id.to_string())
                    .or_insert_with(|| ExpressionStats {
                        call_count: 0,
                        complexity: expr.count_operations(),
                        avg_execution_time_ns: 0.0,
                        rust_compiled: false,
                    });

                // Upgrade to Rust compilation if thresholds are met
                if stats.call_count >= *call_threshold || stats.complexity >= *complexity_threshold
                {
                    if stats.rust_compiled {
                        CompilationApproach::RustHotLoad
                    } else {
                        stats.rust_compiled = true;
                        CompilationApproach::UpgradeToRust
                    }
                } else {
                    CompilationApproach::Cranelift
                }
            }
        }
    }

    /// Record execution statistics for adaptive compilation
    pub fn record_execution(&mut self, expr_id: &str, execution_time_ns: u64) {
        let stats = self
            .expression_stats
            .entry(expr_id.to_string())
            .or_insert_with(|| {
                ExpressionStats {
                    call_count: 0,
                    complexity: 0, // We don't have the expression here, so default to 0
                    avg_execution_time_ns: 0.0,
                    rust_compiled: false,
                }
            });

        stats.call_count += 1;
        // Update running average
        let alpha = 0.1; // Exponential moving average factor
        stats.avg_execution_time_ns =
            alpha * execution_time_ns as f64 + (1.0 - alpha) * stats.avg_execution_time_ns;
    }

    /// Get statistics for all tracked expressions
    #[must_use]
    pub fn get_expression_stats(&self) -> &std::collections::HashMap<String, ExpressionStats> {
        &self.expression_stats
    }

    /// Generate Rust source code for hot-loading compilation
    pub fn generate_rust_source(&self, expr: &JITRepr<f64>, function_name: &str) -> Result<String> {
        let expr_code = self.generate_rust_expression(expr)?;

        Ok(format!(
            r#"
#[no_mangle]
pub extern "C" fn {function_name}(x: f64) -> f64 {{
    {expr_code}
}}

#[no_mangle]
pub extern "C" fn {function_name}_two_vars(x: f64, y: f64) -> f64 {{
    let _ = y; // Suppress unused variable warning if not used
    {expr_code}
}}

#[no_mangle]
pub extern "C" fn {function_name}_multi_vars(vars: *const f64, count: usize) -> f64 {{
    if vars.is_null() || count == 0 {{
        return 0.0;
    }}
    
    let x = unsafe {{ *vars }};
    let y = if count > 1 {{ unsafe {{ *vars.add(1) }} }} else {{ 0.0 }};
    let _ = (y, count); // Suppress unused variable warnings
    
    {expr_code}
}}
"#
        ))
    }

    /// Generate Rust expression code from `JITRepr`
    fn generate_rust_expression(&self, expr: &JITRepr<f64>) -> Result<String> {
        match expr {
            JITRepr::Constant(value) => Ok(format!("{value}")),
            JITRepr::Variable(name) => {
                // Map variable names to function parameters
                match name.as_str() {
                    "x" => Ok("x".to_string()),
                    "y" => Ok("y".to_string()),
                    _ => Ok("x".to_string()), // Default to x for unknown variables
                }
            }
            JITRepr::Add(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} + {right_code})"))
            }
            JITRepr::Sub(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} - {right_code})"))
            }
            JITRepr::Mul(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} * {right_code})"))
            }
            JITRepr::Div(left, right) => {
                let left_code = self.generate_rust_expression(left)?;
                let right_code = self.generate_rust_expression(right)?;
                Ok(format!("({left_code} / {right_code})"))
            }
            JITRepr::Pow(base, exp) => {
                let base_code = self.generate_rust_expression(base)?;
                let exp_code = self.generate_rust_expression(exp)?;
                Ok(format!("({base_code}).powf({exp_code})"))
            }
            JITRepr::Neg(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("(-{inner_code})"))
            }
            JITRepr::Ln(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("({inner_code}).ln()"))
            }
            JITRepr::Exp(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("({inner_code}).exp()"))
            }
            JITRepr::Sin(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("({inner_code}).sin()"))
            }
            JITRepr::Cos(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("({inner_code}).cos()"))
            }
            JITRepr::Sqrt(inner) => {
                let inner_code = self.generate_rust_expression(inner)?;
                Ok(format!("({inner_code}).sqrt()"))
            }
        }
    }

    /// Compile Rust source to a dynamic library
    pub fn compile_rust_dylib(
        &self,
        source_code: &str,
        source_path: &std::path::Path,
        output_path: &std::path::Path,
        opt_level: &RustOptLevel,
    ) -> Result<()> {
        // Write source code to file
        std::fs::write(source_path, source_code).map_err(|e| {
            crate::error::MathJITError::JITError(format!("Failed to write source file: {e}"))
        })?;

        // Determine optimization flag
        let opt_flag = match opt_level {
            RustOptLevel::O0 => "-C opt-level=0",
            RustOptLevel::O1 => "-C opt-level=1",
            RustOptLevel::O2 => "-C opt-level=2",
            RustOptLevel::O3 => "-C opt-level=3",
        };

        // Compile with rustc
        let output = std::process::Command::new("rustc")
            .args([
                "--crate-type=dylib",
                opt_flag,
                "-C",
                "panic=abort", // Smaller binary size
                "-C",
                "lto=thin", // Link-time optimization
                source_path.to_str().unwrap(),
                "-o",
                output_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| {
                crate::error::MathJITError::JITError(format!("Failed to run rustc: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::error::MathJITError::JITError(format!(
                "Rust compilation failed: {stderr}"
            )));
        }

        Ok(())
    }

    /// Get the recommended compilation strategy based on expression characteristics
    #[must_use]
    pub fn recommend_strategy(expr: &JITRepr<f64>) -> CompilationStrategy {
        let complexity = expr.count_operations();

        if complexity < 10 {
            // Simple expressions: use fast Cranelift JIT
            CompilationStrategy::CraneliftJIT
        } else if complexity < 50 {
            // Medium complexity: use adaptive approach
            CompilationStrategy::Adaptive {
                call_threshold: 100,
                complexity_threshold: 25,
            }
        } else {
            // Complex expressions: use Rust hot-loading for maximum performance
            CompilationStrategy::HotLoadRust {
                source_dir: std::path::PathBuf::from("/tmp/mathjit_sources"),
                lib_dir: std::path::PathBuf::from("/tmp/mathjit_libs"),
                opt_level: RustOptLevel::O2,
            }
        }
    }

    /// Optimize a JIT representation using symbolic rewrite rules
    pub fn optimize(&mut self, expr: &JITRepr<f64>) -> Result<JITRepr<f64>> {
        let mut optimized = expr.clone();
        let mut iterations = 0;

        // Apply optimization passes until convergence or max iterations
        while iterations < self.config.max_iterations {
            let before = optimized.clone();

            // Apply basic algebraic simplifications
            optimized = Self::apply_arithmetic_rules(&optimized)?;
            optimized = Self::apply_algebraic_rules(&optimized)?;

            if self.config.constant_folding {
                optimized = Self::apply_constant_folding(&optimized)?;
            }

            // Check for convergence
            if Self::expressions_equal(&before, &optimized) {
                break;
            }

            iterations += 1;
        }

        Ok(optimized)
    }

    /// Apply basic arithmetic simplification rules
    fn apply_arithmetic_rules(expr: &JITRepr<f64>) -> Result<JITRepr<f64>> {
        match expr {
            // Identity rules: x + 0 = x, x * 1 = x, etc.
            JITRepr::Add(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, JITRepr::Constant(0.0)) => Ok(left_opt),
                    (JITRepr::Constant(0.0), _) => Ok(right_opt),
                    _ => Ok(JITRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Mul(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, JITRepr::Constant(1.0)) => Ok(left_opt),
                    (JITRepr::Constant(1.0), _) => Ok(right_opt),
                    (_, JITRepr::Constant(0.0)) => Ok(JITRepr::Constant(0.0)),
                    (JITRepr::Constant(0.0), _) => Ok(JITRepr::Constant(0.0)),
                    _ => Ok(JITRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Sub(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, JITRepr::Constant(0.0)) => Ok(left_opt),
                    (l, r) if Self::expressions_equal(l, r) => Ok(JITRepr::Constant(0.0)),
                    _ => Ok(JITRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Div(left, right) => {
                let left_opt = Self::apply_arithmetic_rules(left)?;
                let right_opt = Self::apply_arithmetic_rules(right)?;

                match (&left_opt, &right_opt) {
                    (_, JITRepr::Constant(1.0)) => Ok(left_opt),
                    _ => Ok(JITRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Pow(base, exp) => {
                let base_opt = Self::apply_arithmetic_rules(base)?;
                let exp_opt = Self::apply_arithmetic_rules(exp)?;

                match (&base_opt, &exp_opt) {
                    (_, JITRepr::Constant(0.0)) => Ok(JITRepr::Constant(1.0)),
                    (_, JITRepr::Constant(1.0)) => Ok(base_opt),
                    (JITRepr::Constant(1.0), _) => Ok(JITRepr::Constant(1.0)),
                    _ => Ok(JITRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            // Recursively apply to other expression types
            JITRepr::Neg(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                Ok(JITRepr::Neg(Box::new(inner_opt)))
            }
            JITRepr::Ln(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    JITRepr::Constant(1.0) => Ok(JITRepr::Constant(0.0)),
                    _ => Ok(JITRepr::Ln(Box::new(inner_opt))),
                }
            }
            JITRepr::Exp(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    JITRepr::Constant(0.0) => Ok(JITRepr::Constant(1.0)),
                    _ => Ok(JITRepr::Exp(Box::new(inner_opt))),
                }
            }
            JITRepr::Sin(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    JITRepr::Constant(0.0) => Ok(JITRepr::Constant(0.0)),
                    _ => Ok(JITRepr::Sin(Box::new(inner_opt))),
                }
            }
            JITRepr::Cos(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                match &inner_opt {
                    JITRepr::Constant(0.0) => Ok(JITRepr::Constant(1.0)),
                    _ => Ok(JITRepr::Cos(Box::new(inner_opt))),
                }
            }
            JITRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_arithmetic_rules(inner)?;
                Ok(JITRepr::Sqrt(Box::new(inner_opt)))
            }
            // Base cases
            JITRepr::Constant(_) | JITRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Apply algebraic transformation rules (associativity, commutativity, etc.)
    fn apply_algebraic_rules(expr: &JITRepr<f64>) -> Result<JITRepr<f64>> {
        // For now, just recursively apply to subexpressions
        // In a full implementation, this would handle more complex algebraic transformations
        match expr {
            JITRepr::Add(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(JITRepr::Add(Box::new(left_opt), Box::new(right_opt)))
            }
            JITRepr::Mul(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(JITRepr::Mul(Box::new(left_opt), Box::new(right_opt)))
            }
            JITRepr::Sub(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(JITRepr::Sub(Box::new(left_opt), Box::new(right_opt)))
            }
            JITRepr::Div(left, right) => {
                let left_opt = Self::apply_algebraic_rules(left)?;
                let right_opt = Self::apply_algebraic_rules(right)?;
                Ok(JITRepr::Div(Box::new(left_opt), Box::new(right_opt)))
            }
            JITRepr::Pow(base, exp) => {
                let base_opt = Self::apply_algebraic_rules(base)?;
                let exp_opt = Self::apply_algebraic_rules(exp)?;
                Ok(JITRepr::Pow(Box::new(base_opt), Box::new(exp_opt)))
            }
            JITRepr::Neg(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(JITRepr::Neg(Box::new(inner_opt)))
            }
            JITRepr::Ln(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(JITRepr::Ln(Box::new(inner_opt)))
            }
            JITRepr::Exp(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(JITRepr::Exp(Box::new(inner_opt)))
            }
            JITRepr::Sin(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(JITRepr::Sin(Box::new(inner_opt)))
            }
            JITRepr::Cos(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(JITRepr::Cos(Box::new(inner_opt)))
            }
            JITRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_algebraic_rules(inner)?;
                Ok(JITRepr::Sqrt(Box::new(inner_opt)))
            }
            JITRepr::Constant(_) | JITRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Apply constant folding optimizations
    fn apply_constant_folding(expr: &JITRepr<f64>) -> Result<JITRepr<f64>> {
        match expr {
            JITRepr::Add(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (JITRepr::Constant(a), JITRepr::Constant(b)) => Ok(JITRepr::Constant(a + b)),
                    _ => Ok(JITRepr::Add(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Mul(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (JITRepr::Constant(a), JITRepr::Constant(b)) => Ok(JITRepr::Constant(a * b)),
                    _ => Ok(JITRepr::Mul(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Sub(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (JITRepr::Constant(a), JITRepr::Constant(b)) => Ok(JITRepr::Constant(a - b)),
                    _ => Ok(JITRepr::Sub(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Div(left, right) => {
                let left_opt = Self::apply_constant_folding(left)?;
                let right_opt = Self::apply_constant_folding(right)?;

                match (&left_opt, &right_opt) {
                    (JITRepr::Constant(a), JITRepr::Constant(b)) if *b != 0.0 => {
                        Ok(JITRepr::Constant(a / b))
                    }
                    _ => Ok(JITRepr::Div(Box::new(left_opt), Box::new(right_opt))),
                }
            }
            JITRepr::Pow(base, exp) => {
                let base_opt = Self::apply_constant_folding(base)?;
                let exp_opt = Self::apply_constant_folding(exp)?;

                match (&base_opt, &exp_opt) {
                    (JITRepr::Constant(a), JITRepr::Constant(b)) => {
                        Ok(JITRepr::Constant(a.powf(*b)))
                    }
                    _ => Ok(JITRepr::Pow(Box::new(base_opt), Box::new(exp_opt))),
                }
            }
            // Apply to unary operations
            JITRepr::Neg(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    JITRepr::Constant(a) => Ok(JITRepr::Constant(-a)),
                    _ => Ok(JITRepr::Neg(Box::new(inner_opt))),
                }
            }
            JITRepr::Ln(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    JITRepr::Constant(a) if *a > 0.0 => Ok(JITRepr::Constant(a.ln())),
                    _ => Ok(JITRepr::Ln(Box::new(inner_opt))),
                }
            }
            JITRepr::Exp(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    JITRepr::Constant(a) => Ok(JITRepr::Constant(a.exp())),
                    _ => Ok(JITRepr::Exp(Box::new(inner_opt))),
                }
            }
            JITRepr::Sin(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    JITRepr::Constant(a) => Ok(JITRepr::Constant(a.sin())),
                    _ => Ok(JITRepr::Sin(Box::new(inner_opt))),
                }
            }
            JITRepr::Cos(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    JITRepr::Constant(a) => Ok(JITRepr::Constant(a.cos())),
                    _ => Ok(JITRepr::Cos(Box::new(inner_opt))),
                }
            }
            JITRepr::Sqrt(inner) => {
                let inner_opt = Self::apply_constant_folding(inner)?;
                match &inner_opt {
                    JITRepr::Constant(a) if *a >= 0.0 => Ok(JITRepr::Constant(a.sqrt())),
                    _ => Ok(JITRepr::Sqrt(Box::new(inner_opt))),
                }
            }
            JITRepr::Constant(_) | JITRepr::Variable(_) => Ok(expr.clone()),
        }
    }

    /// Check if two expressions are structurally equal
    fn expressions_equal(a: &JITRepr<f64>, b: &JITRepr<f64>) -> bool {
        match (a, b) {
            (JITRepr::Constant(a), JITRepr::Constant(b)) => (a - b).abs() < f64::EPSILON,
            (JITRepr::Variable(a), JITRepr::Variable(b)) => a == b,
            (JITRepr::Add(a1, a2), JITRepr::Add(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (JITRepr::Mul(a1, a2), JITRepr::Mul(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (JITRepr::Sub(a1, a2), JITRepr::Sub(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (JITRepr::Div(a1, a2), JITRepr::Div(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (JITRepr::Pow(a1, a2), JITRepr::Pow(b1, b2)) => {
                Self::expressions_equal(a1, b1) && Self::expressions_equal(a2, b2)
            }
            (JITRepr::Neg(a), JITRepr::Neg(b)) => Self::expressions_equal(a, b),
            (JITRepr::Ln(a), JITRepr::Ln(b)) => Self::expressions_equal(a, b),
            (JITRepr::Exp(a), JITRepr::Exp(b)) => Self::expressions_equal(a, b),
            (JITRepr::Sin(a), JITRepr::Sin(b)) => Self::expressions_equal(a, b),
            (JITRepr::Cos(a), JITRepr::Cos(b)) => Self::expressions_equal(a, b),
            (JITRepr::Sqrt(a), JITRepr::Sqrt(b)) => Self::expressions_equal(a, b),
            _ => false,
        }
    }
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Enable aggressive optimizations that might change numerical behavior
    pub aggressive: bool,
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable common subexpression elimination
    pub cse: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            aggressive: false,
            constant_folding: true,
            cse: true,
        }
    }
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Number of rules applied
    pub rules_applied: usize,
    /// Optimization time in microseconds
    pub optimization_time_us: u64,
    /// Number of nodes before optimization
    pub nodes_before: usize,
    /// Number of nodes after optimization
    pub nodes_after: usize,
}

/// Trait for optimizable expressions
pub trait OptimizeExpr {
    /// The representation type
    type Repr<T>;

    /// Optimize an expression using symbolic rewrite rules
    fn optimize(expr: Self::Repr<f64>) -> Result<Self::Repr<f64>>;

    /// Optimize with custom configuration
    fn optimize_with_config(
        expr: Self::Repr<f64>,
        config: OptimizationConfig,
    ) -> Result<(Self::Repr<f64>, OptimizationStats)>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{JITEval, JITMathExpr};

    #[test]
    fn test_symbolic_optimizer_creation() {
        let optimizer = SymbolicOptimizer::new();
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_arithmetic_identity_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test x + 0 = x
        let expr = JITEval::add(JITEval::var("x"), JITEval::constant(0.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            JITRepr::Variable(name) => assert_eq!(name, "x"),
            _ => panic!("Expected variable x, got {optimized:?}"),
        }
    }

    #[test]
    fn test_constant_folding() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test 2 + 3 = 5
        let expr = JITEval::add(JITEval::constant(2.0), JITEval::constant(3.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            JITRepr::Constant(value) => assert!((value - 5.0).abs() < f64::EPSILON),
            _ => panic!("Expected constant 5.0, got {optimized:?}"),
        }
    }

    #[test]
    fn test_power_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test x^1 = x
        let expr = JITEval::pow(JITEval::var("x"), JITEval::constant(1.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            JITRepr::Variable(name) => assert_eq!(name, "x"),
            _ => panic!("Expected variable x, got {optimized:?}"),
        }
    }

    #[test]
    fn test_transcendental_optimization() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();

        // Test ln(1) = 0
        let expr = JITEval::ln(JITEval::constant(1.0));
        let optimized = optimizer.optimize(&expr).unwrap();

        match optimized {
            JITRepr::Constant(value) => assert!((value - 0.0).abs() < f64::EPSILON),
            _ => panic!("Expected constant 0.0, got {optimized:?}"),
        }
    }

    #[test]
    fn test_compilation_strategy_creation() {
        // Test default strategy
        let optimizer = SymbolicOptimizer::new().unwrap();
        assert_eq!(
            *optimizer.compilation_strategy(),
            CompilationStrategy::CraneliftJIT
        );

        // Test custom strategy
        let strategy = CompilationStrategy::HotLoadRust {
            source_dir: std::path::PathBuf::from("/tmp/test"),
            lib_dir: std::path::PathBuf::from("/tmp/test_libs"),
            opt_level: RustOptLevel::O2,
        };
        let optimizer = SymbolicOptimizer::with_strategy(strategy.clone()).unwrap();
        assert_eq!(*optimizer.compilation_strategy(), strategy);
    }

    #[test]
    fn test_compilation_approach_selection() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));

        // Test Cranelift strategy
        let approach = optimizer.choose_compilation_approach(&expr, "test_expr");
        assert_eq!(approach, CompilationApproach::Cranelift);

        // Test adaptive strategy
        optimizer.set_compilation_strategy(CompilationStrategy::Adaptive {
            call_threshold: 5,
            complexity_threshold: 10,
        });

        // First few calls should use Cranelift
        for i in 0..5 {
            let approach = optimizer.choose_compilation_approach(&expr, "adaptive_expr");
            if i < 4 {
                assert_eq!(approach, CompilationApproach::Cranelift);
            }
            optimizer.record_execution("adaptive_expr", 1000);
        }

        // After threshold, should upgrade to Rust
        let approach = optimizer.choose_compilation_approach(&expr, "adaptive_expr");
        assert_eq!(approach, CompilationApproach::UpgradeToRust);

        // Subsequent calls should use Rust
        let approach = optimizer.choose_compilation_approach(&expr, "adaptive_expr");
        assert_eq!(approach, CompilationApproach::RustHotLoad);
    }

    #[test]
    fn test_rust_source_generation() {
        let optimizer = SymbolicOptimizer::new().unwrap();

        // Test simple expression: x + 1
        let expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));
        let source = optimizer.generate_rust_source(&expr, "test_func").unwrap();

        assert!(source.contains("#[no_mangle]"));
        assert!(source.contains("pub extern \"C\" fn test_func(x: f64) -> f64"));
        assert!(source.contains("(x + 1)"));
    }

    #[test]
    fn test_strategy_recommendation() {
        // Simple expression should use Cranelift
        let simple_expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));
        let strategy = SymbolicOptimizer::recommend_strategy(&simple_expr);
        assert_eq!(strategy, CompilationStrategy::CraneliftJIT);

        // Complex expression should use adaptive or Rust hot-loading
        let complex_expr = {
            let mut expr = JITEval::var("x");
            // Create a complex expression with many operations
            for i in 1..60 {
                expr = JITEval::add(
                    expr,
                    JITEval::sin(JITEval::mul(
                        JITEval::var("x"),
                        JITEval::constant(f64::from(i)),
                    )),
                );
            }
            expr
        };

        let strategy = SymbolicOptimizer::recommend_strategy(&complex_expr);
        match strategy {
            CompilationStrategy::HotLoadRust { .. } => {} // Expected for very complex expressions
            CompilationStrategy::Adaptive { .. } => {}    // Also acceptable
            _ => panic!("Expected adaptive or hot-load strategy for complex expression"),
        }
    }

    #[test]
    fn test_execution_statistics() {
        let mut optimizer = SymbolicOptimizer::new().unwrap();
        let _expr: JITRepr<f64> = JITEval::var("x");

        // Initialize the expression stats by calling choose_compilation_approach first
        let simple_expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));
        optimizer.choose_compilation_approach(&simple_expr, "test_expr");

        // Record some executions
        optimizer.record_execution("test_expr", 1000);
        optimizer.record_execution("test_expr", 2000);
        optimizer.record_execution("test_expr", 1500);

        let stats = optimizer.get_expression_stats();
        let expr_stats = stats
            .get("test_expr")
            .expect("Expression stats should exist");

        assert_eq!(expr_stats.call_count, 3);
        assert!(expr_stats.avg_execution_time_ns > 0.0);
    }
}
