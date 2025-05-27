//! JIT Compilation for Final Tagless Mathematical Expressions
//!
//! This module provides JIT compilation capabilities using Cranelift for high-performance
//! evaluation of mathematical expressions built with the final tagless approach.

#[cfg(feature = "jit")]
use cranelift_codegen::ir::{InstBuilder, Value};
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{Linkage, Module};
#[cfg(feature = "jit")]
use std::collections::HashMap;

use crate::error::{MathJITError, Result};
use crate::final_tagless::JITRepr;

/// JIT compilation errors
#[derive(Debug)]
pub enum JITError {
    /// Cranelift compilation error
    CompilationError(String),
    /// Unsupported expression type
    UnsupportedExpression(String),
    /// Memory allocation error
    MemoryError(String),
    /// Module error
    ModuleError(String),
}

impl std::fmt::Display for JITError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JITError::CompilationError(msg) => write!(f, "Compilation error: {msg}"),
            JITError::UnsupportedExpression(msg) => write!(f, "Unsupported expression: {msg}"),
            JITError::MemoryError(msg) => write!(f, "Memory error: {msg}"),
            JITError::ModuleError(msg) => write!(f, "Module error: {msg}"),
        }
    }
}

impl std::error::Error for JITError {}

impl From<JITError> for MathJITError {
    fn from(err: JITError) -> Self {
        MathJITError::JITError(err.to_string())
    }
}

/// JIT function signature types
#[derive(Debug, Clone)]
pub enum JITSignature {
    /// Single input: f(x) -> f64
    SingleInput,
    /// Two variables: f(x, y) -> f64
    TwoVariables,
    /// Multiple variables: f(x₁, x₂, ..., xₙ) -> f64
    MultipleVariables(usize),
    /// Data and single parameter: f(x, θ) -> f64
    DataAndParameter,
    /// Data and parameter vector: f(x, θ₁, θ₂, ..., θₙ) -> f64
    DataAndParameters(usize),
}

/// Compiled JIT function
#[cfg(feature = "jit")]
pub struct JITFunction {
    /// Function pointer to the compiled native code
    function_ptr: *const u8,
    /// The JIT module (kept alive to prevent deallocation)
    _module: JITModule,
    /// Function signature information
    pub signature: JITSignature,
    /// Compilation statistics
    pub stats: CompilationStats,
}

/// Compilation statistics
#[derive(Debug, Clone)]
pub struct CompilationStats {
    /// Size of generated machine code in bytes
    pub code_size_bytes: usize,
    /// Number of operations in the expression
    pub operation_count: usize,
    /// Compilation time in microseconds
    pub compilation_time_us: u64,
    /// Number of variables in the expression
    pub variable_count: usize,
}

#[cfg(feature = "jit")]
impl JITFunction {
    /// Call the compiled function with a single input
    pub fn call_single(&self, x: f64) -> f64 {
        match self.signature {
            JITSignature::SingleInput => {
                let func: extern "C" fn(f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(x)
            }
            _ => panic!("Invalid signature for single input call"),
        }
    }

    /// Call the compiled function with two variables
    pub fn call_two_vars(&self, x: f64, y: f64) -> f64 {
        match self.signature {
            JITSignature::TwoVariables => {
                let func: extern "C" fn(f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(x, y)
            }
            _ => panic!("Invalid signature for two variable call"),
        }
    }

    /// Call the compiled function with multiple variables
    pub fn call_multi_vars(&self, vars: &[f64]) -> f64 {
        match &self.signature {
            JITSignature::MultipleVariables(n) => {
                assert!(
                    (vars.len() == *n),
                    "Variable count mismatch: expected {}, got {}",
                    n,
                    vars.len()
                );
                // Support up to 6 variables for now
                match n {
                    1 => {
                        let func: extern "C" fn(f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(vars[0])
                    }
                    2 => {
                        let func: extern "C" fn(f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(vars[0], vars[1])
                    }
                    3 => {
                        let func: extern "C" fn(f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(vars[0], vars[1], vars[2])
                    }
                    4 => {
                        let func: extern "C" fn(f64, f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(vars[0], vars[1], vars[2], vars[3])
                    }
                    5 => {
                        let func: extern "C" fn(f64, f64, f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(vars[0], vars[1], vars[2], vars[3], vars[4])
                    }
                    6 => {
                        let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(vars[0], vars[1], vars[2], vars[3], vars[4], vars[5])
                    }
                    _ => panic!("Unsupported variable count: {n} (max 6)"),
                }
            }
            _ => panic!("Invalid signature for multi-variable call"),
        }
    }

    /// Call the compiled function with data and parameter
    pub fn call_data_param(&self, x: f64, theta: f64) -> f64 {
        match self.signature {
            JITSignature::DataAndParameter => {
                let func: extern "C" fn(f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(x, theta)
            }
            _ => panic!("Invalid signature for data-parameter call"),
        }
    }

    /// Call the compiled function with data and multiple parameters
    pub fn call_data_params(&self, x: f64, params: &[f64]) -> f64 {
        match &self.signature {
            JITSignature::DataAndParameters(n) => {
                assert!(
                    (params.len() == *n),
                    "Parameter count mismatch: expected {}, got {}",
                    n,
                    params.len()
                );
                // For now, support up to 4 parameters
                match n {
                    1 => {
                        let func: extern "C" fn(f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(x, params[0])
                    }
                    2 => {
                        let func: extern "C" fn(f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(x, params[0], params[1])
                    }
                    3 => {
                        let func: extern "C" fn(f64, f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(x, params[0], params[1], params[2])
                    }
                    4 => {
                        let func: extern "C" fn(f64, f64, f64, f64, f64) -> f64 =
                            unsafe { std::mem::transmute(self.function_ptr) };
                        func(x, params[0], params[1], params[2], params[3])
                    }
                    _ => panic!("Unsupported parameter count: {n}"),
                }
            }
            _ => panic!("Invalid signature for data-parameters call"),
        }
    }
}

/// JIT compiler for mathematical expressions
#[cfg(feature = "jit")]
pub struct JITCompiler {
    module: JITModule,
    builder_context: FunctionBuilderContext,
}

#[cfg(feature = "jit")]
impl JITCompiler {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| MathJITError::JITError(format!("Failed to set Cranelift flags: {e}")))?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| MathJITError::JITError(format!("Failed to set Cranelift flags: {e}")))?;
        let isa_builder = cranelift_native::builder()
            .map_err(|e| MathJITError::JITError(format!("Failed to create ISA builder: {e}")))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| MathJITError::JITError(format!("Failed to create ISA: {e}")))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        let builder_context = FunctionBuilderContext::new();

        Ok(Self {
            module,
            builder_context,
        })
    }

    /// Compile a JIT representation to a native function
    pub fn compile_single_var(
        mut self,
        expr: &JITRepr<f64>,
        var_name: &str,
    ) -> Result<JITFunction> {
        let start_time = std::time::Instant::now();

        // Create function signature: f(x: f64) -> f64
        let mut sig = self.module.make_signature();
        sig.params.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::F64,
        ));
        sig.returns.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::F64,
        ));

        // Create function
        let func_id = self
            .module
            .declare_function("jit_func", Linkage::Export, &sig)
            .map_err(|e| MathJITError::JITError(format!("Failed to declare function: {e}")))?;

        // Build function body using Context
        let mut ctx = cranelift_codegen::Context::new();
        ctx.func.signature = sig;
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_context);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get the input parameter (x)
            let x_val = builder.block_params(entry_block)[0];

            // Create variable map
            let mut var_map = HashMap::new();
            var_map.insert(var_name.to_string(), x_val);

            // Generate IR for the expression
            let result = generate_ir_for_expr(&mut builder, expr, &var_map)?;

            // Return the result
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile the function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| MathJITError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| MathJITError::JITError(format!("Failed to finalize definitions: {e}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        let compilation_time = start_time.elapsed();
        let stats = CompilationStats {
            code_size_bytes: 128, // Estimate - Cranelift doesn't provide exact size easily
            operation_count: expr.count_operations(),
            compilation_time_us: compilation_time.as_micros() as u64,
            variable_count: 1,
        };

        Ok(JITFunction {
            function_ptr: code_ptr,
            _module: self.module,
            signature: JITSignature::SingleInput,
            stats,
        })
    }

    /// Compile a JIT representation to a native function with two variables
    pub fn compile_two_vars(
        mut self,
        expr: &JITRepr<f64>,
        var1_name: &str,
        var2_name: &str,
    ) -> Result<JITFunction> {
        let start_time = std::time::Instant::now();

        // Create function signature: f(x: f64, y: f64) -> f64
        let mut sig = self.module.make_signature();
        sig.params.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::F64,
        ));
        sig.params.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::F64,
        ));
        sig.returns.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::F64,
        ));

        // Create function
        let func_id = self
            .module
            .declare_function("jit_func", Linkage::Export, &sig)
            .map_err(|e| MathJITError::JITError(format!("Failed to declare function: {e}")))?;

        // Build function body using Context
        let mut ctx = cranelift_codegen::Context::new();
        ctx.func.signature = sig;
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_context);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get the input parameters
            let block_params = builder.block_params(entry_block);
            let var1_val = block_params[0];
            let var2_val = block_params[1];

            // Create variable map
            let mut var_map = HashMap::new();
            var_map.insert(var1_name.to_string(), var1_val);
            var_map.insert(var2_name.to_string(), var2_val);

            // Generate IR for the expression
            let result = generate_ir_for_expr(&mut builder, expr, &var_map)?;

            // Return the result
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile the function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| MathJITError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| MathJITError::JITError(format!("Failed to finalize definitions: {e}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        let compilation_time = start_time.elapsed();
        let stats = CompilationStats {
            code_size_bytes: 128, // Estimate - Cranelift doesn't provide exact size easily
            operation_count: expr.count_operations(),
            compilation_time_us: compilation_time.as_micros() as u64,
            variable_count: 2,
        };

        Ok(JITFunction {
            function_ptr: code_ptr,
            _module: self.module,
            signature: JITSignature::TwoVariables,
            stats,
        })
    }

    /// Compile a JIT representation to a native function with multiple variables
    pub fn compile_multi_vars(
        mut self,
        expr: &JITRepr<f64>,
        var_names: &[&str],
    ) -> Result<JITFunction> {
        if var_names.is_empty() {
            return Err(MathJITError::JITError(
                "At least one variable required".to_string(),
            ));
        }
        if var_names.len() > 6 {
            return Err(MathJITError::JITError(format!(
                "Too many variables: {} (max 6)",
                var_names.len()
            )));
        }

        let start_time = std::time::Instant::now();

        // Create function signature: f(x₁: f64, x₂: f64, ..., xₙ: f64) -> f64
        let mut sig = self.module.make_signature();
        for _ in 0..var_names.len() {
            sig.params.push(cranelift_codegen::ir::AbiParam::new(
                cranelift_codegen::ir::types::F64,
            ));
        }
        sig.returns.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::F64,
        ));

        // Create function
        let func_id = self
            .module
            .declare_function("jit_func", Linkage::Export, &sig)
            .map_err(|e| MathJITError::JITError(format!("Failed to declare function: {e}")))?;

        // Build function body using Context
        let mut ctx = cranelift_codegen::Context::new();
        ctx.func.signature = sig;
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_context);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get the input parameters
            let block_params = builder.block_params(entry_block);

            // Create variable map
            let mut var_map = HashMap::new();
            for (i, var_name) in var_names.iter().enumerate() {
                var_map.insert((*var_name).to_string(), block_params[i]);
            }

            // Generate IR for the expression
            let result = generate_ir_for_expr(&mut builder, expr, &var_map)?;

            // Return the result
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile the function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| MathJITError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| MathJITError::JITError(format!("Failed to finalize definitions: {e}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        let compilation_time = start_time.elapsed();
        let stats = CompilationStats {
            code_size_bytes: 128, // Estimate - Cranelift doesn't provide exact size easily
            operation_count: expr.count_operations(),
            compilation_time_us: compilation_time.as_micros() as u64,
            variable_count: var_names.len(),
        };

        Ok(JITFunction {
            function_ptr: code_ptr,
            _module: self.module,
            signature: JITSignature::MultipleVariables(var_names.len()),
            stats,
        })
    }
}

/// Generate Cranelift IR for a JIT representation (standalone function to avoid borrowing issues)
#[cfg(feature = "jit")]
fn generate_ir_for_expr(
    builder: &mut FunctionBuilder,
    expr: &JITRepr<f64>,
    var_map: &HashMap<String, Value>,
) -> Result<Value> {
    match expr {
        JITRepr::Constant(value) => Ok(builder.ins().f64const(*value)),
        JITRepr::Variable(name) => var_map
            .get(name)
            .copied()
            .ok_or_else(|| MathJITError::JITError(format!("Unknown variable: {name}"))),
        JITRepr::Add(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fadd(left_val, right_val))
        }
        JITRepr::Sub(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fsub(left_val, right_val))
        }
        JITRepr::Mul(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fmul(left_val, right_val))
        }
        JITRepr::Div(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fdiv(left_val, right_val))
        }
        JITRepr::Neg(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            Ok(builder.ins().fneg(inner_val))
        }
        JITRepr::Pow(base, exp) => {
            let base_val = generate_ir_for_expr(builder, base, var_map)?;
            let exp_val = generate_ir_for_expr(builder, exp, var_map)?;

            // Check if exponent is a constant for optimization
            if let JITRepr::Constant(exp_const) = exp.as_ref() {
                match *exp_const as i32 {
                    0 => Ok(builder.ins().f64const(1.0)),            // x^0 = 1
                    1 => Ok(base_val),                               // x^1 = x
                    2 => Ok(builder.ins().fmul(base_val, base_val)), // x^2 = x*x
                    3 => {
                        let x_squared = builder.ins().fmul(base_val, base_val);
                        Ok(builder.ins().fmul(x_squared, base_val)) // x^3 = x²*x
                    }
                    _ => {
                        // For other powers, use a simple approximation for now
                        // In a real implementation, we'd call libm pow or implement a proper algorithm
                        Ok(builder.ins().fmul(base_val, exp_val)) // Placeholder for general case
                    }
                }
            } else {
                // For variable exponents, use placeholder
                Ok(builder.ins().fmul(base_val, exp_val)) // Placeholder
            }
        }
        JITRepr::Ln(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Placeholder implementation - in practice we'd use optimized ln
            Ok(inner_val) // Placeholder
        }
        JITRepr::Exp(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Placeholder implementation - in practice we'd use optimized exp
            Ok(inner_val) // Placeholder
        }
        JITRepr::Sqrt(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            Ok(builder.ins().sqrt(inner_val))
        }
        JITRepr::Sin(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Placeholder implementation - in practice we'd use optimized sin
            Ok(inner_val) // Placeholder
        }
        JITRepr::Cos(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Placeholder implementation - in practice we'd use optimized cos
            Ok(inner_val) // Placeholder
        }
    }
}

impl JITRepr<f64> {
    /// Count the number of operations in the expression
    #[must_use]
    pub fn count_operations(&self) -> usize {
        match self {
            JITRepr::Constant(_) | JITRepr::Variable(_) => 0,
            JITRepr::Add(left, right)
            | JITRepr::Sub(left, right)
            | JITRepr::Mul(left, right)
            | JITRepr::Div(left, right)
            | JITRepr::Pow(left, right) => 1 + left.count_operations() + right.count_operations(),
            JITRepr::Neg(inner)
            | JITRepr::Ln(inner)
            | JITRepr::Exp(inner)
            | JITRepr::Sqrt(inner)
            | JITRepr::Sin(inner)
            | JITRepr::Cos(inner) => 1 + inner.count_operations(),
        }
    }
}

// Provide stub implementations when JIT feature is disabled
#[cfg(not(feature = "jit"))]
pub struct JITFunction;

#[cfg(not(feature = "jit"))]
pub struct JITCompiler;

#[cfg(not(feature = "jit"))]
impl JITCompiler {
    pub fn new() -> Result<Self> {
        Err(MathJITError::FeatureNotEnabled("jit".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{JITEval, JITMathExpr};

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_compiler_creation() {
        let compiler = JITCompiler::new();
        assert!(compiler.is_ok());
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_simple_jit_compilation() {
        // Create a simple expression: x + 1
        let expr = JITEval::add(JITEval::var("x"), JITEval::constant(1.0));

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_single_var(&expr, "x").unwrap();

        // Test the compiled function
        let result = jit_func.call_single(2.0);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_two_variable_jit_compilation() {
        // Create a two-variable expression: x + y
        let expr = JITEval::add(JITEval::var("x"), JITEval::var("y"));

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_two_vars(&expr, "x", "y").unwrap();

        // Test the compiled function
        let result = jit_func.call_two_vars(2.0, 3.0);
        assert!((result - 5.0).abs() < 1e-10);

        // Test with different values
        let result2 = jit_func.call_two_vars(1.5, 2.5);
        assert!((result2 - 4.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_two_variable_complex_expression() {
        // Create a more complex two-variable expression: x² + 2*x*y + y²
        let x = JITEval::var("x");
        let y = JITEval::var("y");
        let expr = JITEval::add(
            JITEval::add(
                JITEval::pow(x.clone(), JITEval::constant(2.0)),
                JITEval::mul(JITEval::mul(JITEval::constant(2.0), x), y.clone()),
            ),
            JITEval::pow(y, JITEval::constant(2.0)),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_two_vars(&expr, "x", "y").unwrap();

        // Test the compiled function: (x + y)²
        let result = jit_func.call_two_vars(2.0, 3.0);
        let expected = (2.0_f64 + 3.0_f64).powi(2); // Should be 25.0
        assert!((result - expected).abs() < 1e-10);

        assert_eq!(jit_func.stats.variable_count, 2);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_multi_variable_jit_compilation() {
        // Create a three-variable expression: x + y + z
        let expr = JITEval::add(
            JITEval::add(JITEval::var("x"), JITEval::var("y")),
            JITEval::var("z"),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler
            .compile_multi_vars(&expr, &["x", "y", "z"])
            .unwrap();

        // Test the compiled function
        let result = jit_func.call_multi_vars(&[1.0, 2.0, 3.0]);
        assert!((result - 6.0).abs() < 1e-10);

        // Test with different values
        let result2 = jit_func.call_multi_vars(&[0.5, 1.5, 2.5]);
        assert!((result2 - 4.5).abs() < 1e-10);

        assert_eq!(jit_func.stats.variable_count, 3);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_multi_variable_complex_expression() {
        // Create a complex multi-variable expression: x*y + y*z + z*x
        let x = JITEval::var("x");
        let y = JITEval::var("y");
        let z = JITEval::var("z");
        let expr = JITEval::add(
            JITEval::add(
                JITEval::mul(x.clone(), y.clone()),
                JITEval::mul(y, z.clone()),
            ),
            JITEval::mul(z, x),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler
            .compile_multi_vars(&expr, &["x", "y", "z"])
            .unwrap();

        // Test the compiled function
        let result = jit_func.call_multi_vars(&[2.0, 3.0, 4.0]);
        let expected = 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 2.0; // 6 + 12 + 8 = 26
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_multi_variable_error_cases() {
        let expr = JITEval::var("x");
        let compiler = JITCompiler::new().unwrap();

        // Test empty variable list
        let result = compiler.compile_multi_vars(&expr, &[]);
        assert!(result.is_err());

        // Test too many variables
        let compiler2 = JITCompiler::new().unwrap();
        let too_many_vars = vec!["x1", "x2", "x3", "x4", "x5", "x6", "x7"];
        let result2 = compiler2.compile_multi_vars(&expr, &too_many_vars);
        assert!(result2.is_err());
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_variable_count_limits() {
        // Test maximum supported variables (6)
        let expr = JITEval::add(
            JITEval::add(
                JITEval::add(
                    JITEval::add(
                        JITEval::add(JITEval::var("x1"), JITEval::var("x2")),
                        JITEval::var("x3"),
                    ),
                    JITEval::var("x4"),
                ),
                JITEval::var("x5"),
            ),
            JITEval::var("x6"),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler
            .compile_multi_vars(&expr, &["x1", "x2", "x3", "x4", "x5", "x6"])
            .unwrap();

        // Test the compiled function
        let result = jit_func.call_multi_vars(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!((result - 21.0).abs() < 1e-10); // 1+2+3+4+5+6 = 21

        assert_eq!(jit_func.stats.variable_count, 6);
    }

    #[test]
    #[cfg(not(feature = "jit"))]
    fn test_jit_disabled() {
        let compiler = JITCompiler::new();
        assert!(compiler.is_err());
    }
}
