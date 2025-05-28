//! JIT Compilation Module
//!
//! This module provides JIT compilation capabilities using Cranelift.

#[cfg(feature = "cranelift")]
use cranelift::prelude::*;
#[cfg(feature = "cranelift")]
use cranelift_codegen::ir::Function;
#[cfg(feature = "cranelift")]
use cranelift_codegen::Context;
#[cfg(feature = "cranelift")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "cranelift")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "cranelift")]
use cranelift_module::{Linkage, Module};
#[cfg(feature = "cranelift")]
use std::collections::HashMap;

use crate::error::{MathCompileError, Result};
use crate::final_tagless::ASTRepr;

/// Generate Cranelift IR for evaluating a polynomial using Horner's method
#[cfg(feature = "cranelift")]
fn generate_polynomial_ir(builder: &mut FunctionBuilder, x: Value, coeffs: &[f64]) -> Value {
    if coeffs.is_empty() {
        return builder.ins().f64const(0.0);
    }

    // Start with the highest degree coefficient
    let mut result = builder.ins().f64const(coeffs[coeffs.len() - 1]);

    // Apply Horner's method: result = result * x + coeff[i]
    for &coeff in coeffs.iter().rev().skip(1) {
        result = builder.ins().fmul(result, x);
        let coeff_val = builder.ins().f64const(coeff);
        result = builder.ins().fadd(result, coeff_val);
    }

    result
}

/// Generate Cranelift IR for evaluating a rational function
#[cfg(feature = "cranelift")]
fn generate_rational_ir(
    builder: &mut FunctionBuilder,
    x: Value,
    num_coeffs: &[f64],
    den_coeffs: &[f64],
) -> Value {
    let numerator = generate_polynomial_ir(builder, x, num_coeffs);
    let denominator = generate_polynomial_ir(builder, x, den_coeffs);
    builder.ins().fdiv(numerator, denominator)
}

/// Generate Cranelift IR for ln(1+x) for x ∈ [0,1]
/// Max error: 6.248044858924071e-12
#[cfg(feature = "cranelift")]
fn generate_ln_1plus_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        6.248044858924071e-12,
        0.9999999985585198,
        1.3031632785795166,
        0.4385659053064146,
        0.03085953976409006,
    ];
    let den_coeffs = [
        1.0,
        1.8031632248969947,
        1.0068149572238094,
        0.18320686065538652,
        0.0068149572238094085,
    ];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

/// Generate Cranelift IR for exp(x) for x ∈ [-1,1]
/// Max error: 4.249646209318276e-12
#[cfg(feature = "cranelift")]
fn generate_exp_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        0.9999999999980661,
        0.44594866665439437,
        0.08394001153724977,
        0.008028602369117902,
        0.0003359093826009222,
    ];
    let den_coeffs = [
        1.0,
        -0.5540513333089334,
        0.13799134473142305,
        -0.01960374294724866,
        0.0016192031795560164,
        -6.374775984025426e-5,
    ];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

/// Generate Cranelift IR for cos(x) for x ∈ [0, π/4]
/// Max error: 8.492520741606233e-11
#[cfg(feature = "cranelift")]
fn generate_cos_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    let num_coeffs = [
        1.0000000000849252,
        -0.04419808517009371,
        -0.468545034572871,
        0.022095248245365844,
        0.025958373239365604,
        -0.0018934016585943506,
    ];
    let den_coeffs = [1.0, -0.04419807131962928, 0.03145459448704991];
    generate_rational_ir(builder, x, &num_coeffs, &den_coeffs)
}

/// Generate Cranelift IR for sin(x) using shifted cosine: sin(x) = cos(π/2 - x)
/// This leverages our high-precision cosine implementation
#[cfg(feature = "cranelift")]
fn generate_sin_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    // sin(x) = cos(π/2 - x)
    let pi_over_2 = builder.ins().f64const(std::f64::consts::PI / 2.0);
    let shifted_x = builder.ins().fsub(pi_over_2, x);
    // Use absolute value since cos(-x) = cos(x)
    let abs_shifted_x = builder.ins().fabs(shifted_x);
    generate_cos_ir(builder, abs_shifted_x)
}

/// Generate efficient Cranelift IR for integer powers using optimal multiplication sequences
#[cfg(feature = "cranelift")]
fn generate_integer_power_ir(builder: &mut FunctionBuilder, base: Value, exp: i32) -> Value {
    match exp {
        0 => builder.ins().f64const(1.0), // x^0 = 1
        1 => base,                        // x^1 = x
        -1 => {
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, base) // x^-1 = 1/x
        }
        2 => builder.ins().fmul(base, base), // x^2 = x*x
        -2 => {
            let x_squared = builder.ins().fmul(base, base);
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, x_squared) // x^-2 = 1/(x*x)
        }
        3 => {
            let x_squared = builder.ins().fmul(base, base);
            builder.ins().fmul(x_squared, base) // x^3 = x²*x
        }
        4 => {
            let x_squared = builder.ins().fmul(base, base);
            builder.ins().fmul(x_squared, x_squared) // x^4 = (x²)²
        }
        5 => {
            let x_squared = builder.ins().fmul(base, base);
            let x_fourth = builder.ins().fmul(x_squared, x_squared);
            builder.ins().fmul(x_fourth, base) // x^5 = x⁴*x
        }
        6 => {
            let x_squared = builder.ins().fmul(base, base);
            let x_cubed = builder.ins().fmul(x_squared, base);
            builder.ins().fmul(x_cubed, x_cubed) // x^6 = (x³)²
        }
        7 => {
            let x_squared = builder.ins().fmul(base, base);
            let x_fourth = builder.ins().fmul(x_squared, x_squared);
            let x_sixth = builder.ins().fmul(x_fourth, x_squared);
            builder.ins().fmul(x_sixth, base) // x^7 = x⁶*x
        }
        8 => {
            let x_squared = builder.ins().fmul(base, base);
            let x_fourth = builder.ins().fmul(x_squared, x_squared);
            builder.ins().fmul(x_fourth, x_fourth) // x^8 = (x⁴)²
        }
        10 => {
            let x_squared = builder.ins().fmul(base, base);
            let x_fourth = builder.ins().fmul(x_squared, x_squared);
            let x_fifth = builder.ins().fmul(x_fourth, base);
            builder.ins().fmul(x_fifth, x_fifth) // x^10 = (x^5)^2
        }
        exp if exp > 8 && exp <= 32 => {
            // Use optimized sequences for common larger powers
            match exp {
                9 => {
                    let x_squared = builder.ins().fmul(base, base);
                    let x_fourth = builder.ins().fmul(x_squared, x_squared);
                    let x_eighth = builder.ins().fmul(x_fourth, x_fourth);
                    builder.ins().fmul(x_eighth, base) // x^9 = x^8 * x
                }
                10 => {
                    let x_squared = builder.ins().fmul(base, base);
                    let x_fourth = builder.ins().fmul(x_squared, x_squared);
                    let x_fifth = builder.ins().fmul(x_fourth, base);
                    builder.ins().fmul(x_fifth, x_fifth) // x^10 = (x^5)^2
                }
                12 => {
                    let x_squared = builder.ins().fmul(base, base);
                    let x_cubed = builder.ins().fmul(x_squared, base);
                    let x_sixth = builder.ins().fmul(x_cubed, x_cubed);
                    builder.ins().fmul(x_sixth, x_sixth) // x^12 = (x^6)^2
                }
                16 => {
                    let x_squared = builder.ins().fmul(base, base);
                    let x_fourth = builder.ins().fmul(x_squared, x_squared);
                    let x_eighth = builder.ins().fmul(x_fourth, x_fourth);
                    builder.ins().fmul(x_eighth, x_eighth) // x^16 = (x^8)^2
                }
                _ => {
                    // For other powers, fall back to exp/ln method
                    let exp_f64 = builder.ins().f64const(f64::from(exp));
                    let one = builder.ins().f64const(1.0);
                    let u = builder.ins().fsub(base, one);
                    let ln_base = generate_ln_1plus_ir(builder, u);
                    let product = builder.ins().fmul(exp_f64, ln_base);
                    generate_exp_ir(builder, product)
                }
            }
        }
        exp if (-32..0).contains(&exp) => {
            // Handle negative exponents: x^-n = 1/(x^n)
            let positive_power = generate_integer_power_ir(builder, base, -exp);
            let one = builder.ins().f64const(1.0);
            builder.ins().fdiv(one, positive_power)
        }
        _ => {
            // Fallback for very large exponents - shouldn't happen with our range check
            builder.ins().f64const(1.0)
        }
    }
}

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

impl From<JITError> for MathCompileError {
    fn from(err: JITError) -> Self {
        MathCompileError::JITError(err.to_string())
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
#[cfg(feature = "cranelift")]
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

#[cfg(feature = "cranelift")]
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
#[cfg(feature = "cranelift")]
pub struct JITCompiler {
    module: JITModule,
    builder_context: FunctionBuilderContext,
}

#[cfg(feature = "cranelift")]
impl JITCompiler {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| MathCompileError::JITError(format!("Failed to set Cranelift flags: {e}")))?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| MathCompileError::JITError(format!("Failed to set Cranelift flags: {e}")))?;
        let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .map_err(|e| MathCompileError::JITError(format!("Failed to create ISA: {e}")))?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| MathCompileError::JITError(format!("Failed to create ISA: {e}")))?;

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
        expr: &ASTRepr<f64>,
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
            .map_err(|e| MathCompileError::JITError(format!("Failed to declare function: {e}")))?;

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
            .map_err(|e| MathCompileError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| MathCompileError::JITError(format!("Failed to finalize definitions: {e}")))?;

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
        expr: &ASTRepr<f64>,
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
            .map_err(|e| MathCompileError::JITError(format!("Failed to declare function: {e}")))?;

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
            .map_err(|e| MathCompileError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| MathCompileError::JITError(format!("Failed to finalize definitions: {e}")))?;

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
        expr: &ASTRepr<f64>,
        var_names: &[&str],
    ) -> Result<JITFunction> {
        if var_names.is_empty() {
            return Err(MathCompileError::JITError(
                "At least one variable required".to_string(),
            ));
        }
        if var_names.len() > 6 {
            return Err(MathCompileError::JITError(format!(
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
            .map_err(|e| MathCompileError::JITError(format!("Failed to declare function: {e}")))?;

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
            .map_err(|e| MathCompileError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| MathCompileError::JITError(format!("Failed to finalize definitions: {e}")))?;

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
#[cfg(feature = "cranelift")]
fn generate_ir_for_expr(
    builder: &mut FunctionBuilder,
    expr: &ASTRepr<f64>,
    var_map: &HashMap<String, Value>,
) -> Result<Value> {
    match expr {
        ASTRepr::Constant(value) => Ok(builder.ins().f64const(*value)),
        ASTRepr::Variable(name) => var_map
            .get(name)
            .copied()
            .ok_or_else(|| MathCompileError::JITError(format!("Unknown variable: {name}"))),
        ASTRepr::Add(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fadd(left_val, right_val))
        }
        ASTRepr::Sub(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fsub(left_val, right_val))
        }
        ASTRepr::Mul(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fmul(left_val, right_val))
        }
        ASTRepr::Div(left, right) => {
            let left_val = generate_ir_for_expr(builder, left, var_map)?;
            let right_val = generate_ir_for_expr(builder, right, var_map)?;
            Ok(builder.ins().fdiv(left_val, right_val))
        }
        ASTRepr::Neg(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            Ok(builder.ins().fneg(inner_val))
        }
        ASTRepr::Pow(base, exp) => {
            let base_val = generate_ir_for_expr(builder, base, var_map)?;

            // Check if exponent is a constant for optimization
            if let ASTRepr::Constant(exp_const) = exp.as_ref() {
                // Handle integer exponents efficiently
                if exp_const.fract() == 0.0 && exp_const.abs() <= 32.0 {
                    let exp_int = *exp_const as i32;
                    return Ok(generate_integer_power_ir(builder, base_val, exp_int));
                }

                // Handle common fractional exponents
                if (*exp_const - 0.5).abs() < f64::EPSILON {
                    Ok(builder.ins().sqrt(base_val)) // x^0.5 = sqrt(x)
                } else if (*exp_const + 0.5).abs() < f64::EPSILON {
                    let sqrt_val = builder.ins().sqrt(base_val);
                    let one = builder.ins().f64const(1.0);
                    Ok(builder.ins().fdiv(one, sqrt_val)) // x^-0.5 = 1/sqrt(x)
                } else if (*exp_const - 1.0 / 3.0).abs() < f64::EPSILON {
                    // Cube root using x^(1/3) = exp(ln(x)/3)
                    let one_third = builder.ins().f64const(1.0 / 3.0);
                    let one = builder.ins().f64const(1.0);
                    let u = builder.ins().fsub(base_val, one);
                    let ln_x = generate_ln_1plus_ir(builder, u);
                    let ln_x_div_3 = builder.ins().fmul(ln_x, one_third);
                    Ok(generate_exp_ir(builder, ln_x_div_3))
                } else {
                    // For other constant exponents, use exp(exp * ln(base))
                    let exp_val = generate_ir_for_expr(builder, exp, var_map)?;
                    let one = builder.ins().f64const(1.0);
                    let u = builder.ins().fsub(base_val, one);
                    let ln_base = generate_ln_1plus_ir(builder, u);
                    let product = builder.ins().fmul(exp_val, ln_base);
                    Ok(generate_exp_ir(builder, product))
                }
            } else {
                // For variable exponents, use exp(exp * ln(base))
                let exp_val = generate_ir_for_expr(builder, exp, var_map)?;
                let one = builder.ins().f64const(1.0);
                let u = builder.ins().fsub(base_val, one);
                let ln_base = generate_ln_1plus_ir(builder, u);
                let product = builder.ins().fmul(exp_val, ln_base);
                Ok(generate_exp_ir(builder, product))
            }
        }
        ASTRepr::Ln(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Use optimal rational approximation for ln(1+x)
            // For ln(x), we compute ln(1 + (x-1)) = ln(1 + u) where u = x-1
            let one = builder.ins().f64const(1.0);
            let u = builder.ins().fsub(inner_val, one);
            Ok(generate_ln_1plus_ir(builder, u))
        }
        ASTRepr::Exp(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Use optimal rational approximation for exp(x) on [-1, 1]
            Ok(generate_exp_ir(builder, inner_val))
        }
        ASTRepr::Sqrt(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            Ok(builder.ins().sqrt(inner_val))
        }
        ASTRepr::Sin(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Use shifted cosine implementation
            Ok(generate_sin_ir(builder, inner_val))
        }
        ASTRepr::Cos(inner) => {
            let inner_val = generate_ir_for_expr(builder, inner, var_map)?;
            // Use optimal rational approximation for cos(x) on [0, π/4]
            // For negative values, use cos(-x) = cos(x)
            let abs_val = builder.ins().fabs(inner_val);
            Ok(generate_cos_ir(builder, abs_val))
        }
    }
}

// Provide stub implementations when JIT feature is disabled
#[cfg(not(feature = "cranelift"))]
pub struct JITFunction;

#[cfg(not(feature = "cranelift"))]
pub struct JITCompiler;

#[cfg(not(feature = "cranelift"))]
impl JITCompiler {
    pub fn new() -> Result<Self> {
        Err(MathCompileError::FeatureNotEnabled("cranelift".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_jit_compiler_creation() {
        let compiler = JITCompiler::new();
        assert!(compiler.is_ok());
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_simple_jit_compilation() {
        // Create a simple expression: x + 1
        let expr = ASTEval::add(ASTEval::var("x"), ASTEval::constant(1.0));

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_single_var(&expr, "x").unwrap();

        // Test the compiled function
        let result = jit_func.call_single(2.0);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_two_variable_jit_compilation() {
        // Create a two-variable expression: x + y
        let expr = ASTEval::add(ASTEval::var("x"), ASTEval::var("y"));

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
    #[cfg(feature = "cranelift")]
    fn test_two_variable_complex_expression() {
        // Create a more complex two-variable expression: x² + 2*x*y + y²
        let x = ASTEval::var("x");
        let y = ASTEval::var("y");
        let expr = ASTEval::add(
            ASTEval::add(
                ASTEval::pow(x.clone(), ASTEval::constant(2.0)),
                ASTEval::mul(ASTEval::mul(ASTEval::constant(2.0), x), y.clone()),
            ),
            ASTEval::pow(y, ASTEval::constant(2.0)),
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
    #[cfg(feature = "cranelift")]
    fn test_multi_variable_jit_compilation() {
        // Create a three-variable expression: x + y + z
        let expr = ASTEval::add(
            ASTEval::add(ASTEval::var("x"), ASTEval::var("y")),
            ASTEval::var("z"),
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
    #[cfg(feature = "cranelift")]
    fn test_multi_variable_complex_expression() {
        // Create a complex multi-variable expression: x*y + y*z + z*x
        let x = ASTEval::var("x");
        let y = ASTEval::var("y");
        let z = ASTEval::var("z");
        let expr = ASTEval::add(
            ASTEval::add(
                ASTEval::mul(x.clone(), y.clone()),
                ASTEval::mul(y, z.clone()),
            ),
            ASTEval::mul(z, x),
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
    #[cfg(feature = "cranelift")]
    fn test_multi_variable_error_cases() {
        let expr = ASTEval::var("x");
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
    #[cfg(feature = "cranelift")]
    fn test_variable_count_limits() {
        // Test maximum supported variables (6)
        let expr = ASTEval::add(
            ASTEval::add(
                ASTEval::add(
                    ASTEval::add(
                        ASTEval::add(ASTEval::var("x1"), ASTEval::var("x2")),
                        ASTEval::var("x3"),
                    ),
                    ASTEval::var("x4"),
                ),
                ASTEval::var("x5"),
            ),
            ASTEval::var("x6"),
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
    #[cfg(not(feature = "cranelift"))]
    fn test_jit_disabled() {
        let compiler = JITCompiler::new();
        assert!(compiler.is_err());
    }
}
