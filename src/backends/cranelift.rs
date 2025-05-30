//! Cranelift JIT Compilation Backend
//!
//! This module provides JIT compilation capabilities using Cranelift for high-performance
//! evaluation of mathematical expressions built with the final tagless approach.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::time::Instant;

use crate::error::{MathCompileError, Result};
use crate::final_tagless::{ASTRepr, VariableRegistry};
use cranelift::prelude::*;

/// Generate Cranelift IR for evaluating a polynomial using Horner's method
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
fn generate_sin_ir(builder: &mut FunctionBuilder, x: Value) -> Value {
    // sin(x) = cos(π/2 - x)
    let pi_over_2 = builder.ins().f64const(std::f64::consts::PI / 2.0);
    let shifted_x = builder.ins().fsub(pi_over_2, x);
    // Use absolute value since cos(-x) = cos(x)
    let abs_shifted_x = builder.ins().fabs(shifted_x);
    generate_cos_ir(builder, abs_shifted_x)
}

/// Generate efficient Cranelift IR for integer powers using optimal multiplication sequences
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
                    // Fallback: use binary exponentiation for other powers
                    let mut result = builder.ins().f64const(1.0);
                    let mut current_base = base;
                    let mut remaining_exp = exp as u32;

                    while remaining_exp > 0 {
                        if remaining_exp & 1 == 1 {
                            result = builder.ins().fmul(result, current_base);
                        }
                        current_base = builder.ins().fmul(current_base, current_base);
                        remaining_exp >>= 1;
                    }
                    result
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
            JITSignature::MultipleVariables(expected_count) => {
                assert!(
                    vars.len() == *expected_count,
                    "Expected {} variables, got {}",
                    expected_count,
                    vars.len()
                );

                match expected_count {
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
                    _ => panic!("Unsupported number of variables: {expected_count}"),
                }
            }
            _ => panic!("Invalid signature for multi-variable call"),
        }
    }

    /// Call the compiled function with data and a single parameter
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
            JITSignature::DataAndParameters(expected_param_count) => {
                assert!(
                    params.len() == *expected_param_count,
                    "Expected {} parameters, got {}",
                    expected_param_count,
                    params.len()
                );

                match expected_param_count {
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
                    _ => panic!("Unsupported number of parameters: {expected_param_count}"),
                }
            }
            _ => panic!("Invalid signature for data-parameters call"),
        }
    }
}

/// JIT compiler for mathematical expressions
pub struct JITCompiler {
    module: JITModule,
    builder_context: FunctionBuilderContext,
}

impl JITCompiler {
    /// Create a new JIT compiler
    pub fn new() -> Result<Self> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .map_err(|e| MathCompileError::JITError(format!("Failed to create ISA: {e}")))?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| MathCompileError::JITError(format!("Failed to create ISA: {e}")))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            module,
            builder_context: FunctionBuilderContext::new(),
        })
    }

    /// Compile a JIT representation to a native function (backward compatibility)
    pub fn compile_single_var(self, expr: &ASTRepr<f64>, var_name: &str) -> Result<JITFunction> {
        let mut registry = VariableRegistry::new();
        registry.register_variable(var_name);
        self.compile_with_registry(expr, &registry)
    }

    /// Compile a JIT representation to a native function with two variables (backward compatibility)
    pub fn compile_two_vars(
        self,
        expr: &ASTRepr<f64>,
        var1_name: &str,
        var2_name: &str,
    ) -> Result<JITFunction> {
        let mut registry = VariableRegistry::new();
        registry.register_variable(var1_name);
        registry.register_variable(var2_name);
        self.compile_with_registry(expr, &registry)
    }

    /// Compile a JIT representation to a native function with multiple variables (backward compatibility)
    pub fn compile_multi_vars(
        self,
        expr: &ASTRepr<f64>,
        var_names: &[&str],
    ) -> Result<JITFunction> {
        let mut registry = VariableRegistry::new();
        for var_name in var_names {
            registry.register_variable(var_name);
        }
        self.compile_with_registry(expr, &registry)
    }

    /// Compile an expression with a variable registry for proper variable mapping
    pub fn compile_with_registry(
        mut self,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
    ) -> Result<JITFunction> {
        let start_time = Instant::now();

        // Create function signature based on registry
        let mut sig = self.module.make_signature();
        for _ in 0..registry.len() {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));

        let func_id = self
            .module
            .declare_function("compiled_expr", Linkage::Export, &sig)
            .map_err(|e| MathCompileError::JITError(format!("Failed to declare function: {e}")))?;

        // Create function context
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;

        // Build the function
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.builder_context);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let block_params = builder.block_params(entry_block);

            // Create variable map using registry
            let mut var_map = HashMap::new();
            for (i, var_name) in registry.get_all_names().iter().enumerate() {
                var_map.insert(var_name.clone(), block_params[i]);
            }

            // Generate IR for the expression
            let result =
                generate_ir_for_expr_with_registry(&mut builder, expr, &var_map, registry)?;

            // Return the result
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        // Compile the function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| MathCompileError::JITError(format!("Failed to define function: {e}")))?;

        self.module.finalize_definitions().map_err(|e| {
            MathCompileError::JITError(format!("Failed to finalize definitions: {e}"))
        })?;

        let code_ptr = self.module.get_finalized_function(func_id);

        let compilation_time = start_time.elapsed();
        let stats = CompilationStats {
            code_size_bytes: 128, // Estimate - Cranelift doesn't provide exact size easily
            operation_count: expr.count_operations(),
            compilation_time_us: compilation_time.as_micros() as u64,
            variable_count: registry.len(),
        };

        Ok(JITFunction {
            function_ptr: code_ptr,
            _module: self.module,
            signature: match registry.len() {
                1 => JITSignature::SingleInput,
                2 => JITSignature::TwoVariables,
                n => JITSignature::MultipleVariables(n),
            },
            stats,
        })
    }
}

/// Generate Cranelift IR for a JIT representation with variable registry
fn generate_ir_for_expr_with_registry(
    builder: &mut FunctionBuilder,
    expr: &ASTRepr<f64>,
    var_map: &HashMap<String, Value>,
    registry: &VariableRegistry,
) -> Result<Value> {
    match expr {
        ASTRepr::Constant(value) => Ok(builder.ins().f64const(*value)),
        ASTRepr::Variable(index) => {
            // Use the registry to map index to variable name
            if let Some(var_name) = registry.get_name(*index) {
                var_map.get(var_name).copied().ok_or_else(|| {
                    JITError::UnsupportedExpression(format!("Variable {var_name} not found")).into()
                })
            } else {
                Err(JITError::UnsupportedExpression(format!(
                    "Variable index {index} not found in registry"
                ))
                .into())
            }
        }
        ASTRepr::Add(left, right) => {
            let left_val = generate_ir_for_expr_with_registry(builder, left, var_map, registry)?;
            let right_val = generate_ir_for_expr_with_registry(builder, right, var_map, registry)?;
            Ok(builder.ins().fadd(left_val, right_val))
        }
        ASTRepr::Sub(left, right) => {
            let left_val = generate_ir_for_expr_with_registry(builder, left, var_map, registry)?;
            let right_val = generate_ir_for_expr_with_registry(builder, right, var_map, registry)?;
            Ok(builder.ins().fsub(left_val, right_val))
        }
        ASTRepr::Mul(left, right) => {
            let left_val = generate_ir_for_expr_with_registry(builder, left, var_map, registry)?;
            let right_val = generate_ir_for_expr_with_registry(builder, right, var_map, registry)?;
            Ok(builder.ins().fmul(left_val, right_val))
        }
        ASTRepr::Div(left, right) => {
            let left_val = generate_ir_for_expr_with_registry(builder, left, var_map, registry)?;
            let right_val = generate_ir_for_expr_with_registry(builder, right, var_map, registry)?;
            Ok(builder.ins().fdiv(left_val, right_val))
        }
        ASTRepr::Neg(inner) => {
            let inner_val = generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
            Ok(builder.ins().fneg(inner_val))
        }
        ASTRepr::Pow(base, exp) => {
            let base_val = generate_ir_for_expr_with_registry(builder, base, var_map, registry)?;
            let exp_val = generate_ir_for_expr_with_registry(builder, exp, var_map, registry)?;

            // Check if exponent is a constant integer for optimization
            if let ASTRepr::Constant(exp_const) = exp.as_ref() {
                if exp_const.fract() == 0.0 && exp_const.abs() <= 32.0 {
                    let exp_int = *exp_const as i32;
                    return Ok(generate_integer_power_ir(builder, base_val, exp_int));
                }
            }

            // General case: use exp(y * ln(x)) for x^y
            let one = builder.ins().f64const(1.0);
            let base_minus_one = builder.ins().fsub(base_val, one);
            let ln_base = generate_ln_1plus_ir(builder, base_minus_one);
            let exp_ln_base = builder.ins().fmul(exp_val, ln_base);
            Ok(generate_exp_ir(builder, exp_ln_base))
        }
        #[cfg(feature = "logexp")]
        ASTRepr::Log(inner) => {
            let inner_val = generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
            let one = builder.ins().f64const(1.0);
            let x_minus_one = builder.ins().fsub(inner_val, one);
            Ok(generate_ln_1plus_ir(builder, x_minus_one))
        }
        #[cfg(feature = "logexp")]
        ASTRepr::Exp(inner) => {
            let inner_val = generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
            Ok(generate_exp_ir(builder, inner_val))
        }
        ASTRepr::Trig(category) => match &category.function {
            crate::ast::function_categories::TrigFunction::Sin(inner) => {
                let inner_val =
                    generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
                Ok(generate_sin_ir(builder, inner_val))
            }
            crate::ast::function_categories::TrigFunction::Cos(inner) => {
                let inner_val =
                    generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
                Ok(generate_cos_ir(builder, inner_val))
            }
            _ => Err(JITError::UnsupportedExpression(
                "Unsupported trigonometric function for JIT compilation".to_string(),
            )
            .into()),
        },
        #[cfg(feature = "special")]
        ASTRepr::Special(_) => Err(JITError::UnsupportedExpression(
            "Special functions not yet supported in JIT compilation".to_string(),
        )
        .into()),
        #[cfg(feature = "linear_algebra")]
        ASTRepr::LinearAlgebra(_) => Err(JITError::UnsupportedExpression(
            "Linear algebra functions not yet supported in JIT compilation".to_string(),
        )
        .into()),
        ASTRepr::Hyperbolic(category) => {
            match &category.function {
                crate::ast::function_categories::HyperbolicFunction::Sinh(inner) => {
                    // sinh(x) = (e^x - e^(-x)) / 2
                    let inner_val =
                        generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
                    let neg_inner = builder.ins().fneg(inner_val);
                    let exp_pos = generate_exp_ir(builder, inner_val);
                    let exp_neg = generate_exp_ir(builder, neg_inner);
                    let diff = builder.ins().fsub(exp_pos, exp_neg);
                    let two = builder.ins().f64const(2.0);
                    Ok(builder.ins().fdiv(diff, two))
                }
                crate::ast::function_categories::HyperbolicFunction::Cosh(inner) => {
                    // cosh(x) = (e^x + e^(-x)) / 2
                    let inner_val =
                        generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
                    let neg_inner = builder.ins().fneg(inner_val);
                    let exp_pos = generate_exp_ir(builder, inner_val);
                    let exp_neg = generate_exp_ir(builder, neg_inner);
                    let sum = builder.ins().fadd(exp_pos, exp_neg);
                    let two = builder.ins().f64const(2.0);
                    Ok(builder.ins().fdiv(sum, two))
                }
                crate::ast::function_categories::HyperbolicFunction::Tanh(inner) => {
                    // tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                    let inner_val =
                        generate_ir_for_expr_with_registry(builder, inner, var_map, registry)?;
                    let neg_inner = builder.ins().fneg(inner_val);
                    let exp_pos = generate_exp_ir(builder, inner_val);
                    let exp_neg = generate_exp_ir(builder, neg_inner);
                    let numerator = builder.ins().fsub(exp_pos, exp_neg);
                    let denominator = builder.ins().fadd(exp_pos, exp_neg);
                    Ok(builder.ins().fdiv(numerator, denominator))
                }
                _ => Err(JITError::UnsupportedExpression(
                    "Unsupported hyperbolic function for JIT compilation".to_string(),
                )
                .into()),
            }
        }
        #[cfg(feature = "logexp")]
        ASTRepr::LogExp(_) => Err(JITError::UnsupportedExpression(
            "Extended log/exp functions not yet supported in JIT compilation".to_string(),
        )
        .into()),
    }
}

/// Generate Cranelift IR for a JIT representation (standalone function to avoid borrowing issues)
fn generate_ir_for_expr(
    builder: &mut FunctionBuilder,
    expr: &ASTRepr<f64>,
    var_map: &HashMap<String, Value>,
) -> Result<Value> {
    // Create a default registry for backward compatibility
    let mut default_registry = VariableRegistry::new();

    // Extract variable names from the var_map and register them
    for var_name in var_map.keys() {
        default_registry.register_variable(var_name);
    }

    generate_ir_for_expr_with_registry(builder, expr, var_map, &default_registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let compiler = JITCompiler::new();
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_simple_jit_compilation() {
        use crate::final_tagless::{ASTEval, ASTMathExpr};

        let expr = ASTEval::add(
            ASTEval::mul(<ASTEval as ASTMathExpr>::var(0), ASTEval::constant(2.0)), // x * 2
            ASTEval::constant(1.0),
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_single_var(&expr, "x").unwrap();

        let result = jit_func.call_single(5.0);
        assert_eq!(result, 11.0); // 5 * 2 + 1 = 11
    }

    #[test]
    fn test_two_variable_jit_compilation() {
        use crate::final_tagless::{ASTEval, ASTMathExpr};

        let expr = ASTEval::add(
            ASTEval::mul(<ASTEval as ASTMathExpr>::var(0), ASTEval::constant(2.0)), // x * 2
            <ASTEval as ASTMathExpr>::var(1),                                       // + y
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler.compile_two_vars(&expr, "x", "y").unwrap();

        let result = jit_func.call_two_vars(3.0, 4.0);
        assert_eq!(result, 10.0); // 3 * 2 + 4 = 10
    }

    #[test]
    fn test_multi_variable_jit_compilation() {
        use crate::final_tagless::{ASTEval, ASTMathExpr};

        let expr = ASTEval::add(
            ASTEval::add(
                <ASTEval as ASTMathExpr>::var(0),
                <ASTEval as ASTMathExpr>::var(1),
            ), // x + y
            <ASTEval as ASTMathExpr>::var(2), // + z
        );

        let compiler = JITCompiler::new().unwrap();
        let jit_func = compiler
            .compile_multi_vars(&expr, &["x", "y", "z"])
            .unwrap();

        let result = jit_func.call_multi_vars(&[1.0, 2.0, 3.0]);
        assert_eq!(result, 6.0); // 1 + 2 + 3 = 6
    }
}
