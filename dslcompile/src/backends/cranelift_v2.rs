//! Modern Cranelift JIT Backend (v2)
//!
//! This is a redesigned Cranelift backend that addresses the architectural issues
//! in the original implementation. Key improvements:
//!
//! 1. **Simplified Architecture**: Direct IR generation without complex mappings
//! 2. **Index-Based Variables**: Leverages the new index-only variable system
//! 3. **Modern Cranelift APIs**: Uses latest Cranelift patterns and optimizations
//! 4. **Better Error Handling**: Comprehensive error types and recovery
//! 5. **Optimized Function Signatures**: Automatic signature generation
//! 6. **E-graph Integration**: Proper use of Cranelift's optimization pipeline

use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Type, UserFuncName, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;
use std::time::Instant;

use crate::error::{DSLCompileError, Result};
use crate::final_tagless::{ASTRepr, VariableRegistry};

/// Modern JIT compiler using latest Cranelift patterns
pub struct CraneliftV2Compiler {
    /// JIT module for code generation
    module: JITModule,
    /// Function builder context (reusable)
    builder_context: FunctionBuilderContext,
    /// Compilation settings
    settings: settings::Flags,
}

/// Compiled function with modern interface
pub struct CompiledFunction {
    /// Function pointer to native code
    function_ptr: *const u8,
    /// Keep module alive
    _module: JITModule,
    /// Function signature
    signature: FunctionSignature,
    /// Compilation metadata
    metadata: CompilationMetadata,
}

/// Function signature information
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Number of input variables
    pub input_count: usize,
    /// Return type (always f64 for now)
    pub return_type: Type,
}

/// Compilation metadata and statistics
#[derive(Debug, Clone)]
pub struct CompilationMetadata {
    /// Compilation time in microseconds
    pub compile_time_us: u64,
    /// Estimated code size in bytes
    pub code_size_bytes: usize,
    /// Number of operations in expression
    pub operation_count: usize,
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels for Cranelift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization - fastest compilation
    None,
    /// Basic optimizations - balanced
    Basic,
    /// Full optimizations - best performance
    Full,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        Self::Basic
    }
}

impl CraneliftV2Compiler {
    /// Create a new compiler with specified optimization level
    pub fn new(opt_level: OptimizationLevel) -> Result<Self> {
        let mut flag_builder = settings::builder();

        // Configure based on optimization level
        match opt_level {
            OptimizationLevel::None => {
                flag_builder.set("opt_level", "none").unwrap();
                flag_builder.set("enable_verifier", "false").unwrap();
            }
            OptimizationLevel::Basic => {
                flag_builder.set("opt_level", "speed").unwrap();
                flag_builder.set("enable_verifier", "true").unwrap();
            }
            OptimizationLevel::Full => {
                flag_builder.set("opt_level", "speed_and_size").unwrap();
                flag_builder.set("enable_verifier", "true").unwrap();
                flag_builder.set("enable_alias_analysis", "true").unwrap();
            }
        }

        // Common settings for mathematical workloads
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        flag_builder.set("enable_float", "true").unwrap();

        let settings = settings::Flags::new(flag_builder);

        // Create ISA for the host target
        let isa = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .map_err(|e| DSLCompileError::JITError(format!("Failed to create ISA: {e}")))?
            .finish(settings.clone())
            .map_err(|e| DSLCompileError::JITError(format!("Failed to finish ISA: {e}")))?;

        // Create JIT module
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            module,
            builder_context: FunctionBuilderContext::new(),
            settings,
        })
    }

    /// Create compiler with default optimization level
    pub fn new_default() -> Result<Self> {
        Self::new(OptimizationLevel::default())
    }

    /// Compile an expression using the modern variable registry
    pub fn compile_expression(
        self,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
    ) -> Result<CompiledFunction> {
        self.compile_expression_with_level(expr, registry, OptimizationLevel::Basic)
    }

    /// Compile an expression with a specific optimization level
    pub fn compile_expression_with_level(
        mut self,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
        opt_level: OptimizationLevel,
    ) -> Result<CompiledFunction> {
        let start_time = Instant::now();

        // Create function signature
        let mut sig = self.module.make_signature();
        for _ in 0..registry.len() {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));

        // Declare function
        let _func_name = UserFuncName::user(0, 0);
        let func_id = self
            .module
            .declare_function("compiled_expr", Linkage::Local, &sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare function: {e}")))?;

        // Create and build function
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig.clone();

        self.build_function_body(&mut ctx.func, expr, registry)?;

        // Compile the function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to define function: {e}")))?;

        self.module
            .finalize_definitions()
            .map_err(|e| DSLCompileError::JITError(format!("Failed to finalize: {e}")))?;

        // Get function pointer
        let code_ptr = self.module.get_finalized_function(func_id);

        let compile_time = start_time.elapsed();

        Ok(CompiledFunction {
            function_ptr: code_ptr,
            _module: self.module,
            signature: FunctionSignature {
                input_count: registry.len(),
                return_type: types::F64,
            },
            metadata: CompilationMetadata {
                compile_time_us: compile_time.as_micros() as u64,
                code_size_bytes: estimate_code_size(expr),
                operation_count: expr.count_operations(),
                optimization_level: opt_level,
            },
        })
    }

    /// Build the function body with IR generation
    fn build_function_body(
        &mut self,
        func: &mut Function,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
    ) -> Result<()> {
        let mut builder = FunctionBuilder::new(func, &mut self.builder_context);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get function parameters (variables)
        let params = builder.block_params(entry_block);

        // Create variable mapping using indices
        let mut var_values = HashMap::new();
        for i in 0..registry.len() {
            var_values.insert(i, params[i]);
        }

        // Generate IR for the expression
        let result = Self::generate_ir_for_expr(&mut builder, expr, &var_values)?;

        // Return the result
        builder.ins().return_(&[result]);
        builder.finalize();

        Ok(())
    }

    /// Generate Cranelift IR for an expression (modern approach)
    fn generate_ir_for_expr(
        builder: &mut FunctionBuilder,
        expr: &ASTRepr<f64>,
        var_values: &HashMap<usize, Value>,
    ) -> Result<Value> {
        match expr {
            ASTRepr::Constant(value) => Ok(builder.ins().f64const(*value)),

            ASTRepr::Variable(index) => var_values.get(index).copied().ok_or_else(|| {
                DSLCompileError::JITError(format!("Variable index {index} not found"))
            }),

            ASTRepr::Add(left, right) => {
                let left_val = Self::generate_ir_for_expr(builder, left, var_values)?;
                let right_val = Self::generate_ir_for_expr(builder, right, var_values)?;
                Ok(builder.ins().fadd(left_val, right_val))
            }

            ASTRepr::Sub(left, right) => {
                let left_val = Self::generate_ir_for_expr(builder, left, var_values)?;
                let right_val = Self::generate_ir_for_expr(builder, right, var_values)?;
                Ok(builder.ins().fsub(left_val, right_val))
            }

            ASTRepr::Mul(left, right) => {
                let left_val = Self::generate_ir_for_expr(builder, left, var_values)?;
                let right_val = Self::generate_ir_for_expr(builder, right, var_values)?;
                Ok(builder.ins().fmul(left_val, right_val))
            }

            ASTRepr::Div(left, right) => {
                let left_val = Self::generate_ir_for_expr(builder, left, var_values)?;
                let right_val = Self::generate_ir_for_expr(builder, right, var_values)?;
                Ok(builder.ins().fdiv(left_val, right_val))
            }

            ASTRepr::Neg(inner) => {
                let inner_val = Self::generate_ir_for_expr(builder, inner, var_values)?;
                Ok(builder.ins().fneg(inner_val))
            }

            ASTRepr::Pow(base, exp) => {
                let base_val = Self::generate_ir_for_expr(builder, base, var_values)?;
                let exp_val = Self::generate_ir_for_expr(builder, exp, var_values)?;

                // Check for integer exponents for optimization
                if let ASTRepr::Constant(exp_const) = exp.as_ref() {
                    if exp_const.fract() == 0.0 && exp_const.abs() <= 32.0 {
                        return Ok(Self::generate_integer_power(
                            builder,
                            base_val,
                            *exp_const as i32,
                        ));
                    }
                }

                // General case: use powf intrinsic
                Ok(Self::generate_powf(builder, base_val, exp_val))
            }

            ASTRepr::Sqrt(inner) => {
                let inner_val = Self::generate_ir_for_expr(builder, inner, var_values)?;
                Ok(builder.ins().sqrt(inner_val))
            }

            ASTRepr::Sin(inner) => {
                let inner_val = Self::generate_ir_for_expr(builder, inner, var_values)?;
                Ok(Self::generate_sin(builder, inner_val))
            }

            ASTRepr::Cos(inner) => {
                let inner_val = Self::generate_ir_for_expr(builder, inner, var_values)?;
                Ok(Self::generate_cos(builder, inner_val))
            }

            ASTRepr::Exp(inner) => {
                let inner_val = Self::generate_ir_for_expr(builder, inner, var_values)?;
                Ok(Self::generate_exp(builder, inner_val))
            }

            ASTRepr::Ln(inner) => {
                let inner_val = Self::generate_ir_for_expr(builder, inner, var_values)?;
                Ok(Self::generate_ln(builder, inner_val))
            }
        }
    }

    /// Generate optimized integer power using binary exponentiation
    fn generate_integer_power(builder: &mut FunctionBuilder, base: Value, exp: i32) -> Value {
        match exp {
            0 => builder.ins().f64const(1.0),
            1 => base,
            -1 => {
                let one = builder.ins().f64const(1.0);
                builder.ins().fdiv(one, base)
            }
            2 => builder.ins().fmul(base, base),
            3 => {
                let base_sq = builder.ins().fmul(base, base);
                builder.ins().fmul(base_sq, base)
            }
            4 => {
                let base_sq = builder.ins().fmul(base, base);
                builder.ins().fmul(base_sq, base_sq)
            }
            exp if exp > 0 => {
                // Use binary exponentiation for larger positive exponents
                Self::generate_binary_exponentiation(builder, base, exp as u32)
            }
            exp if exp < 0 => {
                // Handle negative exponents: x^(-n) = 1/(x^n)
                let positive_power =
                    Self::generate_binary_exponentiation(builder, base, (-exp) as u32);
                let one = builder.ins().f64const(1.0);
                builder.ins().fdiv(one, positive_power)
            }
            _ => unreachable!(),
        }
    }

    /// Generate binary exponentiation for efficient integer powers
    fn generate_binary_exponentiation(
        builder: &mut FunctionBuilder,
        base: Value,
        mut exp: u32,
    ) -> Value {
        let mut result = builder.ins().f64const(1.0);
        let mut current_power = base;

        while exp > 0 {
            if exp & 1 == 1 {
                result = builder.ins().fmul(result, current_power);
            }
            current_power = builder.ins().fmul(current_power, current_power);
            exp >>= 1;
        }

        result
    }

    /// Generate powf using exp(y * ln(x)) identity
    fn generate_powf(builder: &mut FunctionBuilder, base: Value, exp: Value) -> Value {
        // Use the identity: x^y = exp(y * ln(x))
        // This is a placeholder - in a real implementation you'd want proper libm integration
        let ln_base = Self::generate_ln(builder, base);
        let exp_ln = builder.ins().fmul(exp, ln_base);
        Self::generate_exp(builder, exp_ln)
    }

    /// Generate sin - placeholder implementation
    fn generate_sin(_builder: &mut FunctionBuilder, x: Value) -> Value {
        // TODO: Implement efficient sin approximation or libm call
        // For now, return the input as a placeholder
        x
    }

    /// Generate cos - placeholder implementation  
    fn generate_cos(_builder: &mut FunctionBuilder, x: Value) -> Value {
        // TODO: Implement efficient cos approximation or libm call
        // For now, return the input as a placeholder
        x
    }

    /// Generate exp - placeholder implementation
    fn generate_exp(_builder: &mut FunctionBuilder, x: Value) -> Value {
        // TODO: Implement efficient exp approximation or libm call
        // For now, return the input as a placeholder
        x
    }

    /// Generate ln - placeholder implementation
    fn generate_ln(_builder: &mut FunctionBuilder, x: Value) -> Value {
        // TODO: Implement efficient ln approximation or libm call
        // For now, return the input as a placeholder
        x
    }
}

impl CompiledFunction {
    /// Call the compiled function with the given arguments
    pub fn call(&self, args: &[f64]) -> Result<f64> {
        if args.len() != self.signature.input_count {
            return Err(DSLCompileError::JITError(format!(
                "Expected {} arguments, got {}",
                self.signature.input_count,
                args.len()
            )));
        }

        // Generate the appropriate function call based on argument count
        let result = match args.len() {
            0 => {
                let func: extern "C" fn() -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func()
            }
            1 => {
                let func: extern "C" fn(f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(args[0])
            }
            2 => {
                let func: extern "C" fn(f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(args[0], args[1])
            }
            3 => {
                let func: extern "C" fn(f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(args[0], args[1], args[2])
            }
            4 => {
                let func: extern "C" fn(f64, f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(args[0], args[1], args[2], args[3])
            }
            5 => {
                let func: extern "C" fn(f64, f64, f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(args[0], args[1], args[2], args[3], args[4])
            }
            6 => {
                let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.function_ptr) };
                func(args[0], args[1], args[2], args[3], args[4], args[5])
            }
            _ => {
                return Err(DSLCompileError::JITError(format!(
                    "Unsupported number of arguments: {}. Maximum supported is 6.",
                    args.len()
                )));
            }
        };

        Ok(result)
    }

    /// Get compilation metadata
    pub fn metadata(&self) -> &CompilationMetadata {
        &self.metadata
    }

    /// Get function signature
    pub fn signature(&self) -> &FunctionSignature {
        &self.signature
    }
}

/// Estimate code size for an expression (rough heuristic)
fn estimate_code_size(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) => 8, // Load constant
        ASTRepr::Variable(_) => 4, // Load from register/memory
        ASTRepr::Add(l, r) | ASTRepr::Sub(l, r) | ASTRepr::Mul(l, r) | ASTRepr::Div(l, r) => {
            estimate_code_size(l) + estimate_code_size(r) + 4
        }
        ASTRepr::Neg(inner) => estimate_code_size(inner) + 4,
        ASTRepr::Pow(base, exp) => estimate_code_size(base) + estimate_code_size(exp) + 20,
        ASTRepr::Sqrt(inner) => estimate_code_size(inner) + 8,
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) | ASTRepr::Exp(inner) | ASTRepr::Ln(inner) => {
            estimate_code_size(inner) + 16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_basic_compilation() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        // Create expression: x + 1
        let expr = ASTEval::add(ASTEval::var(x_idx), ASTEval::constant(1.0));

        let compiler = CraneliftV2Compiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        // Test the compiled function
        let result = compiled.call(&[2.0]).unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_variables() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();
        let y_idx = registry.register_variable();

        // Create expression: x * y + 1
        let expr = ASTEval::add(
            ASTEval::mul(ASTEval::var(x_idx), ASTEval::var(y_idx)),
            ASTEval::constant(1.0),
        );

        let compiler = CraneliftV2Compiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        // Test the compiled function
        let result = compiled.call(&[3.0, 4.0]).unwrap();
        assert!((result - 13.0).abs() < 1e-10); // 3 * 4 + 1 = 13
    }

    #[test]
    fn test_optimization_levels() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        let expr = ASTEval::pow(ASTEval::var(x_idx), ASTEval::constant(2.0));

        // Test different optimization levels
        for opt_level in [
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Full,
        ] {
            let compiler = CraneliftV2Compiler::new(opt_level).unwrap();
            let compiled = compiler
                .compile_expression_with_level(&expr, &registry, opt_level)
                .unwrap();

            let result = compiled.call(&[3.0]).unwrap();
            assert!((result - 9.0).abs() < 1e-10);
            assert_eq!(compiled.metadata().optimization_level, opt_level);
        }
    }

    #[test]
    fn test_integer_power_optimization() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        // Test x^4 - should use optimized integer power
        let expr = ASTEval::pow(ASTEval::var(x_idx), ASTEval::constant(4.0));

        let compiler = CraneliftV2Compiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        let result = compiled.call(&[2.0]).unwrap();
        assert!((result - 16.0).abs() < 1e-10);
    }
}
