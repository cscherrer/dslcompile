//! Modern Cranelift JIT Backend
//!
//! This is a modern Cranelift backend that addresses the architectural issues
//! in previous implementations. Key improvements:
//!
//! 1. **Simplified Architecture**: Direct IR generation without complex mappings
//! 2. **Index-Based Variables**: Leverages the new index-only variable system
//! 3. **Modern Cranelift APIs**: Uses latest Cranelift patterns and optimizations
//! 4. **Better Error Handling**: Comprehensive error types and recovery
//! 5. **Optimized Function Signatures**: Automatic signature generation
//! 6. **E-graph Integration**: Proper use of Cranelift's optimization pipeline
//! 7. **Fast Math Functions**: Direct libcall integration for transcendental functions

use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Type, Value, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use std::collections::HashMap;
use std::time::Instant;

use crate::ast::{ASTRepr, VariableRegistry};
use crate::error::{DSLCompileError, Result};

/// Modern JIT compiler using latest Cranelift patterns
pub struct CraneliftCompiler {
    /// JIT module for code generation
    module: JITModule,
    /// Function builder context (reusable)
    builder_context: FunctionBuilderContext,
    /// Compilation settings
    settings: settings::Flags,
    /// Optimization level
    opt_level: OptimizationLevel,
}

/// Compiled function with modern interface
pub struct CompiledFunction {
    /// Function pointer to native code
    func_ptr: *const u8,
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
    /// Compilation time in milliseconds
    pub compilation_time_ms: u64,
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
    /// Expression complexity
    pub expression_complexity: usize,
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

/// External math function IDs
struct ExternalMathFunctions {
    sin_id: FuncId,
    cos_id: FuncId,
    exp_id: FuncId,
    log_id: FuncId,
    pow_id: FuncId,
}

/// Local math function references for use within a function
struct LocalMathFunctions {
    sin_ref: cranelift_codegen::ir::FuncRef,
    cos_ref: cranelift_codegen::ir::FuncRef,
    exp_ref: cranelift_codegen::ir::FuncRef,
    log_ref: cranelift_codegen::ir::FuncRef,
    pow_ref: cranelift_codegen::ir::FuncRef,
}

/// Safe wrapper functions for math operations using Rust's std library
mod math_wrappers {
    /// Safe wrapper for sin function
    pub extern "C" fn sin_wrapper(x: f64) -> f64 {
        x.sin()
    }

    /// Safe wrapper for cos function
    pub extern "C" fn cos_wrapper(x: f64) -> f64 {
        x.cos()
    }

    /// Safe wrapper for exp function
    pub extern "C" fn exp_wrapper(x: f64) -> f64 {
        x.exp()
    }

    /// Safe wrapper for natural logarithm function
    pub extern "C" fn log_wrapper(x: f64) -> f64 {
        x.ln()
    }

    /// Safe wrapper for power function
    pub extern "C" fn pow_wrapper(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
}

impl CraneliftCompiler {
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

        // Create JIT builder and register safe math function wrappers
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register safe wrapper functions using Rust's std library
        builder.symbol("sin", math_wrappers::sin_wrapper as *const u8);
        builder.symbol("cos", math_wrappers::cos_wrapper as *const u8);
        builder.symbol("exp", math_wrappers::exp_wrapper as *const u8);
        builder.symbol("log", math_wrappers::log_wrapper as *const u8);
        builder.symbol("pow", math_wrappers::pow_wrapper as *const u8);

        let module = JITModule::new(builder);

        Ok(Self {
            module,
            builder_context: FunctionBuilderContext::new(),
            settings,
            opt_level,
        })
    }

    /// Create compiler with default optimization level
    pub fn new_default() -> Result<Self> {
        Self::new(OptimizationLevel::default())
    }

    /// Compile an expression to a native function
    pub fn compile_expression(
        &mut self,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
    ) -> Result<CompiledFunction> {
        let start_time = Instant::now();

        // Convert to ANF first for optimization and clean structure
        let anf = crate::symbolic::anf::convert_to_anf(expr)?;

        // Declare external math functions in the module
        let math_functions = self.declare_external_math_functions()?;

        // Create function signature
        let mut sig = self.module.make_signature();
        for _ in 0..registry.len() {
            sig.params.push(AbiParam::new(types::F64));
        }
        sig.returns.push(AbiParam::new(types::F64));

        // Declare the function
        let func_id = self
            .module
            .declare_function("compiled_expr", Linkage::Export, &sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare function: {e}")))?;

        // Build the function
        let mut ctx = self.module.make_context();
        ctx.func.signature = sig.clone();

        self.build_function_body_from_anf(&mut ctx.func, &anf, registry, &math_functions)?;

        // Compile the function
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to define function: {e}")))?;

        // Finalize the function
        self.module.finalize_definitions().map_err(|e| {
            DSLCompileError::JITError(format!("Failed to finalize definitions: {e}"))
        })?;

        // Get the compiled function pointer
        let code_ptr = self.module.get_finalized_function(func_id);

        let compilation_time = start_time.elapsed();

        Ok(CompiledFunction {
            func_ptr: code_ptr,
            signature: FunctionSignature {
                input_count: registry.len(),
                return_type: types::F64,
            },
            metadata: CompilationMetadata {
                compilation_time_ms: compilation_time.as_millis() as u64,
                optimization_level: self.opt_level,
                expression_complexity: expr.count_operations(),
            },
        })
    }

    /// Declare external math functions in the module
    fn declare_external_math_functions(&mut self) -> Result<ExternalMathFunctions> {
        // Create signature for single-argument functions: f64 -> f64
        let mut single_arg_sig = self.module.make_signature();
        single_arg_sig.params.push(AbiParam::new(types::F64));
        single_arg_sig.returns.push(AbiParam::new(types::F64));

        // Create signature for double-argument functions: (f64, f64) -> f64
        let mut double_arg_sig = self.module.make_signature();
        double_arg_sig.params.push(AbiParam::new(types::F64));
        double_arg_sig.params.push(AbiParam::new(types::F64));
        double_arg_sig.returns.push(AbiParam::new(types::F64));

        // Declare external functions
        let sin_id = self
            .module
            .declare_function("sin", Linkage::Import, &single_arg_sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare sin: {e}")))?;

        let cos_id = self
            .module
            .declare_function("cos", Linkage::Import, &single_arg_sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare cos: {e}")))?;

        let exp_id = self
            .module
            .declare_function("exp", Linkage::Import, &single_arg_sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare exp: {e}")))?;

        let log_id = self
            .module
            .declare_function("log", Linkage::Import, &single_arg_sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare log: {e}")))?;

        let pow_id = self
            .module
            .declare_function("pow", Linkage::Import, &double_arg_sig)
            .map_err(|e| DSLCompileError::JITError(format!("Failed to declare pow: {e}")))?;

        Ok(ExternalMathFunctions {
            sin_id,
            cos_id,
            exp_id,
            log_id,
            pow_id,
        })
    }

    /// Build the function body from ANF representation with IR generation
    fn build_function_body_from_anf(
        &mut self,
        func: &mut Function,
        anf: &crate::symbolic::anf::ANFExpr<f64>,
        registry: &VariableRegistry,
        math_functions: &ExternalMathFunctions,
    ) -> Result<()> {
        let mut builder = FunctionBuilder::new(func, &mut self.builder_context);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get function parameters
        let params = builder.block_params(entry_block);

        // Create variable mapping for user variables
        let mut var_values = HashMap::new();
        for i in 0..registry.len() {
            var_values.insert(i, params[i]);
        }

        // Import external math functions into this function
        let local_sin = self
            .module
            .declare_func_in_func(math_functions.sin_id, builder.func);
        let local_cos = self
            .module
            .declare_func_in_func(math_functions.cos_id, builder.func);
        let local_exp = self
            .module
            .declare_func_in_func(math_functions.exp_id, builder.func);
        let local_log = self
            .module
            .declare_func_in_func(math_functions.log_id, builder.func);
        let local_pow = self
            .module
            .declare_func_in_func(math_functions.pow_id, builder.func);

        // Create math function references struct
        let local_math_functions = LocalMathFunctions {
            sin_ref: local_sin,
            cos_ref: local_cos,
            exp_ref: local_exp,
            log_ref: local_log,
            pow_ref: local_pow,
        };

        // Generate IR from ANF
        let result =
            Self::generate_ir_from_anf(&mut builder, anf, &var_values, &local_math_functions)?;

        // Return the result
        builder.ins().return_(&[result]);
        builder.finalize();

        Ok(())
    }

    /// Generate IR from ANF expression
    fn generate_ir_from_anf(
        builder: &mut FunctionBuilder,
        anf: &crate::symbolic::anf::ANFExpr<f64>,
        user_vars: &HashMap<usize, Value>,
        math_functions: &LocalMathFunctions,
    ) -> Result<Value> {
        let mut bound_vars: HashMap<u32, Value> = HashMap::new();

        Self::generate_ir_from_anf_with_bindings(
            builder,
            anf,
            user_vars,
            &mut bound_vars,
            math_functions,
        )
    }

    /// Generate IR from ANF with variable bindings
    fn generate_ir_from_anf_with_bindings(
        builder: &mut FunctionBuilder,
        anf: &crate::symbolic::anf::ANFExpr<f64>,
        user_vars: &HashMap<usize, Value>,
        bound_vars: &mut HashMap<u32, Value>,
        math_functions: &LocalMathFunctions,
    ) -> Result<Value> {
        use crate::symbolic::anf::{ANFExpr, VarRef};

        match anf {
            ANFExpr::Atom(atom) => Ok(Self::generate_ir_from_atom(
                builder, atom, user_vars, bound_vars,
            )),
            ANFExpr::Let(var_ref, computation, body) => {
                // Generate IR for the computation
                let comp_result = Self::generate_ir_from_computation(
                    builder,
                    computation,
                    user_vars,
                    bound_vars,
                    math_functions,
                )?;

                // Bind the result to the variable
                if let VarRef::Bound(id) = var_ref {
                    bound_vars.insert(*id, comp_result);
                }

                // Generate IR for the body
                Self::generate_ir_from_anf_with_bindings(
                    builder,
                    body,
                    user_vars,
                    bound_vars,
                    math_functions,
                )
            }


        }
    }

    /// Generate IR from ANF atom
    fn generate_ir_from_atom(
        builder: &mut FunctionBuilder,
        atom: &crate::symbolic::anf::ANFAtom<f64>,
        user_vars: &HashMap<usize, Value>,
        bound_vars: &HashMap<u32, Value>,
    ) -> Value {
        use crate::symbolic::anf::{ANFAtom, VarRef};

        match atom {
            ANFAtom::Constant(value) => builder.ins().f64const(*value),
            ANFAtom::Variable(var_ref) => match var_ref {
                VarRef::User(idx) => *user_vars.get(idx).expect("User variable not found"),
                VarRef::Bound(id) => *bound_vars.get(id).expect("Bound variable not found"),
            },
        }
    }

    /// Generate IR from ANF computation
    fn generate_ir_from_computation(
        builder: &mut FunctionBuilder,
        computation: &crate::symbolic::anf::ANFComputation<f64>,
        user_vars: &HashMap<usize, Value>,
        bound_vars: &HashMap<u32, Value>,
        math_functions: &LocalMathFunctions,
    ) -> Result<Value> {
        use crate::symbolic::anf::ANFComputation;

        match computation {
            ANFComputation::Add(left, right) => {
                let left_val = Self::generate_ir_from_atom(builder, left, user_vars, bound_vars);
                let right_val = Self::generate_ir_from_atom(builder, right, user_vars, bound_vars);
                Ok(builder.ins().fadd(left_val, right_val))
            }
            ANFComputation::Sub(left, right) => {
                let left_val = Self::generate_ir_from_atom(builder, left, user_vars, bound_vars);
                let right_val = Self::generate_ir_from_atom(builder, right, user_vars, bound_vars);
                Ok(builder.ins().fsub(left_val, right_val))
            }
            ANFComputation::Mul(left, right) => {
                let left_val = Self::generate_ir_from_atom(builder, left, user_vars, bound_vars);
                let right_val = Self::generate_ir_from_atom(builder, right, user_vars, bound_vars);
                Ok(builder.ins().fmul(left_val, right_val))
            }
            ANFComputation::Div(left, right) => {
                let left_val = Self::generate_ir_from_atom(builder, left, user_vars, bound_vars);
                let right_val = Self::generate_ir_from_atom(builder, right, user_vars, bound_vars);
                Ok(builder.ins().fdiv(left_val, right_val))
            }
            ANFComputation::Pow(left, right) => {
                let left_val = Self::generate_ir_from_atom(builder, left, user_vars, bound_vars);
                let right_val = Self::generate_ir_from_atom(builder, right, user_vars, bound_vars);
                // Use external pow function (ANF should have already optimized integer powers)
                let call = builder
                    .ins()
                    .call(math_functions.pow_ref, &[left_val, right_val]);
                Ok(builder.inst_results(call)[0])
            }
            ANFComputation::Neg(operand) => {
                let val = Self::generate_ir_from_atom(builder, operand, user_vars, bound_vars);
                Ok(builder.ins().fneg(val))
            }
            ANFComputation::Sin(operand) => {
                let val = Self::generate_ir_from_atom(builder, operand, user_vars, bound_vars);
                let call = builder.ins().call(math_functions.sin_ref, &[val]);
                Ok(builder.inst_results(call)[0])
            }
            ANFComputation::Cos(operand) => {
                let val = Self::generate_ir_from_atom(builder, operand, user_vars, bound_vars);
                let call = builder.ins().call(math_functions.cos_ref, &[val]);
                Ok(builder.inst_results(call)[0])
            }
            ANFComputation::Exp(operand) => {
                let val = Self::generate_ir_from_atom(builder, operand, user_vars, bound_vars);
                let call = builder.ins().call(math_functions.exp_ref, &[val]);
                Ok(builder.inst_results(call)[0])
            }
            ANFComputation::Ln(operand) => {
                let val = Self::generate_ir_from_atom(builder, operand, user_vars, bound_vars);
                let call = builder.ins().call(math_functions.log_ref, &[val]);
                Ok(builder.inst_results(call)[0])
            }
            ANFComputation::Sqrt(operand) => {
                let val = Self::generate_ir_from_atom(builder, operand, user_vars, bound_vars);
                // TODO: Add sqrt to external math functions and use proper declaration
                // For now, use the builtin sqrt instruction
                Ok(builder.ins().sqrt(val))
            }
        }
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
                let func: extern "C" fn() -> f64 = unsafe { std::mem::transmute(self.func_ptr) };
                func()
            }
            1 => {
                let func: extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(self.func_ptr) };
                func(args[0])
            }
            2 => {
                let func: extern "C" fn(f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(args[0], args[1])
            }
            3 => {
                let func: extern "C" fn(f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(args[0], args[1], args[2])
            }
            4 => {
                let func: extern "C" fn(f64, f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(args[0], args[1], args[2], args[3])
            }
            5 => {
                let func: extern "C" fn(f64, f64, f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.func_ptr) };
                func(args[0], args[1], args[2], args[3], args[4])
            }
            6 => {
                let func: extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64 =
                    unsafe { std::mem::transmute(self.func_ptr) };
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
    #[must_use]
    pub fn metadata(&self) -> &CompilationMetadata {
        &self.metadata
    }

    /// Get function signature
    #[must_use]
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
        ASTRepr::Sum { range, body, .. } => {
            // TODO: Implement Sum code size estimation for Cranelift backend
            let body_size = estimate_code_size(body);
            let range_size = match range {
                crate::ast::ast_repr::SumRange::Mathematical { start, end } => {
                    estimate_code_size(start) + estimate_code_size(end)
                }
                crate::ast::ast_repr::SumRange::DataParameter { .. } => 8,
            };
            body_size + range_size + 32 // Rough estimate for iteration overhead
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ExpressionBuilder;

    #[test]
    fn test_basic_compilation() {
        let math = ExpressionBuilder::new();

        // Create expression: x + 1
        let x = math.var();
        let expr = (&x + 1.0).into_ast();

        // Use the registry from the ExpressionBuilder
        let registry = math.registry().borrow().clone();

        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        // Test the compiled function
        let result = compiled.call(&[2.0]).unwrap();
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_variables() {
        let math = ExpressionBuilder::new();

        // Create expression: x * y + 1
        let x = math.var();
        let y = math.var();
        let expr = (&x * &y + 1.0).into_ast();

        // Use the registry from the ExpressionBuilder
        let registry = math.registry().borrow().clone();

        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        // Test the compiled function
        let result = compiled.call(&[3.0, 4.0]).unwrap();
        assert!((result - 13.0).abs() < 1e-10); // 3 * 4 + 1 = 13
    }

    #[test]
    fn test_optimization_levels() {
        let math = ExpressionBuilder::new();

        let x = math.var();
        let expr = x.pow(math.constant(2.0)).into_ast();

        // Use the registry from the ExpressionBuilder
        let registry = math.registry().borrow().clone();

        // Test different optimization levels
        for opt_level in [
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Full,
        ] {
            let mut compiler = CraneliftCompiler::new(opt_level).unwrap();
            let compiled = compiler.compile_expression(&expr, &registry).unwrap();

            let result = compiled.call(&[3.0]).unwrap();
            assert!((result - 9.0).abs() < 1e-10);
            assert_eq!(compiled.metadata().optimization_level, opt_level);
        }
    }

    #[test]
    fn test_binary_exponentiation_optimization() {
        // Test various integer powers that should use binary exponentiation
        let test_cases = vec![
            (2, 4.0),      // 2^2 = 4
            (3, 8.0),      // 2^3 = 8
            (4, 16.0),     // 2^4 = 16
            (5, 32.0),     // 2^5 = 32
            (8, 256.0),    // 2^8 = 256
            (10, 1024.0),  // 2^10 = 1024
            (16, 65536.0), // 2^16 = 65536
        ];

        for (exp, expected) in test_cases {
            // Create a fresh ExpressionBuilder for each test case
            let math = ExpressionBuilder::new();
            let x = math.var();
            let expr = x.pow(math.constant(f64::from(exp))).into_ast();

            // Use the registry from the ExpressionBuilder after creating the expression
            let registry = math.registry().borrow().clone();

            let mut compiler = CraneliftCompiler::new_default().unwrap();
            let compiled = compiler.compile_expression(&expr, &registry).unwrap();

            let result = compiled.call(&[2.0]).unwrap();
            assert!(
                (result - expected).abs() < 1e-10,
                "2^{exp} = {expected} but got {result}"
            );
        }

        // Test negative powers
        let math = ExpressionBuilder::new();
        let x = math.var();
        let expr = x.pow(math.constant(-2.0)).into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        let result = compiled.call(&[2.0]).unwrap();
        assert!((result - 0.25).abs() < 1e-10); // 2^(-2) = 1/4 = 0.25

        // Test fractional powers that should use sqrt optimization
        let math = ExpressionBuilder::new();
        let x = math.var();
        let expr = x.pow(math.constant(0.5)).into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        let result = compiled.call(&[4.0]).unwrap();
        assert!((result - 2.0).abs() < 1e-10); // 4^0.5 = 2

        // Test x^(-0.5) = 1/sqrt(x)
        let math = ExpressionBuilder::new();
        let x = math.var();
        let expr = x.pow(math.constant(-0.5)).into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&expr, &registry).unwrap();

        let result = compiled.call(&[4.0]).unwrap();
        assert!((result - 0.5).abs() < 1e-10); // 4^(-0.5) = 1/2 = 0.5
    }

    #[test]
    fn test_transcendental_functions() {
        // Test sin function
        let math = ExpressionBuilder::new();
        let x = math.var();
        let sin_expr = x.sin().into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&sin_expr, &registry).unwrap();

        let result = compiled.call(&[0.0]).unwrap();
        assert!((result - 0.0).abs() < 1e-10); // sin(0) = 0

        let result = compiled.call(&[std::f64::consts::PI / 2.0]).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // sin(π/2) = 1

        // Test cos function
        let math = ExpressionBuilder::new();
        let x = math.var();
        let cos_expr = x.cos().into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&cos_expr, &registry).unwrap();

        let result = compiled.call(&[0.0]).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // cos(0) = 1

        // Test exp function
        let math = ExpressionBuilder::new();
        let x = math.var();
        let exp_expr = x.exp().into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&exp_expr, &registry).unwrap();

        let result = compiled.call(&[0.0]).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // exp(0) = 1

        let result = compiled.call(&[1.0]).unwrap();
        assert!((result - std::f64::consts::E).abs() < 1e-10); // exp(1) = e

        // Test ln function
        let math = ExpressionBuilder::new();
        let x = math.var();
        let ln_expr = x.ln().into_ast();
        let registry = math.registry().borrow().clone();
        let mut compiler = CraneliftCompiler::new_default().unwrap();
        let compiled = compiler.compile_expression(&ln_expr, &registry).unwrap();

        let result = compiled.call(&[1.0]).unwrap();
        assert!((result - 0.0).abs() < 1e-10); // ln(1) = 0

        let result = compiled.call(&[std::f64::consts::E]).unwrap();
        assert!((result - 1.0).abs() < 1e-10); // ln(e) = 1
    }
}
