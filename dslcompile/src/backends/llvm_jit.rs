//! LLVM JIT Compilation Backend
//!
//! This module provides Just-In-Time (JIT) compilation using LLVM through the inkwell crate.
//! It enables direct compilation of mathematical expressions to native machine code without
//! the overhead of external rustc compilation or FFI calls.
//!
//! # Features
//!
//! - **Direct JIT Compilation**: AST → LLVM IR → Native Machine Code
//! - **Zero FFI Overhead**: Functions compiled directly to callable native code
//! - **LLVM Optimizations**: Full LLVM optimization pipeline for maximum performance
//! - **Memory Efficient**: No temporary files, everything compiled in-memory
//! - **Cross-Platform**: LLVM handles platform-specific code generation
//!
//! # Performance Benefits
//!
//! This approach eliminates the multi-stage compilation overhead while achieving
//! performance identical to statically compiled code. It bridges the gap between
//! DynamicContext's runtime flexibility and StaticContext's compile-time performance.
//!
//! # Requirements
//!
//! This backend requires LLVM 18 to be installed on your system. The inkwell crate
//! (as of version 0.6.0) does not yet support LLVM 19+. Set the `LLVM_SYS_181_PREFIX`
//! environment variable to point to your LLVM 18 installation.

#[cfg(feature = "llvm_jit")]
use crate::{
    ast::{ASTRepr, Scalar, VariableRegistry, ast_utils::collect_variable_indices},
    error::{DSLCompileError, Result},
};

#[cfg(feature = "llvm_jit")]
use inkwell::{
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    builder::Builder,
    values::{FloatValue, FunctionValue},
    OptimizationLevel,
};

#[cfg(feature = "llvm_jit")]
use num_traits::Float;
use std::collections::HashMap;

/// LLVM JIT compiler for mathematical expressions
#[cfg(feature = "llvm_jit")]
pub struct LLVMJITCompiler<'ctx> {
    /// LLVM context for creating types and values
    context: &'ctx Context,
    /// Cache of compiled functions to avoid recompilation
    function_cache: HashMap<String, ExecutionEngine<'ctx>>,
    /// Counter for generating unique function names
    function_counter: usize,
}

#[cfg(feature = "llvm_jit")]
impl<'ctx> LLVMJITCompiler<'ctx> {
    /// Create a new LLVM JIT compiler
    pub fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            function_cache: HashMap::new(),
            function_counter: 0,
        }
    }
    
    /// Helper function to create JIT execution engine with descriptive error messages
    fn create_jit_engine_with_error_context(
        &self,
        module: &Module<'ctx>,
        opt_level: OptimizationLevel,
    ) -> Result<ExecutionEngine<'ctx>> {
        module
            .create_jit_execution_engine(opt_level)
            .map_err(|e| {
                DSLCompileError::CompilationError(format!(
                    "Failed to create LLVM JIT execution engine: {}\n\
                    \n\
                    This error usually indicates one of the following issues:\n\
                    1. LLVM 18 is not properly installed or not found\n\
                    2. LLVM_SYS_181_PREFIX environment variable is not set correctly\n\
                    3. Incompatible LLVM version (currently requires LLVM 18)\n\
                    4. Missing LLVM development libraries\n\
                    \n\
                    To resolve:\n\
                    - Ensure LLVM 18 is installed on your system\n\
                    - Set LLVM_SYS_181_PREFIX to your LLVM 18 installation path\n\
                    - On Ubuntu: sudo apt install llvm-18-dev\n\
                    - On macOS: brew install llvm@18 && export LLVM_SYS_181_PREFIX=$(brew --prefix llvm@18)\n\
                    - On Arch: sudo pacman -S llvm llvm-libs\n\
                    \n\
                    Original error: {}", e, e
                ))
            })
    }

    /// Compile an AST expression to a JIT-compiled function
    ///
    /// This method converts the AST to LLVM IR, applies optimizations,
    /// and returns a callable function pointer with zero overhead.
    ///
    /// For single-variable expressions, use `compile_single_var`.
    /// For multi-variable expressions, use `compile_multi_var`.
    ///
    /// # Optimization Levels
    /// 
    /// Different optimization levels can be used:
    /// - `None`: No optimization (fastest compilation)
    /// - `Less`: Basic optimization
    /// - `Default`: Standard optimization
    /// - `Aggressive`: Maximum optimization (default)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "llvm_jit")]
    /// # {
    /// use dslcompile::prelude::*;
    /// use dslcompile::backends::LLVMJITCompiler;
    /// use inkwell::context::Context;
    ///
    /// let context = Context::create();
    /// let mut compiler = LLVMJITCompiler::new(&context);
    ///
    /// // Create expression: x² + 2x + 1
    /// let expr: ASTRepr<f64> = ASTRepr::add_from_array([
    ///     ASTRepr::mul_from_array([ASTRepr::<f64>::Variable(0), ASTRepr::<f64>::Variable(0)]),
    ///     ASTRepr::mul_from_array([ASTRepr::<f64>::Constant(2.0), ASTRepr::<f64>::Variable(0)]),
    ///     ASTRepr::<f64>::Constant(1.0)
    /// ]);
    ///
    /// let compiled_fn = compiler.compile_single_var(&expr).unwrap();
    /// let result = unsafe { compiled_fn.call(3.0) }; // (3² + 2*3 + 1) = 16
    /// # }
    /// ```
    pub fn compile_expression<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(f64) -> f64>> {
        self.compile_single_var(expr)
    }

    /// Compile a single-variable expression to a JIT-compiled function
    ///
    /// This method is optimized for expressions with a single variable (index 0).
    /// The generated function has signature `fn(f64) -> f64`.
    pub fn compile_single_var<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(f64) -> f64>> {
        self.compile_single_var_with_opt(expr, OptimizationLevel::Aggressive)
    }

    /// Compile a multi-variable expression to a JIT-compiled function
    ///
    /// This method handles expressions with multiple variables.
    /// The generated function has signature `fn(*const f64) -> f64` where
    /// the parameter points to an array of variable values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "llvm_jit")]
    /// # {
    /// use dslcompile::prelude::*;
    /// use dslcompile::backends::LLVMJITCompiler;
    /// use inkwell::context::Context;
    ///
    /// let context = Context::create();
    /// let mut compiler = LLVMJITCompiler::new(&context);
    ///
    /// // Create expression: x * y + z (variables 0, 1, 2)
    /// let expr: ASTRepr<f64> = ASTRepr::add_from_array([
    ///     ASTRepr::mul_from_array([ASTRepr::<f64>::Variable(0), ASTRepr::<f64>::Variable(1)]),
    ///     ASTRepr::<f64>::Variable(2)
    /// ]);
    ///
    /// let compiled_fn = compiler.compile_multi_var(&expr).unwrap();
    /// let vars = [2.0, 3.0, 1.0]; // x=2, y=3, z=1
    /// let result = unsafe { compiled_fn.call(vars.as_ptr()) }; // 2*3 + 1 = 7
    /// # }
    /// ```
    pub fn compile_multi_var<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*const f64) -> f64>> {
        self.compile_multi_var_with_opt(expr, OptimizationLevel::Aggressive)
    }

    /// Compile single-variable expression with specific optimization level
    pub fn compile_single_var_with_opt<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
        opt_level: OptimizationLevel,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(f64) -> f64>> {
        // Generate unique function name
        let function_name = format!("jit_single_{}", self.function_counter);
        self.function_counter += 1;

        // Create module for this expression
        let module = self.context.create_module(&function_name);

        // Generate function signature for single variable: fn(f64) -> f64
        let f64_type = self.context.f64_type();
        let fn_type = f64_type.fn_type(&[f64_type.into()], false);
        let function = module.add_function(&function_name, fn_type, None);

        // Generate function body for single variable
        self.generate_single_var_function_body(function, expr, &module)?;

        // Print LLVM IR for analysis (debug mode)
        #[cfg(debug_assertions)]
        {
            println!("Generated LLVM IR (single-var):");
            println!("{}", module.print_to_string().to_string());
        }

        // Create execution engine with specified optimization level
        let execution_engine = self.create_jit_engine_with_error_context(&module, opt_level)?;

        // Get the compiled function
        let jit_function = unsafe {
            execution_engine
                .get_function(&function_name)
                .map_err(|e| DSLCompileError::CompilationError(format!("Failed to get JIT function: {}", e)))?
        };

        // Store execution engine to keep it alive
        self.function_cache.insert(function_name, execution_engine);

        Ok(jit_function)
    }

    /// Compile multi-variable expression with specific optimization level
    pub fn compile_multi_var_with_opt<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
        opt_level: OptimizationLevel,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(*const f64) -> f64>> {
        // Generate unique function name
        let function_name = format!("jit_multi_{}", self.function_counter);
        self.function_counter += 1;

        // Create module for this expression
        let module = self.context.create_module(&function_name);

        // Analyze expression variables
        let variables = collect_variable_indices(expr);

        // Generate function signature for multi variables: fn(*const f64) -> f64
        let f64_type = self.context.f64_type();
        let f64_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let fn_type = f64_type.fn_type(&[f64_ptr_type.into()], false);
        let function = module.add_function(&function_name, fn_type, None);

        // Generate function body for multiple variables
        self.generate_multi_var_function_body(function, expr, &variables, &module)?;

        // Print LLVM IR for analysis (debug mode)
        #[cfg(debug_assertions)]
        {
            println!("Generated LLVM IR (multi-var):");
            println!("{}", module.print_to_string().to_string());
        }

        // Create execution engine with specified optimization level
        let execution_engine = self.create_jit_engine_with_error_context(&module, opt_level)?;

        // Get the compiled function
        let jit_function = unsafe {
            execution_engine
                .get_function(&function_name)
                .map_err(|e| DSLCompileError::CompilationError(format!("Failed to get JIT function: {}", e)))?
        };

        // Store execution engine to keep it alive
        self.function_cache.insert(function_name, execution_engine);

        Ok(jit_function)
    }

    /// Compile with specific optimization level for performance testing (backward compatibility)
    pub fn compile_expression_with_opt<T: Scalar + Float + Copy + 'static>(
        &mut self,
        expr: &ASTRepr<T>,
        opt_level: OptimizationLevel,
    ) -> Result<JitFunction<'ctx, unsafe extern "C" fn(f64) -> f64>> {
        // Generate unique function name
        let function_name = format!("jit_expr_{}", self.function_counter);
        self.function_counter += 1;

        // Create module for this expression
        let module = self.context.create_module(&function_name);

        // Analyze expression variables
        let variables = collect_variable_indices(expr);
        let max_var_index = variables.iter().max().copied().unwrap_or(0);
        let param_count = max_var_index + 1;

        // Generate function signature based on variable count
        let f64_type = self.context.f64_type();
        let param_types: Vec<inkwell::types::BasicMetadataTypeEnum> = (0..param_count)
            .map(|_| f64_type.into())
            .collect();
        
        let fn_type = f64_type.fn_type(&param_types, false);
        let function = module.add_function(&function_name, fn_type, None);

        // Generate function body
        self.generate_function_body(function, expr, &variables, &module)?;

        // Print LLVM IR for analysis (debug mode)
        #[cfg(debug_assertions)]
        {
            println!("Generated LLVM IR:");
            println!("{}", module.print_to_string().to_string());
        }

        // Create execution engine with specified optimization level
        let execution_engine = self.create_jit_engine_with_error_context(&module, opt_level)?;

        // Get the compiled function
        let jit_function = unsafe {
            execution_engine
                .get_function(&function_name)
                .map_err(|e| DSLCompileError::CompilationError(format!("Failed to get JIT function: {}", e)))?
        };

        // Store execution engine to keep it alive
        self.function_cache.insert(function_name, execution_engine);

        Ok(jit_function)
    }

    /// Generate LLVM IR for single-variable function body
    fn generate_single_var_function_body<T: Scalar + Float + Copy + 'static>(
        &self,
        function: FunctionValue<'ctx>,
        expr: &ASTRepr<T>,
        module: &Module<'ctx>,
    ) -> Result<()> {
        let builder = self.context.create_builder();
        let entry_block = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry_block);

        // Create variable registry for parameter mapping
        let mut registry = VariableRegistry::new();
        registry.register_typed_variable::<f64>(); // Single variable (index 0)

        // Generate expression IR for single variable
        let result_value = self.generate_single_var_expression_ir(&builder, function, expr, &registry, &module)?;

        // Return the result
        builder.build_return(Some(&result_value)).unwrap();

        Ok(())
    }

    /// Generate LLVM IR for multi-variable function body
    fn generate_multi_var_function_body<T: Scalar + Float + Copy + 'static>(
        &self,
        function: FunctionValue<'ctx>,
        expr: &ASTRepr<T>,
        variables: &std::collections::BTreeSet<usize>,
        module: &Module<'ctx>,
    ) -> Result<()> {
        let builder = self.context.create_builder();
        let entry_block = self.context.append_basic_block(function, "entry");
        builder.position_at_end(entry_block);

        // Create variable registry for parameter mapping
        let mut registry = VariableRegistry::new();
        for _ in 0..=variables.iter().max().copied().unwrap_or(0) {
            registry.register_typed_variable::<f64>();
        }

        // Generate expression IR for multiple variables
        let result_value = self.generate_multi_var_expression_ir(&builder, function, expr, &registry, &module)?;

        // Return the result
        builder.build_return(Some(&result_value)).unwrap();

        Ok(())
    }

    /// Generate LLVM IR for the function body (backward compatibility)
    fn generate_function_body<T: Scalar + Float + Copy + 'static>(
        &self,
        function: FunctionValue<'ctx>,
        expr: &ASTRepr<T>,
        variables: &std::collections::BTreeSet<usize>,
        module: &Module<'ctx>,
    ) -> Result<()> {
        // Use single-variable approach for backward compatibility
        self.generate_single_var_function_body(function, expr, module)
    }

    /// Generate LLVM IR for single-variable mathematical expression
    fn generate_single_var_expression_ir<T: Scalar + Float + Copy + 'static>(
        &self,
        builder: &Builder<'ctx>,
        function: FunctionValue<'ctx>,
        expr: &ASTRepr<T>,
        registry: &VariableRegistry,
        module: &Module<'ctx>,
    ) -> Result<FloatValue<'ctx>> {
        match expr {
            ASTRepr::Constant(value) => {
                let float_val = format!("{}", value).parse::<f64>()
                    .map_err(|_| DSLCompileError::InvalidExpression("Invalid constant value".to_string()))?;
                Ok(self.context.f64_type().const_float(float_val))
            }

            ASTRepr::Variable(index) => {
                if *index == 0 {
                    // Single variable - get first parameter directly
                    let param = function.get_nth_param(0)
                        .ok_or_else(|| DSLCompileError::InvalidExpression("Single variable function requires exactly one parameter".to_string()))?;
                    Ok(param.into_float_value())
                } else {
                    return Err(DSLCompileError::InvalidExpression(format!(
                        "Variable index {} not supported in single-variable function", index
                    )));
                }
            }

            ASTRepr::Add(terms) => {
                let mut result = None;
                for term in terms.elements() {
                    let term_value = self.generate_single_var_expression_ir(builder, function, term, registry, module)?;
                    result = Some(match result {
                        None => term_value,
                        Some(acc) => builder.build_float_add(acc, term_value, "add").unwrap(),
                    });
                }
                result.ok_or_else(|| DSLCompileError::InvalidExpression("Empty addition".to_string()))
            }

            ASTRepr::Mul(factors) => {
                let mut result = None;
                for factor in factors.elements() {
                    let factor_value = self.generate_single_var_expression_ir(builder, function, factor, registry, module)?;
                    result = Some(match result {
                        None => factor_value,
                        Some(acc) => builder.build_float_mul(acc, factor_value, "mul").unwrap(),
                    });
                }
                result.ok_or_else(|| DSLCompileError::InvalidExpression("Empty multiplication".to_string()))
            }

            ASTRepr::Sub(left, right) => {
                let left_value = self.generate_single_var_expression_ir(builder, function, left, registry, module)?;
                let right_value = self.generate_single_var_expression_ir(builder, function, right, registry, module)?;
                Ok(builder.build_float_sub(left_value, right_value, "sub").unwrap())
            }

            ASTRepr::Div(left, right) => {
                let left_value = self.generate_single_var_expression_ir(builder, function, left, registry, module)?;
                let right_value = self.generate_single_var_expression_ir(builder, function, right, registry, module)?;
                Ok(builder.build_float_div(left_value, right_value, "div").unwrap())
            }

            ASTRepr::Pow(base, exp) => {
                let base_value = self.generate_single_var_expression_ir(builder, function, base, registry, module)?;
                let exp_value = self.generate_single_var_expression_ir(builder, function, exp, registry, module)?;
                
                let pow_intrinsic = self.get_or_declare_pow_intrinsic(module);
                let args = [base_value.into(), exp_value.into()];
                let call_result = builder.build_call(pow_intrinsic, &args, "pow").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Sin(inner) => {
                let inner_value = self.generate_single_var_expression_ir(builder, function, inner, registry, module)?;
                let sin_intrinsic = self.get_or_declare_sin_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(sin_intrinsic, &args, "sin").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Cos(inner) => {
                let inner_value = self.generate_single_var_expression_ir(builder, function, inner, registry, module)?;
                let cos_intrinsic = self.get_or_declare_cos_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(cos_intrinsic, &args, "cos").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Ln(inner) => {
                let inner_value = self.generate_single_var_expression_ir(builder, function, inner, registry, module)?;
                let log_intrinsic = self.get_or_declare_log_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(log_intrinsic, &args, "ln").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Exp(inner) => {
                let inner_value = self.generate_single_var_expression_ir(builder, function, inner, registry, module)?;
                let exp_intrinsic = self.get_or_declare_exp_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(exp_intrinsic, &args, "exp").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Sqrt(inner) => {
                let inner_value = self.generate_single_var_expression_ir(builder, function, inner, registry, module)?;
                let sqrt_intrinsic = self.get_or_declare_sqrt_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(sqrt_intrinsic, &args, "sqrt").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Neg(inner) => {
                let inner_value = self.generate_single_var_expression_ir(builder, function, inner, registry, module)?;
                let zero = self.context.f64_type().const_float(0.0);
                Ok(builder.build_float_sub(zero, inner_value, "neg").unwrap())
            }

            _ => Err(DSLCompileError::InvalidExpression(format!(
                "Expression type not yet supported in LLVM JIT: {:?}",
                std::any::type_name::<ASTRepr<T>>()
            ))),
        }
    }

    /// Generate LLVM IR for multi-variable mathematical expression
    fn generate_multi_var_expression_ir<T: Scalar + Float + Copy + 'static>(
        &self,
        builder: &Builder<'ctx>,
        function: FunctionValue<'ctx>,
        expr: &ASTRepr<T>,
        registry: &VariableRegistry,
        module: &Module<'ctx>,
    ) -> Result<FloatValue<'ctx>> {
        match expr {
            ASTRepr::Constant(value) => {
                let float_val = format!("{}", value).parse::<f64>()
                    .map_err(|_| DSLCompileError::InvalidExpression("Invalid constant value".to_string()))?;
                Ok(self.context.f64_type().const_float(float_val))
            }

            ASTRepr::Variable(index) => {
                // Multi-variable - load from array parameter
                let vars_array_ptr = function.get_nth_param(0)
                    .ok_or_else(|| DSLCompileError::InvalidExpression("Multi-variable function requires array parameter".to_string()))?;
                
                // Create GEP to access vars_array[index]
                let index_value = self.context.i32_type().const_int(*index as u64, false);
                let var_ptr = unsafe {
                    builder.build_gep(
                        self.context.f64_type(),
                        vars_array_ptr.into_pointer_value(),
                        &[index_value],
                        "var_ptr"
                    ).unwrap()
                };
                
                // Load the variable value
                let var_value = builder.build_load(self.context.f64_type(), var_ptr, "var_value").unwrap();
                Ok(var_value.into_float_value())
            }

            ASTRepr::Add(terms) => {
                let mut result = None;
                for term in terms.elements() {
                    let term_value = self.generate_multi_var_expression_ir(builder, function, term, registry, module)?;
                    result = Some(match result {
                        None => term_value,
                        Some(acc) => builder.build_float_add(acc, term_value, "add").unwrap(),
                    });
                }
                result.ok_or_else(|| DSLCompileError::InvalidExpression("Empty addition".to_string()))
            }

            ASTRepr::Mul(factors) => {
                let mut result = None;
                for factor in factors.elements() {
                    let factor_value = self.generate_multi_var_expression_ir(builder, function, factor, registry, module)?;
                    result = Some(match result {
                        None => factor_value,
                        Some(acc) => builder.build_float_mul(acc, factor_value, "mul").unwrap(),
                    });
                }
                result.ok_or_else(|| DSLCompileError::InvalidExpression("Empty multiplication".to_string()))
            }

            ASTRepr::Sub(left, right) => {
                let left_value = self.generate_multi_var_expression_ir(builder, function, left, registry, module)?;
                let right_value = self.generate_multi_var_expression_ir(builder, function, right, registry, module)?;
                Ok(builder.build_float_sub(left_value, right_value, "sub").unwrap())
            }

            ASTRepr::Div(left, right) => {
                let left_value = self.generate_multi_var_expression_ir(builder, function, left, registry, module)?;
                let right_value = self.generate_multi_var_expression_ir(builder, function, right, registry, module)?;
                Ok(builder.build_float_div(left_value, right_value, "div").unwrap())
            }

            ASTRepr::Pow(base, exp) => {
                let base_value = self.generate_multi_var_expression_ir(builder, function, base, registry, module)?;
                let exp_value = self.generate_multi_var_expression_ir(builder, function, exp, registry, module)?;
                
                let pow_intrinsic = self.get_or_declare_pow_intrinsic(module);
                let args = [base_value.into(), exp_value.into()];
                let call_result = builder.build_call(pow_intrinsic, &args, "pow").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Sin(inner) => {
                let inner_value = self.generate_multi_var_expression_ir(builder, function, inner, registry, module)?;
                let sin_intrinsic = self.get_or_declare_sin_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(sin_intrinsic, &args, "sin").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Cos(inner) => {
                let inner_value = self.generate_multi_var_expression_ir(builder, function, inner, registry, module)?;
                let cos_intrinsic = self.get_or_declare_cos_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(cos_intrinsic, &args, "cos").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Ln(inner) => {
                let inner_value = self.generate_multi_var_expression_ir(builder, function, inner, registry, module)?;
                let log_intrinsic = self.get_or_declare_log_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(log_intrinsic, &args, "ln").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Exp(inner) => {
                let inner_value = self.generate_multi_var_expression_ir(builder, function, inner, registry, module)?;
                let exp_intrinsic = self.get_or_declare_exp_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(exp_intrinsic, &args, "exp").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Sqrt(inner) => {
                let inner_value = self.generate_multi_var_expression_ir(builder, function, inner, registry, module)?;
                let sqrt_intrinsic = self.get_or_declare_sqrt_intrinsic(module);
                let args = [inner_value.into()];
                let call_result = builder.build_call(sqrt_intrinsic, &args, "sqrt").unwrap();
                Ok(call_result.try_as_basic_value().left().unwrap().into_float_value())
            }

            ASTRepr::Neg(inner) => {
                let inner_value = self.generate_multi_var_expression_ir(builder, function, inner, registry, module)?;
                let zero = self.context.f64_type().const_float(0.0);
                Ok(builder.build_float_sub(zero, inner_value, "neg").unwrap())
            }

            _ => Err(DSLCompileError::InvalidExpression(format!(
                "Expression type not yet supported in LLVM JIT: {:?}",
                std::any::type_name::<ASTRepr<T>>()
            ))),
        }
    }

    /// Generate LLVM IR for a mathematical expression (backward compatibility)
    fn generate_expression_ir<T: Scalar + Float + Copy + 'static>(
        &self,
        builder: &Builder<'ctx>,
        function: FunctionValue<'ctx>,
        expr: &ASTRepr<T>,
        registry: &VariableRegistry,
        module: &Module<'ctx>,
    ) -> Result<FloatValue<'ctx>> {
        // Use single-variable approach for backward compatibility
        self.generate_single_var_expression_ir(builder, function, expr, registry, module)
    }

    /// Get or declare LLVM math intrinsics
    fn get_or_declare_pow_intrinsic(&self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let intrinsic_name = "llvm.pow.f64";
        match module.get_function(intrinsic_name) {
            Some(function) => function,
            None => {
                let f64_type = self.context.f64_type();
                let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
                module.add_function(intrinsic_name, fn_type, None)
            }
        }
    }

    fn get_or_declare_sin_intrinsic(&self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let intrinsic_name = "llvm.sin.f64";
        match module.get_function(intrinsic_name) {
            Some(function) => function,
            None => {
                let f64_type = self.context.f64_type();
                let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                module.add_function(intrinsic_name, fn_type, None)
            }
        }
    }

    fn get_or_declare_cos_intrinsic(&self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let intrinsic_name = "llvm.cos.f64";
        match module.get_function(intrinsic_name) {
            Some(function) => function,
            None => {
                let f64_type = self.context.f64_type();
                let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                module.add_function(intrinsic_name, fn_type, None)
            }
        }
    }

    fn get_or_declare_log_intrinsic(&self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let intrinsic_name = "llvm.log.f64";
        match module.get_function(intrinsic_name) {
            Some(function) => function,
            None => {
                let f64_type = self.context.f64_type();
                let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                module.add_function(intrinsic_name, fn_type, None)
            }
        }
    }

    fn get_or_declare_exp_intrinsic(&self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let intrinsic_name = "llvm.exp.f64";
        match module.get_function(intrinsic_name) {
            Some(function) => function,
            None => {
                let f64_type = self.context.f64_type();
                let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                module.add_function(intrinsic_name, fn_type, None)
            }
        }
    }

    fn get_or_declare_sqrt_intrinsic(&self, module: &Module<'ctx>) -> FunctionValue<'ctx> {
        let intrinsic_name = "llvm.sqrt.f64";
        match module.get_function(intrinsic_name) {
            Some(function) => function,
            None => {
                let f64_type = self.context.f64_type();
                let fn_type = f64_type.fn_type(&[f64_type.into()], false);
                module.add_function(intrinsic_name, fn_type, None)
            }
        }
    }

    /// Clear the function cache
    pub fn clear_cache(&mut self) {
        self.function_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.function_cache.len(), self.function_cache.capacity())
    }
}

#[cfg(not(feature = "llvm_jit"))]
pub struct LLVMJITCompiler;

#[cfg(not(feature = "llvm_jit"))]
impl LLVMJITCompiler {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "llvm_jit")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;
    use inkwell::context::Context;

    #[test]
    fn test_basic_constant() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        let expr = ASTRepr::Constant(42.0);
        let compiled_fn = compiler.compile_expression(&expr).unwrap();

        // Note: This test requires no parameters since it's just a constant
        // We'll need to adjust this once we handle variable parameter counts properly
    }

    #[test]
    fn test_simple_addition() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        // x + 5
        let expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(5.0),
        ]);

        let compiled_fn = compiler.compile_expression(&expr).unwrap();
        let result = unsafe { compiled_fn.call(3.0) };
        assert!((result - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quadratic_expression() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        // x² + 2x + 1 = (x + 1)²
        let expr = ASTRepr::add_from_array([
            ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Variable(0)]), // x²
            ASTRepr::mul_from_array([ASTRepr::Constant(2.0), ASTRepr::Variable(0)]), // 2x
            ASTRepr::Constant(1.0), // 1
        ]);

        let compiled_fn = compiler.compile_expression(&expr).unwrap();
        
        // Test with x = 3: 3² + 2*3 + 1 = 9 + 6 + 1 = 16
        let result = unsafe { compiled_fn.call(3.0) };
        assert!((result - 16.0).abs() < f64::EPSILON);
        
        // Test with x = 0: 0² + 2*0 + 1 = 1
        let result = unsafe { compiled_fn.call(0.0) };
        assert!((result - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trigonometric_functions() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        // sin(x)
        let sin_expr: ASTRepr<f64> = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let sin_fn = compiler.compile_expression(&sin_expr).unwrap();
        
        let result = unsafe { sin_fn.call(0.0) };
        assert!(result.abs() < f64::EPSILON); // sin(0) = 0

        // cos(x)
        let cos_expr: ASTRepr<f64> = ASTRepr::Cos(Box::new(ASTRepr::Variable(0)));
        let cos_fn = compiler.compile_expression(&cos_expr).unwrap();
        
        let result = unsafe { cos_fn.call(0.0) };
        assert!((result - 1.0).abs() < f64::EPSILON); // cos(0) = 1
    }

    #[test]
    fn test_cache_functionality() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        let (initial_size, _) = compiler.cache_stats();
        assert_eq!(initial_size, 0);

        // Compile a function
        let expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0),
            ASTRepr::Constant(1.0),
        ]);
        let _compiled_fn = compiler.compile_expression(&expr).unwrap();

        let (cache_size, _) = compiler.cache_stats();
        assert_eq!(cache_size, 1);

        // Clear cache
        compiler.clear_cache();
        let (final_size, _) = compiler.cache_stats();
        assert_eq!(final_size, 0);
    }

    #[test]
    fn test_multi_variable_expression() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        // x * y + z (variables 0, 1, 2)
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Variable(1)]),
            ASTRepr::Variable(2),
        ]);

        let compiled_fn = compiler.compile_multi_var(&expr).unwrap();
        
        // Test with x=2, y=3, z=1: 2*3 + 1 = 7
        let vars = [2.0, 3.0, 1.0];
        let result = unsafe { compiled_fn.call(vars.as_ptr()) };
        assert!((result - 7.0).abs() < f64::EPSILON);
        
        // Test with x=0, y=5, z=10: 0*5 + 10 = 10
        let vars = [0.0, 5.0, 10.0];
        let result = unsafe { compiled_fn.call(vars.as_ptr()) };
        assert!((result - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_vs_multi_variable_compatibility() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        // Same expression compiled different ways: x² + 2x + 1
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::mul_from_array([ASTRepr::Variable(0), ASTRepr::Variable(0)]),
            ASTRepr::mul_from_array([ASTRepr::Constant(2.0), ASTRepr::Variable(0)]),
            ASTRepr::Constant(1.0),
        ]);

        // Compile as single-variable function
        let single_fn = compiler.compile_single_var(&expr).unwrap();
        
        // Compile as multi-variable function
        let multi_fn = compiler.compile_multi_var(&expr).unwrap();

        let test_x = 4.0;
        let expected = test_x * test_x + 2.0 * test_x + 1.0; // 25

        // Test single-variable function
        let single_result = unsafe { single_fn.call(test_x) };
        assert!((single_result - expected).abs() < f64::EPSILON);

        // Test multi-variable function
        let vars = [test_x];
        let multi_result = unsafe { multi_fn.call(vars.as_ptr()) };
        assert!((multi_result - expected).abs() < f64::EPSILON);

        // Both should produce identical results
        assert!((single_result - multi_result).abs() < f64::EPSILON);
    }

    #[test]
    fn test_complex_multi_variable_expression() {
        let context = Context::create();
        let mut compiler = LLVMJITCompiler::new(&context);

        // Complex expression: sin(x) * cos(y) + ln(z + 1) + sqrt(w)
        let expr: ASTRepr<f64> = ASTRepr::add_from_array([
            ASTRepr::mul_from_array([
                ASTRepr::Sin(Box::new(ASTRepr::Variable(0))),
                ASTRepr::Cos(Box::new(ASTRepr::Variable(1))),
            ]),
            ASTRepr::Ln(Box::new(ASTRepr::add_from_array([
                ASTRepr::Variable(2),
                ASTRepr::Constant(1.0),
            ]))),
            ASTRepr::Sqrt(Box::new(ASTRepr::Variable(3))),
        ]);

        let compiled_fn = compiler.compile_multi_var(&expr).unwrap();
        
        // Test with x=0, y=0, z=0, w=4: sin(0)*cos(0) + ln(1) + sqrt(4) = 0*1 + 0 + 2 = 2
        let vars = [0.0, 0.0, 0.0, 4.0];
        let result = unsafe { compiled_fn.call(vars.as_ptr()) };
        let expected = 0.0 * 1.0 + 0.0 + 2.0; // 2.0
        assert!((result - expected).abs() < f64::EPSILON);
    }
}