//! JIT Compilation Interface
//!
//! This module provides a simplified interface for JIT compilation using the modern
//! Cranelift backend. It wraps the new CraneliftCompiler for backward compatibility.

use crate::backends::cranelift::{CraneliftCompiler, CompiledFunction, OptimizationLevel};
use crate::error::{DSLCompileError, Result};
use crate::final_tagless::{ASTRepr, VariableRegistry};

/// Legacy JIT compiler interface - now wraps the modern Cranelift backend
pub struct JITCompiler {
    optimization_level: OptimizationLevel,
}

impl JITCompiler {
    /// Create a new JIT compiler with default optimization
    pub fn new() -> Result<Self> {
        Ok(Self {
            optimization_level: OptimizationLevel::Basic,
        })
    }

    /// Create a new JIT compiler with specified optimization level
    pub fn with_optimization(opt_level: OptimizationLevel) -> Result<Self> {
        Ok(Self {
            optimization_level: opt_level,
        })
    }

    /// Compile an expression to native code
    pub fn compile_expression(
        &self,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
    ) -> Result<CompiledFunction> {
        let compiler = CraneliftCompiler::new(self.optimization_level)?;
        compiler.compile_expression_with_level(expr, registry, self.optimization_level)
    }

    /// Compile and immediately call a function with given arguments
    pub fn compile_and_call(
        &self,
        expr: &ASTRepr<f64>,
        registry: &VariableRegistry,
        args: &[f64],
    ) -> Result<f64> {
        let compiled = self.compile_expression(expr, registry)?;
        compiled.call(args)
    }
}

impl Default for JITCompiler {
    fn default() -> Self {
        Self::new().expect("Failed to create default JIT compiler")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::ASTEval;

    #[test]
    fn test_jit_basic_compilation() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        let expr = ASTEval::add(ASTEval::var(x_idx), ASTEval::constant(1.0));

        let jit = JITCompiler::new().unwrap();
        let result = jit.compile_and_call(&expr, &registry, &[2.0]).unwrap();

        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jit_optimization_levels() {
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable();

        let expr = ASTEval::mul(ASTEval::var(x_idx), ASTEval::var(x_idx));

        for opt_level in [
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Full,
        ] {
            let jit = JITCompiler::with_optimization(opt_level).unwrap();
            let result = jit.compile_and_call(&expr, &registry, &[3.0]).unwrap();
            assert!((result - 9.0).abs() < 1e-10);
        }
    }
}
