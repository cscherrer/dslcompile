//! JIT Compilation Module
//!
//! This module provides just-in-time compilation capabilities for mathematical expressions.
//! It serves as a high-level interface to various compilation backends.

use crate::Result;
use crate::ast::{ExpressionBuilder, VariableRegistry, ASTRepr};
use crate::backends::cranelift::{CraneliftCompiler, OptimizationLevel};

/// JIT compiler that can compile expressions to native code
pub struct JITCompiler {
    cranelift_compiler: CraneliftCompiler,
}

impl JITCompiler {
    /// Create a new JIT compiler with default settings
    pub fn new() -> Result<Self> {
        Ok(Self {
            cranelift_compiler: CraneliftCompiler::new(OptimizationLevel::Full)?,
        })
    }

    /// Compile an expression to a callable function
    pub fn compile(&mut self, expr: &ASTRepr<f64>) -> Result<Box<dyn Fn(&[f64]) -> f64>> {
        let registry = VariableRegistry::for_expression(expr);
        let compiled = self.cranelift_compiler.compile_expression(expr, &registry)?;
        
        Ok(Box::new(move |args: &[f64]| {
            compiled.call(args).unwrap_or(f64::NAN)
        }))
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

    #[test]
    fn test_simple_jit_compilation() {
        let mut compiler = JITCompiler::new().unwrap();
        
        let math = ExpressionBuilder::new();
        let mut registry = VariableRegistry::new();
        let _x_idx = registry.register_variable();
        
        let x = math.var();
        let expr = (&x + 1.0).into_ast();
        
        let func = compiler.compile(&expr).unwrap();
        
        let result = func(&[5.0]);
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_quadratic_jit_compilation() {
        let mut compiler = JITCompiler::new().unwrap();
        
        let math = ExpressionBuilder::new();
        let mut registry = VariableRegistry::new();
        let _x_idx = registry.register_variable();
        
        let x = math.var();
        let expr = (&x * &x).into_ast();
        
        let func = compiler.compile(&expr).unwrap();
        
        let result = func(&[3.0]);
        assert_eq!(result, 9.0);
    }
}
