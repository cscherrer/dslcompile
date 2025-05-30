//! JAX-inspired Functional Transformations
//!
//! This module provides composable transformations that can be applied to mathematical
//! expressions, similar to JAX's transformation system (jit, grad, vmap, etc.).
//!
//! # Design Philosophy
//!
//! Following JAX's approach, transformations are:
//! - **Composable**: `jit(grad(f))` works seamlessly
//! - **Pure**: No side effects, return new transformed expressions
//! - **Lazy**: Transformations are applied when needed
//! - **Type-safe**: Compile-time guarantees about transformation validity

use crate::error::Result;
use crate::final_tagless::{ASTRepr, VariableRegistry};
use crate::backends::{RustCodeGenerator, RustCompiler, CompiledRustFunction};
use crate::symbolic::symbolic_ad::{SymbolicAD, SymbolicADConfig};
use crate::interval_domain::{IntervalDomain, IntervalDomainAnalyzer};
use std::collections::HashMap;

/// A transformed mathematical expression with metadata
#[derive(Debug, Clone)]
pub struct TransformedExpr<T> {
    /// The transformed expression
    pub expr: ASTRepr<T>,
    /// Metadata about applied transformations
    pub metadata: TransformationMetadata,
    /// Variable registry for the transformed expression
    pub registry: VariableRegistry,
}

/// Metadata tracking applied transformations
#[derive(Debug, Clone, Default)]
pub struct TransformationMetadata {
    /// List of transformations applied in order
    pub transformations: Vec<String>,
    /// Compilation hints for optimization
    pub compilation_hints: CompilationHints,
    /// Domain constraints from analysis
    pub domain_constraints: HashMap<usize, IntervalDomain<f64>>,
}

/// Compilation hints derived from transformations
#[derive(Debug, Clone, Default)]
pub struct CompilationHints {
    /// Whether this expression benefits from JIT compilation
    pub should_jit: bool,
    /// Whether vectorization is beneficial
    pub vectorizable: bool,
    /// Expected call frequency (for adaptive compilation)
    pub call_frequency_hint: CallFrequency,
    /// Memory access patterns
    pub memory_pattern: MemoryPattern,
}

#[derive(Debug, Clone, Default)]
pub enum CallFrequency {
    #[default]
    Unknown,
    Rare,      // < 10 calls
    Moderate,  // 10-1000 calls  
    Frequent,  // > 1000 calls
}

#[derive(Debug, Clone, Default)]
pub enum MemoryPattern {
    #[default]
    Unknown,
    Sequential,  // Good for vectorization
    Random,      // Poor cache locality
    Structured,  // Regular patterns
}

/// JAX-like transformation system
pub struct Transformations;

impl Transformations {
    /// JIT compile an expression (like JAX's `jit`)
    pub fn jit(expr: ASTRepr<f64>) -> JitTransformation {
        JitTransformation::new(expr)
    }

    /// Compute gradient (like JAX's `grad`)
    pub fn grad(expr: ASTRepr<f64>) -> GradTransformation {
        GradTransformation::new(expr)
    }

    /// Vectorize over a dimension (like JAX's `vmap`)
    pub fn vmap(expr: ASTRepr<f64>, axis: usize) -> VmapTransformation {
        VmapTransformation::new(expr, axis)
    }

    /// Apply domain analysis (unique to this library)
    pub fn analyze_domains(expr: ASTRepr<f64>) -> DomainTransformation {
        DomainTransformation::new(expr)
    }

    /// Partial evaluation with static values (like JAX's partial evaluation)
    pub fn partial_eval(expr: ASTRepr<f64>, static_values: HashMap<String, f64>) -> PartialEvalTransformation {
        PartialEvalTransformation::new(expr, static_values)
    }
}

/// JIT compilation transformation
pub struct JitTransformation {
    expr: ASTRepr<f64>,
    registry: VariableRegistry,
    pub metadata: TransformationMetadata,
}

impl JitTransformation {
    fn new(expr: ASTRepr<f64>) -> Self {
        let mut metadata = TransformationMetadata::default();
        metadata.transformations.push("jit".to_string());
        metadata.compilation_hints.should_jit = true;
        metadata.compilation_hints.call_frequency_hint = CallFrequency::Frequent;

        Self {
            expr,
            registry: VariableRegistry::new(),
            metadata,
        }
    }

    /// Compile the JIT transformation
    pub fn compile(self) -> Result<CompiledRustFunction> {
        let codegen = RustCodeGenerator::new();
        let rust_code = codegen.generate_function_with_registry(
            &self.expr,
            "jit_compiled",
            "f64",
            &self.registry,
        )?;

        let compiler = RustCompiler::new();
        compiler.compile_and_load(&rust_code, "jit_compiled")
    }

    /// Chain with gradient computation
    pub fn grad(self) -> GradJitTransformation {
        GradJitTransformation::new(self.expr, self.metadata)
    }

    /// Chain with vectorization
    pub fn vmap(self, axis: usize) -> VmapJitTransformation {
        VmapJitTransformation::new(self.expr, self.metadata, axis)
    }
}

/// Gradient computation transformation
pub struct GradTransformation {
    expr: ASTRepr<f64>,
    registry: VariableRegistry,
    pub metadata: TransformationMetadata,
}

impl GradTransformation {
    fn new(expr: ASTRepr<f64>) -> Self {
        let mut metadata = TransformationMetadata::default();
        metadata.transformations.push("grad".to_string());
        metadata.compilation_hints.vectorizable = true;

        Self {
            expr,
            registry: VariableRegistry::new(),
            metadata,
        }
    }

    /// Compute the gradient
    pub fn compute(self) -> Result<TransformedExpr<f64>> {
        let mut config = SymbolicADConfig::default();
        config.num_variables = self.registry.len().max(1);

        let mut ad = SymbolicAD::with_config(config)?;
        let result = ad.compute_with_derivatives(&self.expr)?;

        // For now, return the first derivative if available
        let grad_expr = result.first_derivatives
            .into_iter()
            .next()
            .map(|(_, expr)| expr)
            .unwrap_or_else(|| ASTRepr::Constant(0.0));

        Ok(TransformedExpr {
            expr: grad_expr,
            metadata: self.metadata,
            registry: self.registry,
        })
    }

    /// Chain with JIT compilation
    pub fn jit(self) -> GradJitTransformation {
        GradJitTransformation::new(self.expr, self.metadata)
    }
}

/// Combined gradient + JIT transformation
pub struct GradJitTransformation {
    expr: ASTRepr<f64>,
    pub metadata: TransformationMetadata,
}

impl GradJitTransformation {
    fn new(expr: ASTRepr<f64>, mut metadata: TransformationMetadata) -> Self {
        if !metadata.transformations.contains(&"grad".to_string()) {
            metadata.transformations.push("grad".to_string());
        }
        if !metadata.transformations.contains(&"jit".to_string()) {
            metadata.transformations.push("jit".to_string());
        }
        metadata.compilation_hints.should_jit = true;
        metadata.compilation_hints.vectorizable = true;

        Self { expr, metadata }
    }

    /// Compile both function and gradient
    pub fn compile(self) -> Result<(CompiledRustFunction, CompiledRustFunction)> {
        // Compute gradient first
        let grad_transform = GradTransformation::new(self.expr.clone());
        let grad_result = grad_transform.compute()?;

        // Compile both function and gradient
        let codegen = RustCodeGenerator::new();
        
        let func_code = codegen.generate_function_generic(&self.expr, "func", "f64")?;
        let grad_code = codegen.generate_function_generic(&grad_result.expr, "grad", "f64")?;

        let compiler = RustCompiler::new();
        let func_compiled = compiler.compile_and_load(&func_code, "func")?;
        
        let compiler2 = RustCompiler::new();
        let grad_compiled = compiler2.compile_and_load(&grad_code, "grad")?;

        Ok((func_compiled, grad_compiled))
    }
}

/// Vectorization transformation (placeholder for future implementation)
pub struct VmapTransformation {
    expr: ASTRepr<f64>,
    axis: usize,
    pub metadata: TransformationMetadata,
}

impl VmapTransformation {
    fn new(expr: ASTRepr<f64>, axis: usize) -> Self {
        let mut metadata = TransformationMetadata::default();
        metadata.transformations.push(format!("vmap(axis={})", axis));
        metadata.compilation_hints.vectorizable = true;
        metadata.compilation_hints.memory_pattern = MemoryPattern::Sequential;

        Self { expr, axis, metadata }
    }

    /// Apply vectorization (placeholder)
    pub fn apply(self) -> Result<TransformedExpr<f64>> {
        // TODO: Implement actual vectorization logic
        // For now, return the original expression
        Ok(TransformedExpr {
            expr: self.expr,
            metadata: self.metadata,
            registry: VariableRegistry::new(),
        })
    }
}

/// Combined vectorization + JIT transformation
pub struct VmapJitTransformation {
    expr: ASTRepr<f64>,
    metadata: TransformationMetadata,
    axis: usize,
}

impl VmapJitTransformation {
    fn new(expr: ASTRepr<f64>, mut metadata: TransformationMetadata, axis: usize) -> Self {
        metadata.transformations.push(format!("vmap(axis={})", axis));
        metadata.compilation_hints.vectorizable = true;
        metadata.compilation_hints.memory_pattern = MemoryPattern::Sequential;

        Self { expr, metadata, axis }
    }
}

/// Domain analysis transformation
pub struct DomainTransformation {
    expr: ASTRepr<f64>,
    pub metadata: TransformationMetadata,
}

impl DomainTransformation {
    fn new(expr: ASTRepr<f64>) -> Self {
        let mut metadata = TransformationMetadata::default();
        metadata.transformations.push("domain_analysis".to_string());

        Self { expr, metadata }
    }

    /// Apply domain analysis
    pub fn analyze(self) -> Result<TransformedExpr<f64>> {
        let mut analyzer = IntervalDomainAnalyzer::new(0.0);
        
        // Set up some default domains for variables
        // In practice, these would come from user hints or inference
        analyzer.set_variable_domain(0, IntervalDomain::Top);
        
        let domain = analyzer.analyze_domain(&self.expr);
        
        let mut metadata = self.metadata;
        // Store domain information in metadata
        // This could be used by subsequent transformations
        
        Ok(TransformedExpr {
            expr: self.expr,
            metadata,
            registry: VariableRegistry::new(),
        })
    }

    /// Chain with JIT compilation
    pub fn jit(self) -> DomainJitTransformation {
        DomainJitTransformation::new(self.expr, self.metadata)
    }
}

/// Combined domain analysis + JIT transformation
pub struct DomainJitTransformation {
    expr: ASTRepr<f64>,
    metadata: TransformationMetadata,
}

impl DomainJitTransformation {
    fn new(expr: ASTRepr<f64>, mut metadata: TransformationMetadata) -> Self {
        if !metadata.transformations.contains(&"jit".to_string()) {
            metadata.transformations.push("jit".to_string());
        }
        metadata.compilation_hints.should_jit = true;

        Self { expr, metadata }
    }
}

/// Partial evaluation transformation
pub struct PartialEvalTransformation {
    expr: ASTRepr<f64>,
    static_values: HashMap<String, f64>,
    metadata: TransformationMetadata,
}

impl PartialEvalTransformation {
    fn new(expr: ASTRepr<f64>, static_values: HashMap<String, f64>) -> Self {
        let mut metadata = TransformationMetadata::default();
        metadata.transformations.push("partial_eval".to_string());
        
        Self { expr, static_values, metadata }
    }

    /// Apply partial evaluation
    pub fn apply(self) -> Result<TransformedExpr<f64>> {
        // TODO: Implement actual partial evaluation
        // For now, return the original expression
        Ok(TransformedExpr {
            expr: self.expr,
            metadata: self.metadata,
            registry: VariableRegistry::new(),
        })
    }

    /// Chain with JIT compilation
    pub fn jit(self) -> PartialEvalJitTransformation {
        PartialEvalJitTransformation::new(self.expr, self.static_values, self.metadata)
    }
}

/// Combined partial evaluation + JIT transformation
pub struct PartialEvalJitTransformation {
    expr: ASTRepr<f64>,
    static_values: HashMap<String, f64>,
    metadata: TransformationMetadata,
}

impl PartialEvalJitTransformation {
    fn new(expr: ASTRepr<f64>, static_values: HashMap<String, f64>, mut metadata: TransformationMetadata) -> Self {
        if !metadata.transformations.contains(&"jit".to_string()) {
            metadata.transformations.push("jit".to_string());
        }
        metadata.compilation_hints.should_jit = true;

        Self { expr, static_values, metadata }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_jit_transformation() {
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        let jit_transform = Transformations::jit(expr);
        
        assert_eq!(jit_transform.metadata.transformations, vec!["jit"]);
        assert!(jit_transform.metadata.compilation_hints.should_jit);
    }

    #[test]
    fn test_grad_transformation() {
        let expr = ASTEval::mul(ASTEval::var(0), ASTEval::var(0)); // x²
        let grad_transform = Transformations::grad(expr);
        
        assert_eq!(grad_transform.metadata.transformations, vec!["grad"]);
        assert!(grad_transform.metadata.compilation_hints.vectorizable);
    }

    #[test]
    fn test_transformation_chaining() {
        let expr = ASTEval::mul(ASTEval::var(0), ASTEval::var(0)); // x²
        let chained = Transformations::jit(expr).grad();
        
        assert!(chained.metadata.transformations.contains(&"jit".to_string()));
        assert!(chained.metadata.transformations.contains(&"grad".to_string()));
        assert!(chained.metadata.compilation_hints.should_jit);
        assert!(chained.metadata.compilation_hints.vectorizable);
    }

    #[test]
    fn test_domain_analysis_transformation() {
        let expr = ASTEval::ln(ASTEval::var(0)); // ln(x)
        let domain_transform = Transformations::analyze_domains(expr);
        
        assert_eq!(domain_transform.metadata.transformations, vec!["domain_analysis"]);
    }
} 