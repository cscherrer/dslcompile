//! JAX-inspired Computation Tracing
//!
//! This module provides tracing capabilities to analyze computation patterns,
//! similar to JAX's tracing system for understanding program structure.

use crate::final_tagless::ASTRepr;
use crate::interval_domain::IntervalDomain;
use std::collections::{HashMap, HashSet};

/// Computation trace capturing execution patterns
#[derive(Debug, Clone)]
pub struct ComputationTrace {
    /// Operations in execution order
    pub operations: Vec<TracedOperation>,
    /// Variable access patterns
    pub variable_access: HashMap<usize, AccessPattern>,
    /// Memory access patterns
    pub memory_patterns: Vec<MemoryAccess>,
    /// Computational complexity estimate
    pub complexity: ComplexityEstimate,
}

/// A traced operation with metadata
#[derive(Debug, Clone)]
pub struct TracedOperation {
    /// The operation type
    pub op_type: OperationType,
    /// Input dependencies
    pub inputs: Vec<usize>,
    /// Output variable index
    pub output: usize,
    /// Estimated cost
    pub cost: f64,
    /// Domain information if available
    pub domain: Option<IntervalDomain<f64>>,
}

/// Types of operations that can be traced
#[derive(Debug, Clone, PartialEq)]
pub enum OperationType {
    /// Arithmetic operations
    Add, Sub, Mul, Div, Pow,
    /// Unary operations
    Neg, Ln, Exp, Sin, Cos, Sqrt,
    /// Constants and variables
    Constant(f64),
    Variable(usize),
    /// Composite operations
    Polynomial { degree: usize },
    Rational { num_degree: usize, den_degree: usize },
}

/// Variable access patterns for optimization
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Number of times accessed
    pub access_count: usize,
    /// Whether accessed sequentially
    pub sequential: bool,
    /// Whether the variable is read-only
    pub read_only: bool,
    /// Estimated data size
    pub data_size_hint: Option<usize>,
}

/// Memory access patterns
#[derive(Debug, Clone)]
pub struct MemoryAccess {
    /// Variable being accessed
    pub variable: usize,
    /// Access type
    pub access_type: AccessType,
    /// Position in computation
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
}

/// Computational complexity estimate
#[derive(Debug, Clone)]
pub struct ComplexityEstimate {
    /// Total operation count
    pub operation_count: usize,
    /// Estimated FLOPs
    pub flops: f64,
    /// Memory bandwidth requirement
    pub memory_bandwidth: f64,
    /// Parallelization potential (0.0 = sequential, 1.0 = fully parallel)
    pub parallelization_potential: f64,
    /// Vectorization potential
    pub vectorization_potential: f64,
}

/// Computation tracer
pub struct ComputationTracer {
    /// Current trace being built
    trace: ComputationTrace,
    /// Next available variable index
    next_var_index: usize,
    /// Variable mapping for expressions
    var_mapping: HashMap<String, usize>,
}

impl ComputationTracer {
    /// Create a new computation tracer
    pub fn new() -> Self {
        Self {
            trace: ComputationTrace {
                operations: Vec::new(),
                variable_access: HashMap::new(),
                memory_patterns: Vec::new(),
                complexity: ComplexityEstimate {
                    operation_count: 0,
                    flops: 0.0,
                    memory_bandwidth: 0.0,
                    parallelization_potential: 0.0,
                    vectorization_potential: 0.0,
                },
            },
            next_var_index: 0,
            var_mapping: HashMap::new(),
        }
    }

    /// Trace an expression and return the computation trace
    pub fn trace_expression(&mut self, expr: &ASTRepr<f64>) -> ComputationTrace {
        self.trace_recursive(expr);
        self.analyze_patterns();
        self.trace.clone()
    }

    /// Recursively trace an expression
    fn trace_recursive(&mut self, expr: &ASTRepr<f64>) -> usize {
        match expr {
            ASTRepr::Constant(value) => {
                let output_var = self.allocate_var();
                self.add_operation(TracedOperation {
                    op_type: OperationType::Constant(*value),
                    inputs: vec![],
                    output: output_var,
                    cost: 0.0, // Constants are free
                    domain: Some(IntervalDomain::Constant(*value)),
                });
                output_var
            }

            ASTRepr::Variable(index) => {
                let output_var = self.allocate_var();
                self.add_operation(TracedOperation {
                    op_type: OperationType::Variable(*index),
                    inputs: vec![],
                    output: output_var,
                    cost: 1.0, // Memory access cost
                    domain: None, // Unknown domain for variables
                });
                
                // Track variable access
                self.track_variable_access(*index);
                output_var
            }

            ASTRepr::Add(left, right) => {
                let left_var = self.trace_recursive(left);
                let right_var = self.trace_recursive(right);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Add,
                    inputs: vec![left_var, right_var],
                    output: output_var,
                    cost: 1.0, // Single FLOP
                    domain: None, // Would need domain analysis
                });
                output_var
            }

            ASTRepr::Sub(left, right) => {
                let left_var = self.trace_recursive(left);
                let right_var = self.trace_recursive(right);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Sub,
                    inputs: vec![left_var, right_var],
                    output: output_var,
                    cost: 1.0,
                    domain: None,
                });
                output_var
            }

            ASTRepr::Mul(left, right) => {
                let left_var = self.trace_recursive(left);
                let right_var = self.trace_recursive(right);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Mul,
                    inputs: vec![left_var, right_var],
                    output: output_var,
                    cost: 1.0,
                    domain: None,
                });
                output_var
            }

            ASTRepr::Div(left, right) => {
                let left_var = self.trace_recursive(left);
                let right_var = self.trace_recursive(right);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Div,
                    inputs: vec![left_var, right_var],
                    output: output_var,
                    cost: 2.0, // Division is more expensive
                    domain: None,
                });
                output_var
            }

            ASTRepr::Pow(base, exp) => {
                let base_var = self.trace_recursive(base);
                let exp_var = self.trace_recursive(exp);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Pow,
                    inputs: vec![base_var, exp_var],
                    output: output_var,
                    cost: 10.0, // Power is expensive
                    domain: None,
                });
                output_var
            }

            ASTRepr::Neg(inner) => {
                let inner_var = self.trace_recursive(inner);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Neg,
                    inputs: vec![inner_var],
                    output: output_var,
                    cost: 1.0,
                    domain: None,
                });
                output_var
            }

            ASTRepr::Ln(inner) => {
                let inner_var = self.trace_recursive(inner);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Ln,
                    inputs: vec![inner_var],
                    output: output_var,
                    cost: 5.0, // Transcendental functions are expensive
                    domain: None,
                });
                output_var
            }

            ASTRepr::Exp(inner) => {
                let inner_var = self.trace_recursive(inner);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Exp,
                    inputs: vec![inner_var],
                    output: output_var,
                    cost: 5.0,
                    domain: None,
                });
                output_var
            }

            ASTRepr::Sin(inner) => {
                let inner_var = self.trace_recursive(inner);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Sin,
                    inputs: vec![inner_var],
                    output: output_var,
                    cost: 5.0,
                    domain: None,
                });
                output_var
            }

            ASTRepr::Cos(inner) => {
                let inner_var = self.trace_recursive(inner);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Cos,
                    inputs: vec![inner_var],
                    output: output_var,
                    cost: 5.0,
                    domain: None,
                });
                output_var
            }

            ASTRepr::Sqrt(inner) => {
                let inner_var = self.trace_recursive(inner);
                let output_var = self.allocate_var();
                
                self.add_operation(TracedOperation {
                    op_type: OperationType::Sqrt,
                    inputs: vec![inner_var],
                    output: output_var,
                    cost: 3.0,
                    domain: None,
                });
                output_var
            }
        }
    }

    /// Allocate a new variable index
    fn allocate_var(&mut self) -> usize {
        let var = self.next_var_index;
        self.next_var_index += 1;
        var
    }

    /// Add an operation to the trace
    fn add_operation(&mut self, op: TracedOperation) {
        self.trace.complexity.operation_count += 1;
        self.trace.complexity.flops += op.cost;
        self.trace.operations.push(op);
    }

    /// Track variable access patterns
    fn track_variable_access(&mut self, var_index: usize) {
        let access = self.trace.variable_access.entry(var_index).or_insert(AccessPattern {
            access_count: 0,
            sequential: true,
            read_only: true,
            data_size_hint: None,
        });
        access.access_count += 1;
    }

    /// Analyze patterns in the trace
    fn analyze_patterns(&mut self) {
        self.analyze_parallelization_potential();
        self.analyze_vectorization_potential();
        self.analyze_memory_patterns();
    }

    /// Analyze parallelization potential
    fn analyze_parallelization_potential(&mut self) {
        // Simple heuristic: operations that don't depend on each other can be parallelized
        let total_ops = self.trace.operations.len();
        if total_ops == 0 {
            return;
        }

        let mut dependency_graph: HashMap<usize, HashSet<usize>> = HashMap::new();
        
        // Build dependency graph
        for op in &self.trace.operations {
            let deps: HashSet<usize> = op.inputs.iter().copied().collect();
            dependency_graph.insert(op.output, deps);
        }

        // Estimate parallelizable operations (very simplified)
        let mut parallelizable = 0;
        for op in &self.trace.operations {
            if op.inputs.len() <= 1 {
                parallelizable += 1;
            }
        }

        self.trace.complexity.parallelization_potential = 
            parallelizable as f64 / total_ops as f64;
    }

    /// Analyze vectorization potential
    fn analyze_vectorization_potential(&mut self) {
        // Count operations that can be vectorized
        let vectorizable_ops = self.trace.operations.iter()
            .filter(|op| matches!(op.op_type, 
                OperationType::Add | OperationType::Sub | 
                OperationType::Mul | OperationType::Div))
            .count();

        let total_ops = self.trace.operations.len();
        if total_ops > 0 {
            self.trace.complexity.vectorization_potential = 
                vectorizable_ops as f64 / total_ops as f64;
        }
    }

    /// Analyze memory access patterns
    fn analyze_memory_patterns(&mut self) {
        // Track memory accesses
        for (pos, op) in self.trace.operations.iter().enumerate() {
            if let OperationType::Variable(var_index) = op.op_type {
                self.trace.memory_patterns.push(MemoryAccess {
                    variable: var_index,
                    access_type: AccessType::Read,
                    position: pos,
                });
            }
        }

        // Estimate memory bandwidth requirement
        let memory_ops = self.trace.memory_patterns.len();
        self.trace.complexity.memory_bandwidth = memory_ops as f64 * 8.0; // 8 bytes per f64
    }
}

impl Default for ComputationTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trace analysis utilities
pub struct TraceAnalyzer;

impl TraceAnalyzer {
    /// Recommend compilation strategy based on trace
    pub fn recommend_compilation_strategy(trace: &ComputationTrace) -> CompilationRecommendation {
        let complexity = &trace.complexity;
        
        if complexity.operation_count < 10 {
            CompilationRecommendation::DirectEvaluation
        } else if complexity.vectorization_potential > 0.7 {
            CompilationRecommendation::VectorizedCompilation
        } else if complexity.parallelization_potential > 0.5 {
            CompilationRecommendation::ParallelCompilation
        } else if complexity.flops > 100.0 {
            CompilationRecommendation::JITCompilation
        } else {
            CompilationRecommendation::StandardCompilation
        }
    }

    /// Identify optimization opportunities
    pub fn identify_optimizations(trace: &ComputationTrace) -> Vec<OptimizationOpportunity> {
        let mut opportunities = Vec::new();

        // Check for common subexpressions
        if Self::has_repeated_patterns(trace) {
            opportunities.push(OptimizationOpportunity::CommonSubexpressionElimination);
        }

        // Check for constant folding opportunities
        if Self::has_constant_operations(trace) {
            opportunities.push(OptimizationOpportunity::ConstantFolding);
        }

        // Check for vectorization opportunities
        if trace.complexity.vectorization_potential > 0.5 {
            opportunities.push(OptimizationOpportunity::Vectorization);
        }

        // Check for parallelization opportunities
        if trace.complexity.parallelization_potential > 0.3 {
            opportunities.push(OptimizationOpportunity::Parallelization);
        }

        opportunities
    }

    /// Check for repeated computation patterns
    fn has_repeated_patterns(trace: &ComputationTrace) -> bool {
        // Simple heuristic: look for identical operation sequences
        trace.operations.len() > 5 // Placeholder logic
    }

    /// Check for operations on constants
    fn has_constant_operations(trace: &ComputationTrace) -> bool {
        trace.operations.iter().any(|op| {
            matches!(op.op_type, OperationType::Constant(_))
        })
    }
}

/// Compilation recommendations based on trace analysis
#[derive(Debug, Clone, PartialEq)]
pub enum CompilationRecommendation {
    DirectEvaluation,
    StandardCompilation,
    JITCompilation,
    VectorizedCompilation,
    ParallelCompilation,
}

/// Optimization opportunities identified from traces
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationOpportunity {
    CommonSubexpressionElimination,
    ConstantFolding,
    Vectorization,
    Parallelization,
    MemoryLayoutOptimization,
    LoopFusion,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::final_tagless::{ASTEval, ASTMathExpr};

    #[test]
    fn test_simple_trace() {
        let mut tracer = ComputationTracer::new();
        let expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        
        let trace = tracer.trace_expression(&expr);
        
        assert_eq!(trace.operations.len(), 3); // var, constant, add
        assert!(trace.complexity.operation_count > 0);
    }

    #[test]
    fn test_complex_trace() {
        let mut tracer = ComputationTracer::new();
        let expr = ASTEval::mul(
            ASTEval::add(ASTEval::var(0), ASTEval::var(1)),
            ASTEval::sin(ASTEval::var(0))
        );
        
        let trace = tracer.trace_expression(&expr);
        
        assert!(trace.operations.len() > 3);
        assert!(trace.complexity.flops > 0.0);
    }

    #[test]
    fn test_compilation_recommendation() {
        let mut tracer = ComputationTracer::new();
        let simple_expr = ASTEval::add(ASTEval::var(0), ASTEval::constant(1.0));
        let trace = tracer.trace_expression(&simple_expr);
        
        let recommendation = TraceAnalyzer::recommend_compilation_strategy(&trace);
        assert_eq!(recommendation, CompilationRecommendation::DirectEvaluation);
    }

    #[test]
    fn test_optimization_identification() {
        let mut tracer = ComputationTracer::new();
        let expr = ASTEval::add(ASTEval::constant(1.0), ASTEval::constant(2.0));
        let trace = tracer.trace_expression(&expr);
        
        let optimizations = TraceAnalyzer::identify_optimizations(&trace);
        assert!(optimizations.contains(&OptimizationOpportunity::ConstantFolding));
    }
} 