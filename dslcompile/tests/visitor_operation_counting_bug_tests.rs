//! Test to reproduce and fix the visitor operation counting bug
//!
//! The bug shows operation counts as 0 when they should be much higher

use dslcompile::prelude::*;
use dslcompile::ast::ast_utils::visitors::{OperationCountVisitor, SummationCountVisitor, DepthVisitor};
use frunk::hlist;

#[test]
fn test_simple_expression_should_have_operations() {
    // Simple expression that should have 1 addition operation
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    
    let expr = &x + &y;
    let ast = ctx.to_ast(&expr);
    
    println!("Simple addition AST: {:#?}", ast);
    
    let op_count = OperationCountVisitor::count_operations(&ast);
    println!("Operation count: {}", op_count);
    
    // This should be at least 1 (for the addition), not 0!
    assert!(op_count >= 1, "Expected at least 1 operation, got {}", op_count);
}

#[test]
fn test_complex_expression_should_have_many_operations() {
    // Complex expression like in Normal log-density
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    let mu = ctx.var::<f64>();
    let sigma = ctx.var::<f64>();
    
    // Simplified normal log-density: -0.5 * ((x - μ) / σ)²
    let centered = &x - &mu;           // 1 subtraction
    let standardized = &centered / &sigma;  // 1 division  
    let squared = &standardized * &standardized;  // 1 multiplication
    let log_density = -0.5 * &squared;       // 1 multiplication (with constant)
    
    let ast = ctx.to_ast(&log_density);
    
    println!("Complex expression AST: {:#?}", ast);
    
    let op_count = OperationCountVisitor::count_operations(&ast);
    println!("Operation count: {}", op_count);
    
    // Should have at least 4 operations: sub, div, mul, mul
    assert!(op_count >= 4, "Expected at least 4 operations, got {}", op_count);
}

#[test]
fn test_summation_should_have_operations_and_summations() {
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>();
    
    let data = vec![1.0, 2.0, 3.0];
    let sum_expr = ctx.sum(&data, |x| &x + &a); // Addition inside summation
    
    let ast = ctx.to_ast(&sum_expr);
    
    println!("Summation AST: {:#?}", ast);
    
    let op_count = OperationCountVisitor::count_operations(&ast);
    let sum_count = SummationCountVisitor::count_summations(&ast);
    let depth = DepthVisitor::compute_depth(&ast);
    
    println!("Operation count: {}", op_count);
    println!("Summation count: {}", sum_count);
    println!("Depth: {}", depth);
    
    // Should have at least 1 summation
    assert!(sum_count >= 1, "Expected at least 1 summation, got {}", sum_count);
    
    // Should have operations for the addition inside the lambda
    assert!(op_count >= 1, "Expected at least 1 operation, got {}", op_count);
    
    // Should have depth > 1
    assert!(depth > 1, "Expected depth > 1, got {}", depth);
}

#[test]
fn test_deeply_nested_should_have_many_operations() {
    let mut ctx = DynamicContext::new();
    let mut expr = ctx.var::<f64>();
    
    // Create a deeply nested expression with many operations
    for i in 1..=10 {
        expr = &expr + (i as f64);  // Each iteration adds 1 operation
    }
    
    let ast = ctx.to_ast(&expr);
    
    let op_count = OperationCountVisitor::count_operations(&ast);
    let depth = DepthVisitor::compute_depth(&ast);
    
    println!("Deeply nested - Operation count: {}", op_count);
    println!("Deeply nested - Depth: {}", depth);
    
    // Should have 10 addition operations
    assert!(op_count >= 10, "Expected at least 10 operations, got {}", op_count);
    
    // Should have significant depth
    assert!(depth >= 5, "Expected depth >= 5, got {}", depth);
}

#[test]
fn test_debug_ast_structure_for_operation_counting() {
    // Debug test to understand why operation counting might be failing
    let mut ctx = DynamicContext::new();
    let x = ctx.var::<f64>();
    let y = ctx.var::<f64>();
    
    let simple_add = &x + &y;
    let ast = ctx.to_ast(&simple_add);
    
    println!("=== DEBUG AST STRUCTURE ===");
    println!("AST: {:#?}", ast);
    
    // Manually examine the AST structure
    match &ast {
        dslcompile::ast::ast_repr::ASTRepr::Add(left, right) => {
            println!("Found Add node!");
            println!("  Left: {:#?}", left);
            println!("  Right: {:#?}", right);
        }
        other => {
            println!("Expected Add node, got: {:#?}", other);
        }
    }
    
    let op_count = OperationCountVisitor::count_operations(&ast);
    println!("Final operation count: {}", op_count);
}