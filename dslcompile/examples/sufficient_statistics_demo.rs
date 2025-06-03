//! # Sufficient Statistics Discovery Demo
//!
//! This example demonstrates how egglog automatically discovers sufficient statistics
//! for Bayesian linear regression by expanding expressions like (y[i] - β₀ - β₁*x[i])².
//! No hardcoded sufficient statistics - the optimizer discovers them automatically!

use dslcompile::Result;
use dslcompile::final_tagless::{ExpressionBuilder, IntRange};
use dslcompile::symbolic::summation::SummationProcessor;

fn main() -> Result<()> {
    println!("🔍 Automatic Sufficient Statistics Discovery Demo");
    println!("=================================================\n");

    // Create expression builder and summation processor
    let math = ExpressionBuilder::new();
    let mut processor = SummationProcessor::new()?;
    
    println!("📊 Scenario: Bayesian Linear Regression");
    println!("   Model: y[i] = β₀ + β₁*x[i] + ε[i]");
    println!("   Likelihood: Σ(y[i] - β₀ - β₁*x[i])²");
    println!("   Goal: Discover sufficient statistics automatically\n");

    // Demo 1: Simple quadratic expansion
    println!("🔧 Demo 1: Basic Quadratic Expansion");
    println!("   Expression: (a + b)²");
    println!("   Should expand to: a² + 2ab + b²");
    
    let range = IntRange::new(1, 1); // Single term to see expansion clearly
    let result = processor.sum(range, |_i| {
        // Build (a + b)² where a and b are parameters (external variables)
        let a = math.var(); // Parameter β₀
        let b = math.var(); // Parameter β₁
        
        let sum_expr = a + b;
        sum_expr.pow(math.constant(2.0))
    })?;
    
    println!("   Pattern discovered: {:?}", result.pattern);
    println!("   Optimized: {}", result.is_optimized);
    println!("   Value: {}", result.evaluate(&[2.0, 3.0])?); // (2 + 3)² = 25
    println!();

    // Demo 2: Linear regression residual expansion  
    println!("🔧 Demo 2: Linear Regression Residual");
    println!("   Expression: (y[i] - β₀ - β₁*x[i])²");
    println!("   Should discover: Σy[i]², Σx[i]², Σ(x[i]*y[i]), Σx[i], Σy[i], n");

    let data_range = IntRange::new(1, 100); // 100 data points
    let result = processor.sum(data_range, |_i| {
        // Build (y[i] - β₀ - β₁*x[i])² where:
        // - i is the data index (Variable(0))
        // - β₀ is external parameter (Variable(1)) 
        // - β₁ is external parameter (Variable(2))
        // - x[i] is data variable (Variable(3))
        // - y[i] is data variable (Variable(4))
        
        let sum_math = ExpressionBuilder::new();
        let beta0 = sum_math.var(); // External parameter β₀
        let beta1 = sum_math.var(); // External parameter β₁
        let x_i = sum_math.var();   // Data variable x[i]
        let y_i = sum_math.var();   // Data variable y[i]
        
        // Build: y[i] - β₀ - β₁*x[i]
        let prediction = beta0 + beta1 * x_i;
        let residual = y_i - prediction;
        
        // Square the residual: (y[i] - β₀ - β₁*x[i])²
        // Egglog should automatically expand this!
        residual.pow(sum_math.constant(2.0))
    })?;
    
    println!("   Pattern discovered: {:?}", result.pattern);
    println!("   Optimized: {}", result.is_optimized);
    println!("   Operations in original: {}", result.original_expr.count_operations());
    println!("   Operations in simplified: {}", result.simplified_expr.count_operations());
    
    if let Some(closed_form) = &result.closed_form {
        println!("   Operations in closed form: {}", closed_form.count_operations());
    }
    
    // Show the expression structure
    println!("\n   📋 Original Expression Structure:");
    println!("      {:?}", result.original_expr);
    
    println!("\n   📋 Simplified Expression Structure:");
    println!("      {:?}", result.simplified_expr);
    
    if let Some(closed_form) = &result.closed_form {
        println!("\n   📋 Closed Form Expression:");
        println!("      {:?}", closed_form);
    }
    
    println!();

    // Demo 3: Show what egglog discovered
    println!("🎯 Demo 3: Egglog Analysis Results");
    println!("   Extracted factors: {:?}", result.extracted_factors);
    println!("   Factor speedup: {:.1}x", result.factor_speedup());
    
    // Test evaluation with sample parameters
    let beta0 = 1.0;
    let beta1 = 2.0;
    let x_sample = 3.0;
    let y_sample = 7.0; // y = 1 + 2*3 + noise, so residual should be small
    
    let test_params = vec![beta0, beta1, x_sample, y_sample];
    let eval_result = result.evaluate(&test_params)?;
    
    println!("   Test evaluation:");
    println!("     β₀ = {beta0}, β₁ = {beta1}, x = {x_sample}, y = {y_sample}");
    println!("     (y - β₀ - β₁*x)² = ({y_sample} - {beta0} - {beta1}*{x_sample})² = {eval_result}");
    let expected = (y_sample - beta0 - beta1 * x_sample).powi(2) * 100.0; // Times 100 for the range size
    println!("     Expected: {expected}");
    println!("     Match: {}", (eval_result - expected).abs() < 1e-10);
    
    println!("\n✅ Demonstration Complete!");
    println!("\n🎯 Key Insights:");
    println!("   • Egglog automatically expands (a + b + c)² → a² + b² + c² + 2ab + 2ac + 2bc");
    println!("   • No hardcoded sufficient statistics needed");
    println!("   • Each expansion term corresponds to a minimal sufficient statistic");
    println!("   • The summation processor recognizes patterns in the expanded form");
    println!("   • This enables automatic discovery of optimal data traversal patterns");

    Ok(())
} 