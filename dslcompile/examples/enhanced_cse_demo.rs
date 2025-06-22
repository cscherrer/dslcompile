//! Enhanced CSE Demonstration with Cost Visibility
//!
//! This example demonstrates the improved Common Subexpression Elimination (CSE)
//! capabilities in Phase 3, including:
//! - Advanced cost analysis with detailed breakdowns
//! - Automatic CSE candidate detection
//! - Egglog integration for cost-aware optimization
//! - Multi-level CSE optimization recommendations

use dslcompile::prelude::*;

fn main() {
    println!("=== Enhanced CSE with Cost Visibility Demo ===\n");

    println!("Variables: x = Variable(0), y = Variable(1), z = Variable(2)\n");

    // Example 1: Simple repeated subexpression
    println!("--- Example 1: Basic CSE Analysis ---");
    let mut ctx1 = DynamicContext::new();
    let x1: DynamicExpr<f64, 0> = ctx1.var();
    let y1: DynamicExpr<f64, 0> = ctx1.var();
    let z1: DynamicExpr<f64, 0> = ctx1.var();

    let expensive_computation = (&x1 + &y1).ln() * (&x1 + &y1).sin();
    let repeated_expr = &expensive_computation + &expensive_computation * &z1;

    println!("Expression: ln(x + y) * sin(x + y) + ln(x + y) * sin(x + y) * z");

    let analysis = ctx1.analyze_cse(&repeated_expr);
    print_analysis_results(&analysis);

    // Example 2: Complex expression with multiple CSE opportunities
    println!("\n--- Example 2: Multi-level CSE Opportunities ---");
    let mut ctx2 = DynamicContext::new();
    let x2: DynamicExpr<f64, 0> = ctx2.var();
    let y2: DynamicExpr<f64, 0> = ctx2.var();
    let z2: DynamicExpr<f64, 0> = ctx2.var();

    let x_squared = &x2 * &x2;
    let y_squared = &y2 * &y2;
    let sum_squares = &x_squared + &y_squared;
    let complex_base = sum_squares.sqrt(); // sqrt(x² + y²)
    let ln_base = complex_base.ln();
    let sin_base = (&x_squared + &y_squared).sqrt().sin();
    let mul_base = (&x_squared + &y_squared).sqrt() * &z2;
    let final_sum_squares = &x_squared + &y_squared;
    let complex_expr = &ln_base + &sin_base + &mul_base + &final_sum_squares;

    println!("Expression: ln(√(x² + y²)) + sin(√(x² + y²)) + √(x² + y²) * z + (x² + y²)");

    let complex_analysis = ctx2.analyze_cse(&complex_expr);
    print_analysis_results(&complex_analysis);

    // Example 3: Manual CSE with cost comparison
    println!("\n--- Example 3: Manual CSE with Cost Comparison ---");
    let mut ctx3 = DynamicContext::new();
    let x3: DynamicExpr<f64, 0> = ctx3.var();
    let y3: DynamicExpr<f64, 0> = ctx3.var();

    let log_x = x3.ln();
    let log_y = y3.ln();
    let log_sum_base = &log_x + &log_y;
    let original_expr = &log_sum_base * &log_sum_base + log_sum_base.sin();
    println!("Original: (ln(x) + ln(y)) * (ln(x) + ln(y)) + sin(ln(x) + ln(y))");

    let original_analysis = ctx3.analyze_cse(&original_expr);
    println!("Original cost: {:.1}", original_analysis.original_cost);
    print_cost_breakdown(&original_analysis.cost_breakdown);

    // Apply manual CSE - create fresh variables
    let mut ctx3b = DynamicContext::new();
    let x3b: DynamicExpr<f64, 0> = ctx3b.var();
    let y3b: DynamicExpr<f64, 0> = ctx3b.var();

    let cse_expr = ctx3b.let_bind(x3b.ln() + y3b.ln(), |log_sum| {
        let log_sum_expr = DynamicExpr::from(log_sum);
        &log_sum_expr * &log_sum_expr + log_sum_expr.sin()
    });
    println!("\nWith CSE: let log_sum = ln(x) + ln(y) in log_sum² + sin(log_sum)");

    let cse_analysis = ctx3b.analyze_cse(&cse_expr);
    println!("CSE cost: {:.1}", cse_analysis.original_cost);
    print_cost_breakdown(&cse_analysis.cost_breakdown);

    let savings = original_analysis.original_cost - cse_analysis.original_cost;
    println!(
        "Cost savings: {:.1} ({:.1}% reduction)",
        savings,
        (savings / original_analysis.original_cost) * 100.0
    );

    // Example 4: CSE optimization suggestions
    println!("\n--- Example 4: Automatic CSE Suggestions ---");
    let mut ctx4 = DynamicContext::new();
    let x4: DynamicExpr<f64, 0> = ctx4.var();
    let y4: DynamicExpr<f64, 0> = ctx4.var();

    let exp_x = x4.exp();
    let exp_y = y4.exp();
    let exp_sum = &exp_x + &exp_y;
    let sin_exp = exp_sum.sin();
    let cos_exp = (&exp_x + &exp_y).cos();
    let final_exp_sum = &exp_x + &exp_y;
    let optimization_target = &sin_exp * &cos_exp + &final_exp_sum;
    println!("Expression: sin(exp(x) + exp(y)) * cos(exp(x) + exp(y)) + (exp(x) + exp(y))");

    let suggestions = ctx4.suggest_cse_optimizations(&optimization_target);
    print_optimization_suggestions(&suggestions);

    // Example 5: Nested CSE with scope management
    println!("\n--- Example 5: Nested CSE with Scope Safety ---");
    let mut ctx5 = DynamicContext::new();
    let x5: DynamicExpr<f64, 0> = ctx5.var();
    let y5: DynamicExpr<f64, 0> = ctx5.var();

    let trig_base = x5.sin() + y5.cos();
    let nested_cse = ctx5.let_bind(trig_base, |trig_sum| {
        let trig_sum_expr = DynamicExpr::from(trig_sum.clone());
        let squared_expr = &trig_sum_expr * &trig_sum_expr;
        let bound_var_expr = DynamicExpr::from(trig_sum);
        &squared_expr + &bound_var_expr + squared_expr.ln()
    });
    println!("Nested CSE: let trig_sum = sin(x) + cos(y) in");
    println!("           let squared = trig_sum² in");
    println!("           squared + trig_sum + ln(squared)");

    let nested_analysis = ctx5.analyze_cse(&nested_cse);
    println!("Nested CSE cost: {:.1}", nested_analysis.original_cost);
    print_cost_breakdown(&nested_analysis.cost_breakdown);

    // Example 6: Custom threshold analysis
    println!("\n--- Example 6: Custom Threshold Analysis ---");
    let mut ctx6 = DynamicContext::new();
    let x6: DynamicExpr<f64, 0> = ctx6.var();
    let y6: DynamicExpr<f64, 0> = ctx6.var();

    let threshold_expr = (&x6 + &y6) * (&x6 + &y6) + (&x6 + &y6).sin();
    let custom_analysis = ctx6.analyze_cse_with_thresholds(
        &threshold_expr,
        2.0, // Lower cost threshold
        2,   // Frequency threshold
        2.0, // Higher complexity weight
    );

    println!("Custom threshold analysis (more aggressive CSE detection):");
    print_analysis_results(&custom_analysis);

    println!("\n=== Summary ===");
    println!("✓ Enhanced cost analysis with detailed breakdowns");
    println!("✓ Automatic CSE candidate detection");
    println!("✓ Cost-aware optimization recommendations");
    println!("✓ Multi-level CSE support with scope safety");
    println!("✓ Customizable analysis thresholds");
    println!("✓ Integration with egglog cost models");
}

fn print_analysis_results(analysis: &CSEAnalysis) {
    println!("Cost Analysis:");
    println!("  Original cost: {:.1}", analysis.original_cost);
    println!("  Optimized cost: {:.1}", analysis.optimized_cost);
    println!("  Potential savings: {:.1}", analysis.savings);

    if !analysis.candidates.is_empty() {
        println!("  CSE Candidates:");
        for candidate in &analysis.candidates {
            println!(
                "    - Frequency: {}, Cost: {:.1}, Savings: {:.1}",
                candidate.frequency, candidate.computation_cost, candidate.potential_savings
            );
        }
    } else {
        println!("  No beneficial CSE candidates found");
    }

    print_cost_breakdown(&analysis.cost_breakdown);
}

fn print_cost_breakdown(breakdown: &CostBreakdown) {
    println!("  Cost Breakdown:");
    println!("    Operations: {:.1}", breakdown.operation_cost);
    println!("    Transcendental: {:.1}", breakdown.transcendental_cost);
    println!("    Summations: {:.1}", breakdown.summation_cost);
    println!("    CSE: {:.1}", breakdown.cse_cost);
    println!("    Variables: {:.1}", breakdown.variable_cost);
    println!("    Total: {:.1}", breakdown.total());
}

fn print_optimization_suggestions(suggestions: &[CSEOptimization]) {
    if suggestions.is_empty() {
        println!("No optimization suggestions above threshold");
        return;
    }

    println!("Optimization Suggestions:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!("  {}. Priority: {:?}", i + 1, suggestion.recommended_action);
        println!("     Frequency: {}", suggestion.frequency);
        println!("     Cost savings: {:.1}", suggestion.cost_savings);
        println!("     Complexity score: {:.1}", suggestion.complexity_score);
    }
}
