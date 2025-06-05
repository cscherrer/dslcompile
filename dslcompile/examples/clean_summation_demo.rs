/// Clean Functional Summation Optimization Demo
/// 
/// This demonstrates the simple functional approach without complex state:
/// - Direct recursive evaluation
/// - No complex SummationResult structures
/// - Composable optimizations

use dslcompile::ast::ASTRepr;
use dslcompile::Result;

/// Simple functional summation optimizer
struct SummationOptimizer;

impl SummationOptimizer {
    fn new() -> Self {
        Self
    }
    
    /// Clean recursive optimization - returns final value directly
    fn optimize_summation(&self, start: i64, end: i64, expr: ASTRepr<f64>) -> Result<f64> {
        match expr {
            // Sum splitting: Σ(a + b) = Σ(a) + Σ(b)
            ASTRepr::Add(left, right) => {
                let left_val = self.optimize_summation(start, end, *left)?;
                let right_val = self.optimize_summation(start, end, *right)?;
                Ok(left_val + right_val)
            }
            
            // Factor extraction: Σ(k * f) = k * Σ(f)
            ASTRepr::Mul(left, right) => {
                if let ASTRepr::Constant(factor) = left.as_ref() {
                    let inner_val = self.optimize_summation(start, end, *right)?;
                    Ok(factor * inner_val)
                } else if let ASTRepr::Constant(factor) = right.as_ref() {
                    let inner_val = self.optimize_summation(start, end, *left)?;
                    Ok(factor * inner_val)
                } else {
                    // No constant factor, fall back to numerical
                    self.evaluate_numerically(start, end, &ASTRepr::Mul(left, right))
                }
            }
            
            // Constant: Σ(c) = c * n
            ASTRepr::Constant(value) => {
                let n = (end - start + 1) as f64;
                Ok(value * n)
            }
            
            // Variable (index variable): Σ(i) = n(n+1)/2
            ASTRepr::Variable(0) => {
                let n = end as f64;
                Ok(n * (n + 1.0) / 2.0)
            }
            
            // Power of index variable: Σ(i^k)
            ASTRepr::Pow(base, exp) if matches!(base.as_ref(), ASTRepr::Variable(0)) => {
                if let ASTRepr::Constant(k) = exp.as_ref() {
                    self.evaluate_power_sum(start, end, *k)
                } else {
                    self.evaluate_numerically(start, end, &ASTRepr::Pow(base, exp))
                }
            }
            
            // Fall back to numerical evaluation for complex expressions
            _ => self.evaluate_numerically(start, end, &expr),
        }
    }
    
    /// Helper method for numerical evaluation fallback
    fn evaluate_numerically(&self, start: i64, end: i64, expr: &ASTRepr<f64>) -> Result<f64> {
        let mut sum = 0.0;
        for i in start..=end {
            let value = self.eval_with_vars(expr, &[i as f64]);
            sum += value;
        }
        Ok(sum)
    }
    
    /// Helper method for evaluating power sums Σ(i^k)
    fn evaluate_power_sum(&self, _start: i64, end: i64, exponent: f64) -> Result<f64> {
        if exponent == 1.0 {
            // Σ(i) = n(n+1)/2
            let n = end as f64;
            Ok(n * (n + 1.0) / 2.0)
        } else if exponent == 2.0 {
            // Σ(i²) = n(n+1)(2n+1)/6
            let n = end as f64;
            Ok(n * (n + 1.0) * (2.0 * n + 1.0) / 6.0)
        } else {
            // Fall back to numerical evaluation for other powers
            let expr = ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)),
                Box::new(ASTRepr::Constant(exponent)),
            );
            self.evaluate_numerically(_start, end, &expr)
        }
    }
    
    /// Simple expression evaluation with variables
    fn eval_with_vars(&self, expr: &ASTRepr<f64>, vars: &[f64]) -> f64 {
        match expr {
            ASTRepr::Constant(c) => *c,
            ASTRepr::Variable(idx) => vars.get(*idx).copied().unwrap_or(0.0),
            ASTRepr::Add(left, right) => {
                self.eval_with_vars(left, vars) + self.eval_with_vars(right, vars)
            }
            ASTRepr::Sub(left, right) => {
                self.eval_with_vars(left, vars) - self.eval_with_vars(right, vars)
            }
            ASTRepr::Mul(left, right) => {
                self.eval_with_vars(left, vars) * self.eval_with_vars(right, vars)
            }
            ASTRepr::Div(left, right) => {
                self.eval_with_vars(left, vars) / self.eval_with_vars(right, vars)
            }
            ASTRepr::Pow(left, right) => {
                let base = self.eval_with_vars(left, vars);
                let exp = self.eval_with_vars(right, vars);
                base.powf(exp)
            }
            ASTRepr::Neg(inner) => -self.eval_with_vars(inner, vars),
            ASTRepr::Sqrt(inner) => self.eval_with_vars(inner, vars).sqrt(),
            ASTRepr::Sin(inner) => self.eval_with_vars(inner, vars).sin(),
            ASTRepr::Cos(inner) => self.eval_with_vars(inner, vars).cos(),
            ASTRepr::Exp(inner) => self.eval_with_vars(inner, vars).exp(),
            ASTRepr::Ln(inner) => self.eval_with_vars(inner, vars).ln(),
        }
    }
}

fn main() -> Result<()> {
    println!("🚀 Clean Functional Summation Optimization Demo");
    println!("==============================================");

    let optimizer = SummationOptimizer::new();

    // Test 1: Sum splitting - Σ(i + i²) for i=1..10
    println!("\n🎯 Test 1: Sum Splitting - Σ(i + i²) for i=1..10");
    let expr1 = ASTRepr::Add(
        Box::new(ASTRepr::Variable(0)),  // i
        Box::new(ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),  // i
            Box::new(ASTRepr::Constant(2.0)) // ²
        ))
    );
    
    let result1 = optimizer.optimize_summation(1, 10, expr1)?;
    let expected1 = 440.0; // Σ(i) + Σ(i²) = 55 + 385 = 440
    let error1 = (result1 - expected1).abs();
    
    println!("   Expected: {}, Actual: {}, Error: {:.2e}", expected1, result1, error1);
    println!("   ✅ Sum splitting optimization: {}", if error1 < 1e-10 { "PERFECT" } else { "FAILED" });

    // Test 2: Constant factor distribution - Σ(5 * i) for i=1..10  
    println!("\n🎯 Test 2: Factor Extraction - Σ(5 * i) for i=1..10");
    let expr2 = ASTRepr::Mul(
        Box::new(ASTRepr::Constant(5.0)),  // 5
        Box::new(ASTRepr::Variable(0))     // i
    );
    
    let result2 = optimizer.optimize_summation(1, 10, expr2)?;
    let expected2 = 275.0; // 5 * Σ(i) = 5 * 55 = 275
    let error2 = (result2 - expected2).abs();
    
    println!("   Expected: {}, Actual: {}, Error: {:.2e}", expected2, result2, error2);
    println!("   ✅ Factor extraction optimization: {}", if error2 < 1e-10 { "PERFECT" } else { "FAILED" });

    // Test 3: Combined optimizations - Σ(3*i + 2*i²) for i=1..5
    println!("\n🎯 Test 3: Combined Optimizations - Σ(3*i + 2*i²) for i=1..5");
    let expr3 = ASTRepr::Add(
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(3.0)),  // 3
            Box::new(ASTRepr::Variable(0))     // i
        )),
        Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(2.0)),  // 2
            Box::new(ASTRepr::Pow(
                Box::new(ASTRepr::Variable(0)), // i
                Box::new(ASTRepr::Constant(2.0)) // ²
            ))
        ))
    );
    
    let result3 = optimizer.optimize_summation(1, 5, expr3)?;
    let expected3 = 155.0; // 3*Σ(i) + 2*Σ(i²) = 3*15 + 2*55 = 45 + 110 = 155
    let error3 = (result3 - expected3).abs();
    
    println!("   Expected: {}, Actual: {}, Error: {:.2e}", expected3, result3, error3);
    println!("   ✅ Combined optimizations: {}", if error3 < 1e-10 { "PERFECT" } else { "FAILED" });

    println!("\n🎉 Summary:");
    println!("   - Sum splitting: {} accuracy", if error1 < 1e-10 { "Perfect" } else { "Failed" });
    println!("   - Factor extraction: {} accuracy", if error2 < 1e-10 { "Perfect" } else { "Failed" });
    println!("   - Combined: {} accuracy", if error3 < 1e-10 { "Perfect" } else { "Failed" });
    
    if error1 < 1e-10 && error2 < 1e-10 && error3 < 1e-10 {
        println!("   🎯 ALL OPTIMIZATIONS WORKING PERFECTLY!");
    } else {
        println!("   ❌ Some optimizations need fixes");
    }

    Ok(())
} 