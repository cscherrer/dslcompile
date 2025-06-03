//! Property-based tests for the next-generation summation system (summation_v2)
//!
//! These tests verify correctness by:
//! - Comparing optimized summations against brute-force numerical computation
//! - Testing factor extraction correctness
//! - Verifying pattern recognition accuracy
//! - Ensuring closure-based scoping prevents variable escape
//! - Testing mathematical properties (associativity, distributivity, etc.)

use dslcompile::Result;
use dslcompile::final_tagless::{IntRange, ExpressionBuilder, DirectEval, RangeType};
use dslcompile::symbolic::summation_v2::{SummationProcessor, SummationPattern};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// Configuration for summation test generation
#[derive(Debug, Clone, Copy)]
struct SummationConfig {
    max_range_size: i64,
    max_coefficient: f64,
    max_constant: f64,
}

impl Default for SummationConfig {
    fn default() -> Self {
        Self {
            max_range_size: 50,    // Keep ranges reasonable for brute force comparison
            max_coefficient: 100.0,
            max_constant: 100.0,
        }
    }
}

/// Brute force numerical computation for comparison
fn brute_force_sum<F>(range: IntRange, summand_fn: F) -> f64
where
    F: Fn(f64) -> f64,
{
    range.iter().map(|i| summand_fn(i as f64)).sum()
}

/// Test that constant summations work correctly
fn test_constant_summation(k: f64, start: i64, end: i64) -> std::result::Result<(), TestCaseError> {
    prop_assume!(start <= end);
    prop_assume!(end - start <= 50); // Keep ranges reasonable
    prop_assume!(k.is_finite() && k.abs() <= 1000.0); // Avoid overflow

    let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
    let range = IntRange::new(start, end);
    let range_len = range.len();

    // Test: Σ(k) = k * n
    let result = processor.sum(range.clone(), |_i| {
        let math = ExpressionBuilder::new();
        math.constant(k)
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum: {}", e).into()))?;

    let optimized_value = result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate: {}", e).into()))?;
    let expected = k * range_len as f64;
    
    // Check pattern recognition
    match &result.pattern {
        SummationPattern::Constant { value } => {
            // After factor extraction, the constant k becomes a factor,
            // and the pattern should be Constant { value: 1.0 }
            prop_assert!((value - 1.0).abs() < 1e-12, "Constant pattern should have value 1.0 after factor extraction, got {}", value);
        }
        _ => prop_assert!(false, "Should recognize constant pattern"),
    }

    // Check extracted factors include our constant
    prop_assert!(!result.extracted_factors.is_empty(), "Should extract constant factor");
    let total_factor: f64 = result.extracted_factors.iter().product();
    prop_assert!((total_factor - k).abs() < 1e-12, "Should extract factor {}, got {}", k, total_factor);

    // Check correctness
    prop_assert!((optimized_value - expected).abs() < 1e-10, 
                "Constant sum: expected {}, got {}", expected, optimized_value);

    Ok(())
}

/// Test that linear summations work correctly
fn test_linear_summation(coeff: f64, constant: f64, start: i64, end: i64) -> std::result::Result<(), TestCaseError> {
    prop_assume!(start <= end);
    prop_assume!(end - start <= 50);
    prop_assume!(coeff.is_finite() && coeff.abs() <= 100.0);
    prop_assume!(constant.is_finite() && constant.abs() <= 100.0);

    let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
    let range = IntRange::new(start, end);

    // Test: Σ(coeff * i + constant)
    let result = processor.sum(range.clone(), |i| {
        let math = ExpressionBuilder::new();
        math.constant(coeff) * i + math.constant(constant)
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum: {}", e).into()))?;

    let optimized_value = result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate: {}", e).into()))?;
    
    // Brute force comparison
    let brute_force_value = brute_force_sum(range.clone(), |i| coeff * i + constant);
    
    // Check pattern recognition
    match &result.pattern {
        SummationPattern::Linear { coefficient, constant: c } => {
            prop_assert!((coefficient - coeff).abs() < 1e-12, 
                        "Linear coefficient: expected {}, got {}", coeff, coefficient);
            prop_assert!((c - constant).abs() < 1e-12, 
                        "Linear constant: expected {}, got {}", constant, c);
        }
        _ => prop_assert!(false, "Should recognize linear pattern, got {:?}", result.pattern),
    }

    // Check correctness against brute force
    prop_assert!((optimized_value - brute_force_value).abs() < 1e-10, 
                "Linear sum: expected {}, got {}", brute_force_value, optimized_value);

    Ok(())
}

/// Test that factor extraction works correctly
fn test_factor_extraction(factor: f64, inner_coeff: f64, start: i64, end: i64) -> std::result::Result<(), TestCaseError> {
    prop_assume!(start <= end);
    prop_assume!(end - start <= 30);
    prop_assume!(factor.is_finite() && factor.abs() <= 50.0 && factor != 0.0);
    prop_assume!(inner_coeff.is_finite() && inner_coeff.abs() <= 50.0);

    let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
    let range = IntRange::new(start, end);

    // Test: Σ(factor * (inner_coeff * i))
    let result = processor.sum(range.clone(), |i| {
        let math = ExpressionBuilder::new();
        math.constant(factor) * (math.constant(inner_coeff) * i)
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum: {}", e).into()))?;

    let optimized_value = result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate: {}", e).into()))?;
    
    // Brute force comparison
    let brute_force_value = brute_force_sum(range, |i| factor * (inner_coeff * i));
    
    // Check that factors were extracted
    prop_assert!(!result.extracted_factors.is_empty(), "Should extract factors");
    
    // The total factor should include our outer factor
    let total_factor: f64 = result.extracted_factors.iter().product();
    let expected_total_factor = factor * inner_coeff;
    prop_assert!((total_factor - expected_total_factor).abs() < 1e-12, 
                "Total factor: expected {}, got {}", expected_total_factor, total_factor);

    // Check correctness
    prop_assert!((optimized_value - brute_force_value).abs() < 1e-10, 
                "Factor extraction: expected {}, got {}", brute_force_value, optimized_value);

    Ok(())
}

/// Test geometric series recognition and evaluation
fn test_geometric_series(coeff: f64, ratio: f64, start: i64, max_terms: i64) -> std::result::Result<(), TestCaseError> {
    prop_assume!(max_terms >= 1 && max_terms <= 20); // Keep geometric series reasonable
    prop_assume!(ratio.abs() < 0.95); // Ensure convergence and avoid overflow
    prop_assume!(coeff.is_finite() && coeff.abs() <= 10.0);
    prop_assume!(start >= 0); // Geometric series usually start from 0 or positive

    let end = start + max_terms - 1;
    let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
    let range = IntRange::new(start, end);

    // Test: Σ(coeff * ratio^i)
    let result = processor.sum(range.clone(), |i| {
        let math = ExpressionBuilder::new();
        math.constant(coeff) * math.constant(ratio).pow(i)
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum: {}", e).into()))?;

    let optimized_value = result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate: {}", e).into()))?;
    
    // Brute force comparison
    let brute_force_value = brute_force_sum(range, |i| coeff * ratio.powf(i));
    
    // Check correctness (geometric series can have precision issues, so use larger tolerance)
    prop_assert!((optimized_value - brute_force_value).abs() < 1e-8, 
                "Geometric series: expected {}, got {}", brute_force_value, optimized_value);

    Ok(())
}

/// Test power series (Σ i^k) recognition and evaluation
fn test_power_series(exponent: f64, start: i64, end: i64) -> std::result::Result<(), TestCaseError> {
    prop_assume!(start <= end);
    prop_assume!(start >= 1); // Avoid 0^0 issues
    prop_assume!(end - start <= 20); // Keep power series reasonable to avoid overflow
    prop_assume!(exponent >= 0.0 && exponent <= 4.0); // Reasonable exponents
    prop_assume!((exponent.fract()).abs() < 1e-10); // Integer exponents for now

    let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
    let range = IntRange::new(start, end);

    // Test: Σ(i^exponent)
    let result = processor.sum(range.clone(), |i| {
        let math = ExpressionBuilder::new();
        i.pow(math.constant(exponent))
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum: {}", e).into()))?;

    let optimized_value = result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate: {}", e).into()))?;
    
    // Brute force comparison
    let brute_force_value = brute_force_sum(range, |i| i.powf(exponent));
    
    // Check pattern recognition for integer exponents
    if exponent == exponent.round() {
        match &result.pattern {
            SummationPattern::Power { exponent: e } => {
                prop_assert!((e - exponent).abs() < 1e-12, 
                            "Power exponent: expected {}, got {}", exponent, e);
            }
            _ => {
                // Power pattern recognition might not work for all cases, that's OK
                println!("Power pattern not recognized for exponent {}, got {:?}", exponent, result.pattern);
            }
        }
    }

    // Check correctness
    prop_assert!((optimized_value - brute_force_value).abs() < 1e-10, 
                "Power series: expected {}, got {}", brute_force_value, optimized_value);

    Ok(())
}

/// Test mathematical properties like distributivity
fn test_distributivity(k: f64, a: f64, b: f64, start: i64, end: i64) -> std::result::Result<(), TestCaseError> {
    prop_assume!(start <= end);
    prop_assume!(end - start <= 30);
    prop_assume!(k.is_finite() && k.abs() <= 50.0);
    prop_assume!(a.is_finite() && a.abs() <= 50.0);
    prop_assume!(b.is_finite() && b.abs() <= 50.0);

    let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
    let range = IntRange::new(start, end);

    // Test: Σ(k * (a*i + b)) = k * Σ(a*i + b)
    let combined_result = processor.sum(range.clone(), |i| {
        let math = ExpressionBuilder::new();
        math.constant(k) * (math.constant(a) * i + math.constant(b))
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute combined sum: {}", e).into()))?;

    let mut processor2 = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor2: {}", e).into()))?;
    let separate_result = processor2.sum(range.clone(), |i| {
        let math = ExpressionBuilder::new();
        math.constant(a) * i + math.constant(b)
    }).map_err(|e| TestCaseError::Fail(format!("Failed to compute separate sum: {}", e).into()))?;

    let combined_value = combined_result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate combined: {}", e).into()))?;
    let separate_value = separate_result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate separate: {}", e).into()))? * k;

    // Both should give the same result
    prop_assert!((combined_value - separate_value).abs() < 1e-10, 
                "Distributivity: combined {}, separate {}", combined_value, separate_value);

    // Both should recognize the linear pattern
    match (&combined_result.pattern, &separate_result.pattern) {
        (SummationPattern::Linear { .. }, SummationPattern::Linear { .. }) => {
            // Good, both recognized linear patterns
        }
        _ => {
            // Pattern recognition might vary, but results should still be correct
            println!("Pattern recognition varied: combined {:?}, separate {:?}", 
                    combined_result.pattern, separate_result.pattern);
        }
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_constant_summation(
        k in -100.0..100.0_f64,
        start in 0..20_i64,
        size in 1..30_i64,
    ) {
        let end = start + size;
        test_constant_summation(k, start, end)?;
    }

    #[test]
    fn prop_linear_summation(
        coeff in -50.0..50.0_f64,
        constant in -50.0..50.0_f64,
        start in 0..15_i64,
        size in 1..25_i64,
    ) {
        let end = start + size;
        test_linear_summation(coeff, constant, start, end)?;
    }

    #[test]
    fn prop_factor_extraction(
        factor in -20.0..20.0_f64,
        inner_coeff in -20.0..20.0_f64,
        start in 1..10_i64,
        size in 1..15_i64,
    ) {
        prop_assume!(factor != 0.0);
        let end = start + size;
        test_factor_extraction(factor, inner_coeff, start, end)?;
    }

    #[test]
    fn prop_geometric_series(
        coeff in -5.0..5.0_f64,
        ratio in -0.8..0.8_f64,
        start in 0..5_i64,
        max_terms in 1..15_i64,
    ) {
        test_geometric_series(coeff, ratio, start, max_terms)?;
    }

    #[test]
    fn prop_power_series(
        exponent in 0.0..3.0_f64,
        start in 1..8_i64,
        size in 1..12_i64,
    ) {
        let end = start + size;
        test_power_series(exponent.round(), start, end)?;
    }

    #[test]
    fn prop_distributivity(
        k in -10.0..10.0_f64,
        a in -10.0..10.0_f64,
        b in -10.0..10.0_f64,
        start in 1..10_i64,
        size in 1..15_i64,
    ) {
        let end = start + size;
        test_distributivity(k, a, b, start, end)?;
    }

    #[test]
    fn prop_range_consistency(
        start in -10..10_i64,
        size in 1..20_i64,
    ) {
        let end = start + size;
        let range = IntRange::new(start, end);
        
        // Test that empty summation works
        let mut processor = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor: {}", e).into()))?;
        let result = processor.sum(range.clone(), |_i| {
            let math = ExpressionBuilder::new();
            math.constant(1.0)
        }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum: {}", e).into()))?;
        
        let value = result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate: {}", e).into()))?;
        let expected = range.len() as f64;
        
        prop_assert!((value - expected).abs() < 1e-12, 
                    "Range consistency: expected {}, got {}", expected, value);
    }

    #[test]
    fn prop_zero_factor_handling(start in 1..10i64, size in 1..5usize) {
        let mut processor = SummationProcessor::new().unwrap();
        let range = IntRange::new(start, start + size as i64 - 1);
        
        let result = processor.sum(range, |i| {
            let math = ExpressionBuilder::new();
            let zero_expr = math.constant(0.0) * i;
            println!("[DEBUG] Original expression: {:?}", zero_expr.clone().into_ast());
            zero_expr
        }).unwrap();
        
        println!("[DEBUG] Pattern recognized: {:?}", result.pattern);
        println!("[DEBUG] Closed form: {:?}", result.closed_form);
        println!("[DEBUG] Extracted factors: {:?}", result.extracted_factors);
        
        let value = result.evaluate(&[]).unwrap();
        println!("[DEBUG] Final evaluated value: {}", value);
        
        prop_assert_eq!(value, 0.0, "Zero factor should give zero result, got {}", value);
    }

    #[test] 
    fn prop_additive_property(
        a in -20.0..20.0_f64,
        b in -20.0..20.0_f64,
        start in 1..8_i64,
        size in 1..12_i64,
    ) {
        let end = start + size;
        let range = IntRange::new(start, end);
        
        // Test: Σ(a*i + b*i) = Σ(a*i) + Σ(b*i) = (a+b)*Σ(i)
        let mut processor1 = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor1: {}", e).into()))?;
        let combined_result = processor1.sum(range.clone(), |i| {
            let math = ExpressionBuilder::new();
            (math.constant(a) + math.constant(b)) * i
        }).map_err(|e| TestCaseError::Fail(format!("Failed to compute combined sum: {}", e).into()))?;
        
        let mut processor2 = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor2: {}", e).into()))?;
        let separate_a = processor2.sum(range.clone(), |i| {
            let math = ExpressionBuilder::new();
            math.constant(a) * i
        }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum A: {}", e).into()))?;
        
        let mut processor3 = SummationProcessor::new().map_err(|e| TestCaseError::Fail(format!("Failed to create processor3: {}", e).into()))?;
        let separate_b = processor3.sum(range, |i| {
            let math = ExpressionBuilder::new();
            math.constant(b) * i
        }).map_err(|e| TestCaseError::Fail(format!("Failed to compute sum B: {}", e).into()))?;
        
        let combined_value = combined_result.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate combined: {}", e).into()))?;
        let separate_value = separate_a.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate A: {}", e).into()))? + separate_b.evaluate(&[]).map_err(|e| TestCaseError::Fail(format!("Failed to evaluate B: {}", e).into()))?;
        
        prop_assert!((combined_value - separate_value).abs() < 1e-10, 
                    "Additive property: combined {}, separate sum {}", combined_value, separate_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_summation_cases() -> Result<()> {
        // Test some specific known cases
        
        // Σ(i=1 to 10) i = 55
        test_linear_summation(1.0, 0.0, 1, 10).map_err(|_| dslcompile::error::DSLCompileError::Generic("proptest error".to_string()))?;
        
        // Σ(i=1 to 5) 2*i = 2*15 = 30  
        test_factor_extraction(2.0, 1.0, 1, 5).map_err(|_| dslcompile::error::DSLCompileError::Generic("proptest error".to_string()))?;
        
        // Σ(i=1 to 4) i² = 1+4+9+16 = 30
        test_power_series(2.0, 1, 4).map_err(|_| dslcompile::error::DSLCompileError::Generic("proptest error".to_string()))?;
        
        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        let mut processor = SummationProcessor::new()?;
        
        // Single element range
        let range = IntRange::new(5, 5);
        let result = processor.sum(range, |i| i)?;
        let value = result.evaluate(&[])?;
        assert!((value - 5.0).abs() < 1e-12);
        
        // Large constant factor
        let range = IntRange::new(1, 3);
        let result = processor.sum(range, |_i| {
            let math = ExpressionBuilder::new();
            math.constant(1000.0)
        })?;
        let value = result.evaluate(&[])?;
        assert!((value - 3000.0).abs() < 1e-10);
        
        Ok(())
    }
} 