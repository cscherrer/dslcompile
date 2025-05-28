//! Power Optimization Utilities
//!
//! This module provides common utilities for optimizing integer power operations
//! across different compilation backends (Rust codegen, Cranelift, etc.).
//!
//! # Features
//!
//! - **Integer Power Detection**: Identify when exponents are small integers
//! - **Optimized Power Patterns**: Generate efficient code for common power values
//! - **Backend-Agnostic**: Provides both string-based and IR-based optimizations

use num_traits::Float;

/// Configuration for power optimizations
#[derive(Debug, Clone)]
pub struct PowerOptConfig {
    /// Maximum integer exponent to optimize (beyond this, use generic powf/powi)
    pub max_optimized_exponent: i32,
    /// Whether to use unsafe optimizations (e.g., reusing intermediate values)
    pub unsafe_optimizations: bool,
    /// Whether to optimize negative exponents
    pub optimize_negative_exponents: bool,
}

impl Default for PowerOptConfig {
    fn default() -> Self {
        Self {
            max_optimized_exponent: 10,
            unsafe_optimizations: false,
            optimize_negative_exponents: true,
        }
    }
}

/// Try to convert a floating-point value to an integer for power optimization
pub fn try_convert_to_integer<T: Float>(value: T, tolerance: Option<f64>) -> Option<i32> {
    let float_val = value.to_f64().unwrap_or(0.0);
    let tol = tolerance.unwrap_or(1e-12);

    if float_val.fract().abs() < tol && float_val.abs() <= 100.0 {
        Some(float_val.round() as i32)
    } else {
        None
    }
}

/// Generate optimized string-based code for integer powers (for Rust codegen)
#[must_use]
pub fn generate_integer_power_string(
    base_expr: &str,
    exponent: i32,
    config: &PowerOptConfig,
) -> String {
    if exponent.abs() > config.max_optimized_exponent {
        return format!("{base_expr}.powi({exponent})");
    }

    match exponent {
        0 => "1.0".to_string(),
        1 => base_expr.to_string(),
        -1 if config.optimize_negative_exponents => format!("1.0 / {base_expr}"),
        2 => format!("{base_expr} * {base_expr}"),
        -2 if config.optimize_negative_exponents => {
            format!("1.0 / ({base_expr} * {base_expr})")
        }
        3 => format!("{base_expr} * {base_expr} * {base_expr}"),
        4 => {
            if config.unsafe_optimizations {
                format!("{{ let temp = {base_expr} * {base_expr}; temp * temp }}")
            } else {
                format!("{base_expr} * {base_expr} * {base_expr} * {base_expr}")
            }
        }
        5 => format!(
            "{{ let temp = {base_expr} * {base_expr}; temp * temp * {base_expr} }}"
        ),
        6 => format!(
            "{{ let temp = {base_expr} * {base_expr} * {base_expr}; temp * temp }}"
        ),
        8 => format!(
            "{{ let temp2 = {base_expr} * {base_expr}; let temp4 = temp2 * temp2; temp4 * temp4 }}"
        ),
        10 => format!(
            "{{ let temp5 = {base_expr} * {base_expr} * {base_expr} * {base_expr} * {base_expr}; temp5 * temp5 }}"
        ),
        exp if exp < 0 && config.optimize_negative_exponents => {
            format!(
                "1.0 / ({})",
                generate_integer_power_string(base_expr, -exp, config)
            )
        }
        _ => format!("{base_expr}.powi({exponent})"),
    }
}

/// Power optimization strategy for different backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerStrategy {
    /// Use multiplication chains for small exponents
    MultiplicationChain,
    /// Use repeated squaring for larger exponents
    RepeatedSquaring,
    /// Use the generic power function
    Generic,
}

/// Determine the best power strategy for a given exponent
#[must_use]
pub fn determine_power_strategy(exponent: i32, config: &PowerOptConfig) -> PowerStrategy {
    let abs_exp = exponent.abs();

    if abs_exp <= 6 {
        PowerStrategy::MultiplicationChain
    } else if abs_exp <= config.max_optimized_exponent && (abs_exp & (abs_exp - 1)) == 0 {
        // Power of 2, use repeated squaring
        PowerStrategy::RepeatedSquaring
    } else {
        PowerStrategy::Generic
    }
}

/// Generate repeated squaring pattern for powers of 2
#[must_use]
pub fn generate_repeated_squaring_string(base_expr: &str, exponent: i32) -> String {
    let abs_exp = exponent.abs();

    if abs_exp == 1 {
        return base_expr.to_string();
    }

    // For powers of 2, generate efficient repeated squaring
    let mut result = format!("{{ let mut temp = {base_expr}; ");
    let mut current_power = 1;

    while current_power < abs_exp {
        result.push_str("temp = temp * temp; ");
        current_power *= 2;
    }

    result.push_str("temp }");

    if exponent < 0 {
        format!("1.0 / ({result})")
    } else {
        result
    }
}

/// Metadata about power optimization decisions
#[derive(Debug, Clone)]
pub struct PowerOptimizationInfo {
    /// The strategy used for this power
    pub strategy: PowerStrategy,
    /// Whether the optimization was applied
    pub optimized: bool,
    /// The original exponent
    pub exponent: i32,
    /// Estimated performance improvement (relative to generic power)
    pub performance_gain: f64,
}

/// Analyze a power operation and return optimization information
#[must_use]
pub fn analyze_power_optimization(exponent: i32, config: &PowerOptConfig) -> PowerOptimizationInfo {
    let strategy = determine_power_strategy(exponent, config);
    let abs_exp = exponent.abs();

    let (optimized, performance_gain) = match strategy {
        PowerStrategy::MultiplicationChain => (abs_exp <= 6, 2.0 - (f64::from(abs_exp) * 0.1)),
        PowerStrategy::RepeatedSquaring => (abs_exp <= config.max_optimized_exponent, 1.5),
        PowerStrategy::Generic => (false, 1.0),
    };

    PowerOptimizationInfo {
        strategy,
        optimized,
        exponent,
        performance_gain: performance_gain.max(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_convert_to_integer() {
        assert_eq!(try_convert_to_integer(2.0_f64, None), Some(2));
        assert_eq!(try_convert_to_integer(2.1_f64, None), None);
        assert_eq!(try_convert_to_integer(-3.0_f64, None), Some(-3));
        assert_eq!(try_convert_to_integer(0.0_f64, None), Some(0));
    }

    #[test]
    fn test_generate_integer_power_string() {
        let config = PowerOptConfig::default();

        assert_eq!(generate_integer_power_string("x", 0, &config), "1.0");
        assert_eq!(generate_integer_power_string("x", 1, &config), "x");
        assert_eq!(generate_integer_power_string("x", 2, &config), "x * x");
        assert_eq!(generate_integer_power_string("x", -1, &config), "1.0 / x");

        // Test unsafe optimization
        let unsafe_config = PowerOptConfig {
            unsafe_optimizations: true,
            ..Default::default()
        };
        let result = generate_integer_power_string("x", 4, &unsafe_config);
        assert!(result.contains("let temp"));
    }

    #[test]
    fn test_determine_power_strategy() {
        let config = PowerOptConfig::default();

        assert_eq!(
            determine_power_strategy(2, &config),
            PowerStrategy::MultiplicationChain
        );
        assert_eq!(
            determine_power_strategy(8, &config),
            PowerStrategy::RepeatedSquaring
        );
        assert_eq!(
            determine_power_strategy(15, &config),
            PowerStrategy::Generic
        );
    }

    #[test]
    fn test_analyze_power_optimization() {
        let config = PowerOptConfig::default();

        let info = analyze_power_optimization(2, &config);
        assert!(info.optimized);
        assert_eq!(info.strategy, PowerStrategy::MultiplicationChain);
        assert!(info.performance_gain > 1.0);

        let info = analyze_power_optimization(100, &config);
        assert!(!info.optimized);
        assert_eq!(info.strategy, PowerStrategy::Generic);
    }

    #[test]
    fn test_repeated_squaring() {
        let result = generate_repeated_squaring_string("x", 8);
        assert!(result.contains("temp = temp * temp"));

        let result = generate_repeated_squaring_string("x", -8);
        assert!(result.starts_with("1.0 / ("));
    }
}
