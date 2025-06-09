//! Symbolic Summation Demo: Idiomatic Rust Code Generation
//!
//! This demonstrates the approach for truly symbolic summation that generates
//! idiomatic Rust iteration patterns instead of pre-computed constants.
//!
//! Key Features:
//! - Mathematical range summation: (1..=n).map(|i| `expr(i)).sum()`
//! - Data array summation: data.iter().map(|&x| `expr(x)).sum()`
//! - Static context compatibility
//! - Runtime data binding with function parameters

use dslcompile::prelude::*;

/// Represents a symbolic summation for code generation
#[derive(Debug, Clone)]
pub struct SymbolicSum {
    /// Range type for the summation
    pub range: SumRangeSpec,
    /// Body expression as a string (for demo)
    pub body_expr: String,
    /// Iterator variable name
    pub iter_var: String,
}

/// Range specification for different summation types
#[derive(Debug, Clone)]
pub enum SumRangeSpec {
    /// Mathematical range: 1..=n
    MathematicalRange { start: String, end: String },
    /// Data parameter: function parameter that holds array data
    DataParameter { param_name: String },
    /// Static data: compile-time known values
    StaticData { values: Vec<f64> },
}

impl SymbolicSum {
    /// Generate idiomatic Rust code for this summation
    #[must_use]
    pub fn generate_rust_code(&self) -> String {
        match &self.range {
            SumRangeSpec::MathematicalRange { start, end } => {
                format!(
                    "({start}..={end}).map(|{iter_var}| {body_expr}).sum::<f64>()",
                    start = start,
                    end = end,
                    iter_var = self.iter_var,
                    body_expr = self.body_expr
                )
            }
            SumRangeSpec::DataParameter { param_name } => {
                format!(
                    "{param_name}.iter().map(|&{iter_var}| {body_expr}).sum::<f64>()",
                    param_name = param_name,
                    iter_var = self.iter_var,
                    body_expr = self.body_expr
                )
            }
            SumRangeSpec::StaticData { values } => {
                let values_str = format!(
                    "[{}]",
                    values
                        .iter()
                        .map(std::string::ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                format!(
                    "{values_str}.iter().map(|&{iter_var}| {body_expr}).sum::<f64>()",
                    values_str = values_str,
                    iter_var = self.iter_var,
                    body_expr = self.body_expr
                )
            }
        }
    }

    /// Generate optimized Rust code with pattern recognition
    #[must_use]
    pub fn generate_optimized_rust_code(&self) -> String {
        // Pattern recognition for common cases
        match &self.range {
            SumRangeSpec::MathematicalRange { start, end } => {
                if self.body_expr == self.iter_var {
                    // Σ(i) = n*(n+1)/2 optimization
                    format!("((({end}) * (({end}) + 1) - (({start}) - 1) * ({start})) / 2.0)")
                } else if self.body_expr.starts_with("2.0 * ")
                    || self
                        .body_expr
                        .starts_with(&format!("2.0 * {}", self.iter_var))
                {
                    // Σ(2*i) = 2*Σ(i) optimization
                    format!("2.0 * ((({end}) * (({end}) + 1) - (({start}) - 1) * ({start})) / 2.0)")
                } else {
                    // Fall back to iterator pattern
                    self.generate_rust_code()
                }
            }
            SumRangeSpec::DataParameter { param_name } => {
                if self.body_expr == self.iter_var {
                    // Σ(x) = data.iter().sum() optimization
                    format!("{param_name}.iter().sum::<f64>()")
                } else if let Some(factor) =
                    extract_constant_factor(&self.body_expr, &self.iter_var)
                {
                    // Σ(c*x) = c * data.iter().sum() optimization
                    format!("{factor} * {param_name}.iter().sum::<f64>()")
                } else {
                    // General iterator pattern
                    self.generate_rust_code()
                }
            }
            SumRangeSpec::StaticData { .. } => {
                // For static data, always use iterator pattern for now
                self.generate_rust_code()
            }
        }
    }
}

/// Extract constant factor from expressions like "2.5 * x"
fn extract_constant_factor(expr: &str, var: &str) -> Option<f64> {
    if let Some(pos) = expr.find(&format!(" * {var}")) {
        let factor_str = &expr[..pos];
        factor_str.trim().parse::<f64>().ok()
    } else if expr.ends_with(&format!(" * {var}")) {
        let factor_str = &expr[..expr.len() - var.len() - 3];
        factor_str.trim().parse::<f64>().ok()
    } else {
        None
    }
}

/// Generate a complete function with symbolic summations
#[must_use]
pub fn generate_function_with_summations(
    name: &str,
    summations: &[SymbolicSum],
    params: &[(&str, &str)], // (name, type) pairs
) -> String {
    let param_list: Vec<String> = params
        .iter()
        .map(|(name, typ)| format!("{name}: {typ}"))
        .collect();

    let param_str = param_list.join(", ");

    let mut body = String::new();
    for (i, sum) in summations.iter().enumerate() {
        if i > 0 {
            body.push_str(" + ");
        }
        body.push_str(&sum.generate_optimized_rust_code());
    }

    format!(
        r"
#[inline]
pub fn {name}({param_str}) -> f64 {{
    {body}
}}
"
    )
}

fn main() -> Result<()> {
    println!("🔧 Symbolic Summation Demo: Idiomatic Rust Code Generation");
    println!("==========================================================\n");

    // Example 1: Mathematical Range Summation
    println!("📐 Mathematical Range Summation");
    println!("-------------------------------");

    let math_sum = SymbolicSum {
        range: SumRangeSpec::MathematicalRange {
            start: "1".to_string(),
            end: "n".to_string(),
        },
        body_expr: "2.0 * i".to_string(),
        iter_var: "i".to_string(),
    };

    println!("Expression: Σ(2*i) for i = 1 to n");
    println!("Generated code: {}", math_sum.generate_rust_code());
    println!(
        "Optimized code: {}",
        math_sum.generate_optimized_rust_code()
    );
    println!();

    // Example 2: Data Array Summation
    println!("📊 Data Array Summation");
    println!("-----------------------");

    let data_sum = SymbolicSum {
        range: SumRangeSpec::DataParameter {
            param_name: "data".to_string(),
        },
        body_expr: "x * x".to_string(),
        iter_var: "x".to_string(),
    };

    println!("Expression: Σ(x²) for x in data");
    println!("Generated code: {}", data_sum.generate_rust_code());
    println!(
        "Optimized code: {}",
        data_sum.generate_optimized_rust_code()
    );
    println!();

    // Example 3: Static Data Summation
    println!("📋 Static Data Summation");
    println!("------------------------");

    let static_sum = SymbolicSum {
        range: SumRangeSpec::StaticData {
            values: hlist![1.0, 2.0, 3.0, 4.0, 5.0],
        },
        body_expr: "x + 1.0".to_string(),
        iter_var: "x".to_string(),
    };

    println!("Expression: Σ(x + 1) for x in [1, 2, 3, 4, 5]");
    println!("Generated code: {}", static_sum.generate_rust_code());
    println!();

    // Example 4: Complete Function Generation
    println!("🏗️  Complete Function Generation");
    println!("--------------------------------");

    let gaussian_log_density = SymbolicSum {
        range: SumRangeSpec::DataParameter {
            param_name: "data".to_string(),
        },
        body_expr: "-0.5 * ((x - mu) / sigma).powi(2)".to_string(),
        iter_var: "x".to_string(),
    };

    let constant_term = SymbolicSum {
        range: SumRangeSpec::MathematicalRange {
            start: "1".to_string(),
            end: "data.len() as i64".to_string(),
        },
        body_expr: "-0.5 * (2.0 * std::f64::consts::PI * sigma * sigma).ln()".to_string(),
        iter_var: "_i".to_string(),
    };

    let function_code = generate_function_with_summations(
        "gaussian_log_density",
        &[gaussian_log_density, constant_term],
        &[("data", "&[f64]"), ("mu", "f64"), ("sigma", "f64")],
    );

    println!("Generated Gaussian log-density function:");
    println!("{function_code}");

    // Example 5: Demonstration of Performance Benefits
    println!("⚡ Performance Comparison");
    println!("------------------------");

    println!("Naive approach (evaluates at build time):");
    println!("  let sum = 0.0; for i in 1..=n {{ sum += 2.0 * i; }}");
    println!("  Build time: O(n), Eval time: O(1) - returns constant");
    println!();

    println!("Symbolic approach (generates idiomatic iteration):");
    println!("  (1..=n).map(|i| 2.0 * i).sum::<f64>()");
    println!("  Build time: O(1), Eval time: O(n) - true iteration");
    println!();

    println!("Optimized symbolic approach (pattern recognition):");
    println!("  n * (n + 1) // Closed-form when possible");
    println!("  Build time: O(1), Eval time: O(1) - best of both worlds");
    println!();

    // Example 6: Integration with Existing Context
    println!("🔗 Integration with DynamicContext");
    println!("----------------------------------");

    let math = DynamicContext::new();
    println!("Current sum() method behavior:");
    println!("  - Evaluates at build time");
    println!("  - Returns pre-computed constant");
    println!("  - Cannot handle changing data");
    println!();

    println!("Proposed sum_symbolic() method:");
    println!("  - Creates AST node with Sum variant");
    println!("  - Generates iteration code during compilation");
    println!("  - Supports runtime data binding");
    println!("  - Works with static contexts too");
    println!();

    println!("✅ Demo completed successfully!");
    println!("\nThis demonstrates the approach for truly symbolic summation");
    println!("that generates idiomatic, composable, performant Rust code!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mathematical_range_generation() {
        let sum = SymbolicSum {
            range: SumRangeSpec::MathematicalRange {
                start: "1".to_string(),
                end: "10".to_string(),
            },
            body_expr: "i".to_string(),
            iter_var: "i".to_string(),
        };

        let code = sum.generate_rust_code();
        assert!(code.contains("(1..=10).map(|i| i).sum::<f64>()"));
    }

    #[test]
    fn test_data_parameter_generation() {
        let sum = SymbolicSum {
            range: SumRangeSpec::DataParameter {
                param_name: "values".to_string(),
            },
            body_expr: "x * 2.0".to_string(),
            iter_var: "x".to_string(),
        };

        let code = sum.generate_rust_code();
        assert!(code.contains("values.iter().map(|&x| x * 2.0).sum::<f64>()"));
    }

    #[test]
    fn test_optimization_sum_of_indices() {
        let sum = SymbolicSum {
            range: SumRangeSpec::MathematicalRange {
                start: "1".to_string(),
                end: "n".to_string(),
            },
            body_expr: "i".to_string(),
            iter_var: "i".to_string(),
        };

        let optimized = sum.generate_optimized_rust_code();
        assert!(optimized.contains("(n) * ((n) + 1)"));
    }

    #[test]
    fn test_optimization_constant_factor() {
        let sum = SymbolicSum {
            range: SumRangeSpec::DataParameter {
                param_name: "data".to_string(),
            },
            body_expr: "x".to_string(),
            iter_var: "x".to_string(),
        };

        let optimized = sum.generate_optimized_rust_code();
        assert!(optimized.contains("data.iter().sum::<f64>()"));
    }

    #[test]
    fn test_constant_factor_extraction() {
        assert_eq!(extract_constant_factor("2.5 * x", "x"), Some(2.5));
        assert_eq!(extract_constant_factor("x * 3.0", "x"), None); // Currently doesn't handle this case
        assert_eq!(extract_constant_factor("x + 1.0", "x"), None);
    }

    #[test]
    fn test_function_generation() {
        let sum = SymbolicSum {
            range: SumRangeSpec::DataParameter {
                param_name: "data".to_string(),
            },
            body_expr: "x".to_string(),
            iter_var: "x".to_string(),
        };

        let function =
            generate_function_with_summations("test_func", &[sum], &[("data", "&[f64]")]);

        assert!(function.contains("pub fn test_func(data: &[f64]) -> f64"));
        assert!(function.contains("data.iter().sum::<f64>()"));
    }
}
