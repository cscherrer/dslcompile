//! Example: Extending MathCompile from a Downstream Crate
//!
//! This example demonstrates how downstream crates can extend MathCompile with:
//! 1. Custom function categories
//! 2. Custom optimization rules
//! 3. Custom egglog rules
//! 4. Integration with the existing system

use mathcompile::ast::ASTRepr;
use mathcompile::ast::function_categories::{
    ExtensionRegistry, FunctionCategory, OptimizationRule,
};
use mathcompile::final_tagless::traits::NumericType;
use mathcompile::symbolic::{RuleSet, RuleSetBuilder, SymbolicEngine};
use num_traits::Float;

/// Example: Custom Statistics Functions
#[derive(Debug, Clone, PartialEq)]
pub enum StatisticsFunction<T: NumericType> {
    Mean(Vec<Box<ASTRepr<T>>>),
    Variance(Vec<Box<ASTRepr<T>>>),
    StandardDeviation(Vec<Box<ASTRepr<T>>>),
    Correlation(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    Covariance(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatisticsCategory<T: NumericType> {
    pub function: StatisticsFunction<T>,
}

impl<T> FunctionCategory<T> for StatisticsCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    fn to_egglog(&self) -> String {
        match &self.function {
            StatisticsFunction::Mean(args) => {
                let args_str = args
                    .iter()
                    .map(|arg| arg.to_egglog())
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("(Statistics (MeanFunc {}))", args_str)
            }
            StatisticsFunction::Variance(args) => {
                let args_str = args
                    .iter()
                    .map(|arg| arg.to_egglog())
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("(Statistics (VarianceFunc {}))", args_str)
            }
            StatisticsFunction::StandardDeviation(args) => {
                let args_str = args
                    .iter()
                    .map(|arg| arg.to_egglog())
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("(Statistics (StdDevFunc {}))", args_str)
            }
            StatisticsFunction::Correlation(x, y) => {
                format!(
                    "(Statistics (CorrelationFunc {} {}))",
                    x.to_egglog(),
                    y.to_egglog()
                )
            }
            StatisticsFunction::Covariance(x, y) => {
                format!(
                    "(Statistics (CovarianceFunc {} {}))",
                    x.to_egglog(),
                    y.to_egglog()
                )
            }
        }
    }

    fn apply_local_rules(&self, expr: &ASTRepr<T>) -> Option<ASTRepr<T>> {
        // Example: std_dev(x) = sqrt(variance(x))
        match expr {
            // This would be more complex in practice, but shows the concept
            _ => None,
        }
    }

    fn category_name(&self) -> &'static str {
        "statistics"
    }

    fn priority(&self) -> u32 {
        150 // Medium priority
    }
}

/// Example: Custom optimization rule for statistics
pub struct VarianceToStdDevRule;

impl<T> OptimizationRule<T> for VarianceToStdDevRule
where
    T: NumericType + Float,
{
    fn apply(&self, expr: &ASTRepr<T>) -> Option<ASTRepr<T>> {
        // Example: sqrt(variance(x)) -> std_dev(x)
        // This is a simplified example - real implementation would be more complex
        match expr {
            ASTRepr::Pow(base, exp) => {
                // Check if this is sqrt(variance(...))
                if let ASTRepr::Constant(exp_val) = exp.as_ref() {
                    if (*exp_val - T::from(0.5).unwrap()).abs() < T::from(1e-10).unwrap() {
                        // This is a square root - check if base is variance
                        // In practice, you'd need to check if base is a Statistics variant
                        // For now, just return None
                        return None;
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn rule_name(&self) -> &'static str {
        "variance_to_stddev"
    }

    fn priority(&self) -> u32 {
        120
    }

    fn is_applicable(&self, expr: &ASTRepr<T>) -> bool {
        matches!(expr, ASTRepr::Pow(_, _))
    }
}

/// Example: How a downstream crate would set up extensions
pub fn setup_statistics_extension() -> ExtensionRegistry<f64> {
    let mut registry = ExtensionRegistry::new();

    // Register custom optimization rules
    registry.register_rule(Box::new(VarianceToStdDevRule));

    // Register custom egglog rules for statistics
    let statistics_rules = r#"
        ;; Statistics function definitions
        (datatype Statistics
          (MeanFunc (List Math))
          (VarianceFunc (List Math))
          (StdDevFunc (List Math))
          (CorrelationFunc Math Math)
          (CovarianceFunc Math Math))

        ;; Statistics optimization rules
        ;; std_dev(x) = sqrt(variance(x))
        (rewrite (Statistics (StdDevFunc ?x)) (Sqrt (Statistics (VarianceFunc ?x))))
        
        ;; variance(x) = mean(x^2) - mean(x)^2
        (rewrite (Statistics (VarianceFunc ?x)) 
                 (Sub (Statistics (MeanFunc (Map (lambda ?y (Pow ?y (Num 2.0))) ?x)))
                      (Pow (Statistics (MeanFunc ?x)) (Num 2.0))))

        ;; correlation(x, y) = covariance(x, y) / (std_dev(x) * std_dev(y))
        (rewrite (Statistics (CorrelationFunc ?x ?y))
                 (Div (Statistics (CovarianceFunc ?x ?y))
                      (Mul (Statistics (StdDevFunc ?x)) (Statistics (StdDevFunc ?y)))))
    "#;

    registry.register_egglog_rules(statistics_rules.to_string());

    registry
}

/// Example: How to integrate with the symbolic engine
pub fn create_extended_symbolic_engine() -> Result<SymbolicEngine, Box<dyn std::error::Error>> {
    let mut engine = SymbolicEngine::with_default_rules()?;

    // Create custom rule set for statistics
    let stats_rules = RuleSetBuilder::new("statistics")
        .with_priority(150)
        .with_dependency("core_arithmetic")
        .with_content(
            r#"
            ;; Additional statistics rules that integrate with core math
            (rewrite (Add (Statistics (MeanFunc ?x)) (Statistics (MeanFunc ?y)))
                     (Statistics (MeanFunc (Concat ?x ?y))))
        "#,
        )
        .build();

    engine.add_rule_set(stats_rules);

    Ok(engine)
}

/// Example: Usage in downstream code
#[cfg(test)]
mod tests {
    use super::*;
    use mathcompile::ergonomics::MathBuilder;

    #[test]
    fn test_statistics_extension() {
        let registry = setup_statistics_extension();

        // Test that rules are registered
        assert!(!registry.get_egglog_rules().is_empty());

        // In practice, you'd create expressions using your custom functions
        // and apply the registry rules to optimize them
    }

    #[test]
    fn test_extended_symbolic_engine() {
        let engine = create_extended_symbolic_engine();
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert!(engine.has_rule_set("statistics"));
    }

    #[test]
    fn test_custom_optimization_rule() {
        let rule = VarianceToStdDevRule;

        // Create a test expression: x^0.5
        let mut math = MathBuilder::new();
        let x = math.var("x");
        let sqrt_expr = x.pow(&math.constant(0.5));

        // Test if rule is applicable
        assert!(rule.is_applicable(&sqrt_expr));

        // Test rule application (would return None in this simplified example)
        let result = rule.apply(&sqrt_expr);
        // In a real implementation, this might transform the expression
    }
}

/// Example: How downstream crates can define their own AST extensions
///
/// Note: This would require the main ASTRepr enum to have an Extension variant,
/// which could be added in a future version:
///
/// ```rust
/// pub enum ASTRepr<T: NumericType> {
///     // ... existing variants ...
///     Extension(Box<dyn FunctionCategory<T>>),
/// }
/// ```
///
/// This would allow complete extensibility without modifying the core enum.
pub trait ASTExtension<T: NumericType> {
    /// Convert to the core ASTRepr representation
    fn to_ast(&self) -> ASTRepr<T>;

    /// Get the egglog representation
    fn to_egglog(&self) -> String;

    /// Apply optimization rules
    fn optimize(&self) -> Self;
}

/// Example implementation for statistics
impl<T> ASTExtension<T> for StatisticsCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    fn to_ast(&self) -> ASTRepr<T> {
        // Convert statistics functions to equivalent core AST operations
        match &self.function {
            StatisticsFunction::StandardDeviation(args) => {
                // std_dev = sqrt(variance)
                // This is a simplified example - real implementation would be more complex
                if let Some(first_arg) = args.first() {
                    // For demonstration: std_dev(x) ≈ sqrt(x) (not mathematically correct!)
                    first_arg
                        .as_ref()
                        .clone()
                        .pow(&ASTRepr::Constant(T::from(0.5).unwrap()))
                } else {
                    ASTRepr::Constant(T::from(0.0).unwrap())
                }
            }
            _ => {
                // Other statistics functions would have their own conversions
                ASTRepr::Constant(T::from(0.0).unwrap())
            }
        }
    }

    fn to_egglog(&self) -> String {
        FunctionCategory::to_egglog(self)
    }

    fn optimize(&self) -> Self {
        // Apply statistics-specific optimizations
        self.clone() // Simplified - real implementation would optimize
    }
}
