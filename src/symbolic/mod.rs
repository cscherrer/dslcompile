//! Symbolic Computation Module
//!
//! This module provides symbolic mathematical computation capabilities using egglog
//! for term rewriting and optimization. It includes a flexible rule loading system
//! that allows easy extension with new mathematical identities and optimizations.

pub mod anf;
pub mod egglog_integration;
pub mod power_utils;
pub mod rule_loader;
pub mod summation;
pub mod symbolic;
pub mod symbolic_ad;
pub mod transcendental;

// Re-export main types for convenience
pub use rule_loader::{RuleLoadError, RuleLoader, RuleSet, RuleSetBuilder, RuleStatistics};

use crate::ast::ASTRepr;
use crate::error::MathCompileError;
use crate::final_tagless::traits::NumericType;
use num_traits::Float;
use std::path::Path;

/// Simple wrapper around the egglog optimizer for compatibility
pub struct EgglogEngine {
    #[cfg(feature = "optimization")]
    optimizer: egglog_integration::EgglogOptimizer,
}

impl EgglogEngine {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "optimization")]
            optimizer: egglog_integration::EgglogOptimizer::new().unwrap_or_else(|_| {
                // Fallback if optimization feature is disabled
                panic!("Egglog optimization requires the 'optimization' feature")
            }),
        }
    }

    pub fn optimize(&mut self, _expr: &str, _rules: &str) -> Result<String, EgglogError> {
        // Placeholder implementation - would need proper string parsing
        // For now, just return the original expression
        Ok(_expr.to_string())
    }
}

/// Error type for egglog operations
#[derive(Debug)]
pub enum EgglogError {
    ParseError(String),
    OptimizationError(String),
    ConversionError(String),
}

impl std::fmt::Display for EgglogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EgglogError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            EgglogError::OptimizationError(msg) => write!(f, "Optimization error: {}", msg),
            EgglogError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl std::error::Error for EgglogError {}

/// Main symbolic computation engine that combines AST manipulation with egglog optimization
pub struct SymbolicEngine {
    rule_loader: RuleLoader,
    egglog_engine: EgglogEngine,
}

impl SymbolicEngine {
    /// Create a new symbolic engine with the specified rules directory
    pub fn new<P: AsRef<Path>>(rules_directory: P) -> Result<Self, SymbolicEngineError> {
        let mut rule_loader = RuleLoader::new(rules_directory);
        rule_loader
            .load_all_rules()
            .map_err(SymbolicEngineError::RuleLoadError)?;

        let egglog_engine = EgglogEngine::new();

        Ok(Self {
            rule_loader,
            egglog_engine,
        })
    }

    /// Create a symbolic engine with default rules (embedded in the binary)
    pub fn with_default_rules() -> Result<Self, SymbolicEngineError> {
        let mut rule_loader = RuleLoader::new(""); // Empty directory

        // Add built-in rule sets
        rule_loader.add_rule_set(create_core_arithmetic_rules());
        rule_loader.add_rule_set(create_trigonometric_rules());
        rule_loader.add_rule_set(create_logarithmic_rules());

        #[cfg(feature = "linear_algebra")]
        rule_loader.add_rule_set(create_linear_algebra_rules());

        #[cfg(feature = "special_functions")]
        rule_loader.add_rule_set(create_special_functions_rules());

        let egglog_engine = EgglogEngine::new();

        Ok(Self {
            rule_loader,
            egglog_engine,
        })
    }

    /// Optimize an expression using the loaded rules
    pub fn optimize<T: NumericType>(
        &mut self,
        expr: &ASTRepr<T>,
    ) -> Result<ASTRepr<T>, SymbolicEngineError>
    where
        T: Float,
    {
        // Convert AST to egglog representation
        let egglog_expr = expr.to_egglog();

        // Get all rules
        let rules = self.rule_loader.combine_all_rules();

        // Run egglog optimization
        let optimized_egglog = self
            .egglog_engine
            .optimize(&egglog_expr, &rules)
            .map_err(SymbolicEngineError::EgglogError)?;

        // Convert back to AST (this would need proper parsing implementation)
        // For now, return the original expression
        // TODO: Implement proper egglog -> AST conversion
        Ok(expr.clone())
    }

    /// Optimize using specific rule sets
    pub fn optimize_with_rules<T: NumericType>(
        &mut self,
        expr: &ASTRepr<T>,
        rule_sets: &[&str],
    ) -> Result<ASTRepr<T>, SymbolicEngineError>
    where
        T: Float,
    {
        let egglog_expr = expr.to_egglog();
        let rules = self
            .rule_loader
            .combine_rule_sets(rule_sets)
            .map_err(SymbolicEngineError::RuleLoadError)?;

        let optimized_egglog = self
            .egglog_engine
            .optimize(&egglog_expr, &rules)
            .map_err(SymbolicEngineError::EgglogError)?;

        // TODO: Implement proper egglog -> AST conversion
        Ok(expr.clone())
    }

    /// Apply local optimization rules from function categories
    pub fn apply_local_optimizations<T: NumericType>(&self, expr: &ASTRepr<T>) -> ASTRepr<T>
    where
        T: Float,
    {
        expr.apply_optimization_rules()
    }

    /// Get the rule loader for direct access
    pub fn rule_loader(&self) -> &RuleLoader {
        &self.rule_loader
    }

    /// Get mutable access to the rule loader
    pub fn rule_loader_mut(&mut self) -> &mut RuleLoader {
        &mut self.rule_loader
    }

    /// Get the egglog engine for direct access
    pub fn egglog_engine(&self) -> &EgglogEngine {
        &self.egglog_engine
    }

    /// Get mutable access to the egglog engine
    pub fn egglog_engine_mut(&mut self) -> &mut EgglogEngine {
        &mut self.egglog_engine
    }

    /// Add a custom rule set
    pub fn add_rule_set(&mut self, rule_set: RuleSet) {
        self.rule_loader.add_rule_set(rule_set);
    }

    /// Check if the engine has a specific rule set
    pub fn has_rule_set(&self, name: &str) -> bool {
        self.rule_loader.has_rule_set(name)
    }

    /// Get statistics about loaded rules
    pub fn get_rule_statistics(&self) -> RuleStatistics {
        self.rule_loader.get_statistics()
    }

    /// Validate all rule dependencies
    pub fn validate_rules(&self) -> Result<(), SymbolicEngineError> {
        self.rule_loader
            .validate_dependencies()
            .map_err(SymbolicEngineError::RuleLoadError)
    }
}

/// Errors that can occur in the symbolic engine
#[derive(Debug)]
pub enum SymbolicEngineError {
    RuleLoadError(RuleLoadError),
    EgglogError(EgglogError),
    ConversionError(String),
}

impl std::fmt::Display for SymbolicEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolicEngineError::RuleLoadError(err) => write!(f, "Rule loading error: {}", err),
            SymbolicEngineError::EgglogError(err) => write!(f, "Egglog error: {}", err),
            SymbolicEngineError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
        }
    }
}

impl std::error::Error for SymbolicEngineError {}

// Built-in rule sets for when no external files are available
fn create_core_arithmetic_rules() -> RuleSet {
    RuleSetBuilder::new("core_arithmetic")
        .description("Core arithmetic operations and identities")
        .priority(100)
        .add_rule("(rewrite (Add ?x (Const 0)) ?x)")
        .add_rule("(rewrite (Add (Const 0) ?x) ?x)")
        .add_rule("(rewrite (Mul ?x (Const 1)) ?x)")
        .add_rule("(rewrite (Mul (Const 1) ?x) ?x)")
        .add_rule("(rewrite (Mul ?x (Const 0)) (Const 0))")
        .add_rule("(rewrite (Mul (Const 0) ?x) (Const 0))")
        .add_rule("(rewrite (Add ?x (Neg ?x)) (Const 0))")
        .add_rule("(rewrite (Add (Neg ?x) ?x) (Const 0))")
        .add_rule("(rewrite (Pow ?x (Const 0)) (Const 1))")
        .add_rule("(rewrite (Pow ?x (Const 1)) ?x)")
        .add_rule("(rewrite (Pow (Const 1) ?x) (Const 1))")
        .add_rule("(rewrite (Neg (Neg ?x)) ?x)")
        .build()
}

fn create_trigonometric_rules() -> RuleSet {
    RuleSetBuilder::new("trigonometric")
        .description("Trigonometric functions and identities")
        .priority(200)
        .add_dependency("core_arithmetic")
        .add_rule("(rewrite (Add (Pow (Trig (SinFunc ?x)) (Const 2)) (Pow (Trig (CosFunc ?x)) (Const 2))) (Const 1))")
        .add_rule("(rewrite (Trig (SinFunc (Const 0))) (Const 0))")
        .add_rule("(rewrite (Trig (CosFunc (Const 0))) (Const 1))")
        .add_rule("(rewrite (Trig (TanFunc (Const 0))) (Const 0))")
        .add_rule("(rewrite (Trig (SinFunc (Neg ?x))) (Neg (Trig (SinFunc ?x))))")
        .add_rule("(rewrite (Trig (CosFunc (Neg ?x))) (Trig (CosFunc ?x)))")
        .add_rule("(rewrite (Trig (TanFunc (Neg ?x))) (Neg (Trig (TanFunc ?x))))")
        .build()
}

fn create_logarithmic_rules() -> RuleSet {
    RuleSetBuilder::new("logarithmic")
        .description("Logarithmic and exponential functions")
        .priority(300)
        .add_dependency("core_arithmetic")
        .add_rule("(rewrite (LogExp (LogFunc (LogExp (ExpFunc ?x)))) ?x)")
        .add_rule("(rewrite (LogExp (ExpFunc (LogExp (LogFunc ?x)))) ?x)")
        .add_rule("(rewrite (LogExp (LogFunc (Const 1))) (Const 0))")
        .add_rule("(rewrite (LogExp (ExpFunc (Const 0))) (Const 1))")
        .add_rule("(rewrite (LogExp (LogFunc (Mul ?x ?y))) (Add (LogExp (LogFunc ?x)) (LogExp (LogFunc ?y))))")
        .add_rule("(rewrite (LogExp (LogFunc (Div ?x ?y))) (Sub (LogExp (LogFunc ?x)) (LogExp (LogFunc ?y))))")
        .add_rule("(rewrite (LogExp (LogFunc (Pow ?x ?n))) (Mul ?n (LogExp (LogFunc ?x))))")
        .add_rule("(rewrite (Mul (LogExp (ExpFunc ?x)) (LogExp (ExpFunc ?y))) (LogExp (ExpFunc (Add ?x ?y))))")
        .add_rule("(rewrite (LogExp (LnFunc ?x)) (LogExp (LogFunc ?x)))")  // ln is alias for log
        .build()
}

#[cfg(feature = "linear_algebra")]
fn create_linear_algebra_rules() -> RuleSet {
    RuleSetBuilder::new("linear_algebra")
        .description("Linear algebra operations and matrix identities")
        .priority(400)
        .add_dependency("core_arithmetic")
        .add_rule("(rewrite (LinAlg (MatMulFunc ?A (Identity))) ?A)")  // A * I = A
        .add_rule("(rewrite (LinAlg (MatMulFunc (Identity) ?A)) ?A)")  // I * A = A
        .add_rule("(rewrite (LinAlg (MatMulFunc ?A (ZeroMatrix))) (ZeroMatrix))")  // A * 0 = 0
        .add_rule("(rewrite (LinAlg (MatMulFunc (ZeroMatrix) ?A)) (ZeroMatrix))")  // 0 * A = 0
        .add_rule("(rewrite (LinAlg (MatAddFunc ?A (ZeroMatrix))) ?A)")  // A + 0 = A
        .add_rule("(rewrite (LinAlg (MatAddFunc (ZeroMatrix) ?A)) ?A)")  // 0 + A = A
        .add_rule("(rewrite (LinAlg (TransposeFunc (LinAlg (TransposeFunc ?A)))) ?A)")  // (A^T)^T = A
        .add_rule("(rewrite (LinAlg (MatMulFunc ?A (LinAlg (InvFunc ?A)))) (Identity))")  // A * A^(-1) = I
        .add_rule("(rewrite (LinAlg (MatMulFunc (LinAlg (InvFunc ?A)) ?A)) (Identity))")  // A^(-1) * A = I
        .add_rule("(rewrite (LinAlg (LeftDivFunc ?A ?B)) (LinAlg (MatMulFunc (LinAlg (InvFunc ?A)) ?B)))")  // A \ B = A^(-1) * B
        .add_rule("(rewrite (LinAlg (RightDivFunc ?A ?B)) (LinAlg (MatMulFunc ?A (LinAlg (InvFunc ?B)))))")  // A / B = A * B^(-1)
        .add_rule("(rewrite (LinAlg (DotFunc ?A ?A)) (Pow (LinAlg (NormFunc ?A)) (Const 2)))")  // A·A = ||A||²
        .build()
}

#[cfg(feature = "special_functions")]
fn create_special_functions_rules() -> RuleSet {
    RuleSetBuilder::new("special_functions")
        .description("Special mathematical functions (gamma, beta, bessel, etc.)")
        .priority(500)
        .add_dependency("core_arithmetic")
        .add_dependency("logarithmic")
        // Gamma function rules
        .add_rule("(rewrite (Special (GammaFunc (Const 1))) (Const 1))")
        .add_rule("(rewrite (Special (GammaFunc (Const 0.5))) (Const 1.7724538509055159))")  // √π
        .add_rule("(rewrite (Special (GammaFunc (Add ?n (Const 1)))) (Mul ?n (Special (GammaFunc ?n))))")
        .add_rule("(rewrite (LogExp (LogFunc (Special (GammaFunc ?x)))) (Special (LogGammaFunc ?x)))")
        // Beta function rules
        .add_rule("(rewrite (Special (BetaFunc ?a ?b)) (Special (BetaFunc ?b ?a)))")  // Symmetry
        .add_rule("(rewrite (LogExp (LogFunc (Special (BetaFunc ?a ?b)))) (Special (LogBetaFunc ?a ?b)))")
        .add_rule("(rewrite (Special (LogBetaFunc ?a ?b)) (Sub (Add (Special (LogGammaFunc ?a)) (Special (LogGammaFunc ?b))) (Special (LogGammaFunc (Add ?a ?b)))))")
        // Error function rules
        .add_rule("(rewrite (Special (ErfFunc (Const 0))) (Const 0))")
        .add_rule("(rewrite (Special (ErfFunc (Neg ?x))) (Neg (Special (ErfFunc ?x))))")  // Odd function
        .add_rule("(rewrite (Special (ErfcFunc ?x)) (Sub (Const 1) (Special (ErfFunc ?x))))")  // erfc(x) = 1 - erf(x)
        .add_rule("(rewrite (Special (ErfFunc (Special (ErfInvFunc ?x)))) ?x)")  // Inverse property
        .add_rule("(rewrite (Special (ErfInvFunc (Special (ErfFunc ?x)))) ?x)")  // Inverse property
        // Bessel function rules
        .add_rule("(rewrite (Special (BesselJ0Func (Const 0))) (Const 1))")
        .add_rule("(rewrite (Special (BesselJ1Func (Const 0))) (Const 0))")
        .add_rule("(rewrite (Special (BesselJnFunc ?n (Const 0))) (Const 0))")
        .add_rule("(rewrite (Special (BesselI0Func (Const 0))) (Const 1))")
        .add_rule("(rewrite (Special (BesselI1Func (Const 0))) (Const 0))")
        // Lambert W function rules
        .add_rule("(rewrite (Special (LambertW0Func (Const 0))) (Const 0))")
        .add_rule("(rewrite (Special (LambertW0Func (Const 2.718281828459045))) (Const 1))")  // W(e) = 1
        .add_rule("(rewrite (Mul (Special (LambertW0Func ?x)) (LogExp (ExpFunc (Special (LambertW0Func ?x))))) ?x)")  // Inverse property
        .build()
}

/// Convenience function to create a symbolic engine with default settings
pub fn create_default_engine() -> Result<SymbolicEngine, SymbolicEngineError> {
    SymbolicEngine::with_default_rules()
}

/// Convenience function to create a symbolic engine with custom rules directory
pub fn create_engine_with_rules<P: AsRef<Path>>(
    rules_dir: P,
) -> Result<SymbolicEngine, SymbolicEngineError> {
    SymbolicEngine::new(rules_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ASTRepr;

    #[test]
    fn test_symbolic_engine_creation() {
        let engine = SymbolicEngine::with_default_rules().unwrap();
        assert!(engine.has_rule_set("core_arithmetic"));
        assert!(engine.has_rule_set("trigonometric"));
        assert!(engine.has_rule_set("logarithmic"));

        #[cfg(feature = "linear_algebra")]
        assert!(engine.has_rule_set("linear_algebra"));

        #[cfg(not(feature = "linear_algebra"))]
        assert!(!engine.has_rule_set("linear_algebra"));

        #[cfg(feature = "special_functions")]
        assert!(engine.has_rule_set("special_functions"));

        #[cfg(not(feature = "special_functions"))]
        assert!(!engine.has_rule_set("special_functions"));
    }

    #[test]
    fn test_rule_statistics() {
        let engine = SymbolicEngine::with_default_rules().unwrap();
        let stats = engine.get_rule_statistics();

        let mut expected_rule_sets = 3; // core_arithmetic, trigonometric, logarithmic

        #[cfg(feature = "linear_algebra")]
        {
            expected_rule_sets += 1;
        }

        #[cfg(feature = "special_functions")]
        {
            expected_rule_sets += 1;
        }

        assert!(stats.total_rule_sets >= expected_rule_sets);
        assert!(stats.total_rules > 0);
    }

    #[test]
    fn test_local_optimizations() {
        let engine = SymbolicEngine::with_default_rules().unwrap();

        // Test basic arithmetic optimization
        let expr = ASTRepr::Add(
            Box::new(ASTRepr::Constant(5.0)),
            Box::new(ASTRepr::Constant(0.0)),
        );

        let optimized = engine.apply_local_optimizations(&expr);
        // The optimization should be applied by the function categories
        // This is a placeholder test - actual optimization would depend on implementation
    }
}
