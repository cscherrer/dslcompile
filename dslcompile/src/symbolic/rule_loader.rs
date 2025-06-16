//! Rule Loader for Egglog Integration
//!
//! This module provides functionality to load, validate, and combine
//! egglog rule files for mathematical optimization.

use crate::{
    error::{DSLCompileError, Result},
    interval_domain::IntervalDomain,
};
use std::{fs, path::PathBuf};

/// Categories of mathematical rules available
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuleCategory {
    /// Core datatypes (always required)
    CoreDatatypes,
    /// Basic arithmetic operations
    BasicArithmetic,
    /// Domain-aware arithmetic (with preconditions)
    DomainAwareArithmetic,
    /// Transcendental functions (exp, ln, etc.)
    Transcendental,
    /// Trigonometric functions (sin, cos, etc.)
    Trigonometric,
    /// Summation rules (production-ready, focused optimizations)
    Summation,
    /// Dependency analysis for safe optimizations
    DependencyAnalysis,
}

impl RuleCategory {
    /// Get the filename for this rule category
    #[must_use]
    pub fn filename(&self) -> &'static str {
        match self {
            RuleCategory::CoreDatatypes => "core_datatypes.egg",
            RuleCategory::BasicArithmetic => "basic_arithmetic.egg",
            RuleCategory::DomainAwareArithmetic => "domain_aware_arithmetic.egg",
            RuleCategory::Transcendental => "transcendental.egg",
            RuleCategory::Trigonometric => "trigonometric.egg",
            RuleCategory::Summation => "clean_summation_rules.egg",
            RuleCategory::DependencyAnalysis => "dependency_analysis.egg",
        }
    }

    /// Get a description of this rule category
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            RuleCategory::CoreDatatypes => "Core mathematical expression datatypes",
            RuleCategory::BasicArithmetic => "Basic arithmetic operations and identities",
            RuleCategory::DomainAwareArithmetic => "Domain-aware arithmetic with preconditions",
            RuleCategory::Transcendental => "Exponential and logarithmic functions",
            RuleCategory::Trigonometric => "Trigonometric functions and identities",
            RuleCategory::Summation => "Clean summation rules with focused optimizations",
            RuleCategory::DependencyAnalysis => "Dependency analysis for safe variable-aware optimizations",
        }
    }

    /// Get all available rule categories
    #[must_use]
    pub fn all() -> Vec<RuleCategory> {
        vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::BasicArithmetic,
            RuleCategory::DomainAwareArithmetic,
            RuleCategory::Transcendental,
            RuleCategory::Trigonometric,
            RuleCategory::Summation,
            RuleCategory::DependencyAnalysis,
        ]
    }

    /// Get the default set of rule categories for basic optimization
    #[must_use]
    pub fn default_set() -> Vec<RuleCategory> {
        vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::BasicArithmetic,
            RuleCategory::Transcendental,
        ]
    }

    /// Get the domain-aware set of rule categories for safe optimization
    #[must_use]
    pub fn domain_aware_set() -> Vec<RuleCategory> {
        vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::DomainAwareArithmetic,
            RuleCategory::Transcendental,
        ]
    }

    /// Get the clean summation set of rule categories for production-ready summation optimization
    #[must_use]
    pub fn clean_summation_set() -> Vec<RuleCategory> {
        vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::BasicArithmetic,
            RuleCategory::Summation,
        ]
    }

    /// Get the safe optimization set with dependency analysis for variable-aware optimization
    #[must_use]
    pub fn safe_optimization_set() -> Vec<RuleCategory> {
        vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::DependencyAnalysis,
        ]
    }
}

/// Configuration for rule loading
#[derive(Debug, Clone)]
pub struct RuleConfig {
    /// Categories of rules to load
    pub categories: Vec<RuleCategory>,
    /// Custom rules directory (defaults to "`dslcompile/src/egglog_rules`/")
    pub rules_directory: Option<PathBuf>,
    /// Whether to validate rule syntax
    pub validate_syntax: bool,
    /// Whether to include debug comments in the combined program
    pub include_comments: bool,
    /// Whether to generate domain-aware rules dynamically
    pub generate_domain_aware: bool,
    /// Domain constraints for variables (`variable_name` -> domain)
    pub variable_domains: std::collections::HashMap<String, IntervalDomain<f64>>,
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            categories: RuleCategory::default_set(),
            rules_directory: None,
            validate_syntax: true,
            include_comments: false,
            generate_domain_aware: false,
            variable_domains: std::collections::HashMap::new(),
        }
    }
}

impl RuleConfig {
    /// Create a domain-aware configuration
    #[must_use]
    pub fn domain_aware() -> Self {
        Self {
            categories: RuleCategory::domain_aware_set(),
            generate_domain_aware: true,
            ..Default::default()
        }
    }

    /// Add a domain constraint for a variable
    #[must_use]
    pub fn with_variable_domain(mut self, var_name: &str, domain: IntervalDomain<f64>) -> Self {
        self.variable_domains.insert(var_name.to_string(), domain);
        self
    }

    /// Create a clean summation configuration for production-ready summation optimization
    #[must_use]
    pub fn clean_summation() -> Self {
        Self {
            categories: RuleCategory::clean_summation_set(),
            validate_syntax: true,
            include_comments: true,
            ..Default::default()
        }
    }

    /// Create a safe optimization configuration with dependency analysis
    #[must_use]
    pub fn safe_optimization() -> Self {
        Self {
            categories: RuleCategory::safe_optimization_set(),
            validate_syntax: true,
            include_comments: true,
            ..Default::default()
        }
    }
}

/// Rule loader for egglog programs
pub struct RuleLoader {
    config: RuleConfig,
    rules_dir: PathBuf,
}

impl RuleLoader {
    /// Create a new rule loader with the given configuration
    #[must_use]
    pub fn new(config: RuleConfig) -> Self {
        let rules_dir = config
            .rules_directory
            .clone()
            .unwrap_or_else(|| {
                // Try different possible paths depending on working directory
                let candidates = [
                    "dslcompile/src/egglog_rules",
                    "src/egglog_rules", 
                    "../dslcompile/src/egglog_rules",
                ];
                
                for candidate in &candidates {
                    let path = PathBuf::from(candidate);
                    if path.exists() {
                        return path;
                    }
                }
                
                // Fallback to the original path
                PathBuf::from("dslcompile/src/egglog_rules")
            });

        Self { config, rules_dir }
    }

    /// Create a rule loader with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(RuleConfig::default())
    }

    /// Create a domain-aware rule loader
    #[must_use]
    pub fn domain_aware() -> Self {
        Self::new(RuleConfig::domain_aware())
    }

    /// Create a clean summation rule loader for production-ready summation optimization
    #[must_use]
    pub fn clean_summation() -> Self {
        Self::new(RuleConfig::clean_summation())
    }

    /// Create a safe optimization rule loader with dependency analysis
    #[must_use]
    pub fn safe_optimization() -> Self {
        Self::new(RuleConfig::safe_optimization())
    }

    /// Load and combine all configured rule files into a single egglog program
    pub fn load_rules(&self) -> Result<String> {
        let mut program = String::new();

        if self.config.include_comments {
            program.push_str("; Combined Egglog Program for DSLCompile\n");
            program.push_str("; Generated by RuleLoader\n\n");
        }

        // Always load core datatypes first
        if !self
            .config
            .categories
            .contains(&RuleCategory::CoreDatatypes)
        {
            let core_content = self.load_rule_file(&RuleCategory::CoreDatatypes)?;
            program.push_str(&core_content);
            program.push('\n');
        }

        // Load all configured rule categories
        for category in &self.config.categories {
            if self.config.include_comments {
                program.push_str("; ========================================\n");
                program.push_str(&format!("; {}\n", category.description()));
                program.push_str("; ========================================\n\n");
            }

            let content = self.load_rule_file(category)?;
            program.push_str(&content);
            program.push('\n');
        }

        // Generate domain-aware rules if requested
        if self.config.generate_domain_aware {
            if self.config.include_comments {
                program.push_str("; ========================================\n");
                program.push_str("; DYNAMICALLY GENERATED DOMAIN-AWARE RULES\n");
                program.push_str("; ========================================\n\n");
            }

            let domain_rules = self.generate_domain_aware_rules()?;
            program.push_str(&domain_rules);
            program.push('\n');
        }

        if self.config.validate_syntax {
            self.validate_program_syntax(&program)?;
        }

        Ok(program)
    }

    /// Generate domain-aware rules based on variable domains
    fn generate_domain_aware_rules(&self) -> Result<String> {
        let mut rules = String::new();

        rules.push_str("; Domain-aware power rules\n");

        // Generate rules based on known variable domains
        for (var_name, domain) in &self.config.variable_domains {
            if domain.is_positive(0.0) {
                rules.push_str(&format!(
                    "; Variable {var_name} is positive, safe to use x^0 = 1\n"
                ));
                rules.push_str(&format!(
                    "(rewrite (Pow (Var \"{var_name}\") (Num 0.0)) (Num 1.0))\n"
                ));
            }

            if domain.is_non_negative(0.0) {
                rules.push_str(&format!(
                    "; Variable {var_name} is non-negative, safe to use sqrt(x^2) = x\n"
                ));
                rules.push_str(&format!(
                    "(rewrite (Sqrt (Mul (Var \"{var_name}\") (Var \"{var_name}\"))) (Var \"{var_name}\"))\n"
                ));
            }
        }

        // Add IEEE 754 compliant rules with comments
        rules.push_str("\n; IEEE 754 compliant rules (computational, not mathematical)\n");
        rules.push_str("; These follow IEEE 754 standard but may not be mathematically rigorous\n");
        rules.push_str("(rewrite (Pow (Num 0.0) (Num 0.0)) (Num 1.0))  ; IEEE 754: 0^0 = 1\n");

        Ok(rules)
    }

    /// Load a specific rule file
    fn load_rule_file(&self, category: &RuleCategory) -> Result<String> {
        let file_path = self.rules_dir.join(category.filename());

        fs::read_to_string(&file_path).map_err(|e| {
            DSLCompileError::Generic(format!(
                "Failed to load rule file {}: {}",
                file_path.display(),
                e
            ))
        })
    }

    /// Validate the syntax of the combined program
    fn validate_program_syntax(&self, program: &str) -> Result<()> {
        // Basic syntax validation
        let mut paren_count = 0;
        let mut in_comment = false;

        for line in program.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(';') {
                continue;
            }

            for ch in line.chars() {
                match ch {
                    ';' => in_comment = true,
                    '\n' => in_comment = false,
                    '(' if !in_comment => paren_count += 1,
                    ')' if !in_comment => paren_count -= 1,
                    _ => {}
                }
            }
            in_comment = false; // Reset at end of line
        }

        if paren_count != 0 {
            return Err(DSLCompileError::Generic(format!(
                "Unbalanced parentheses in egglog program: {paren_count} unclosed"
            )));
        }

        // Check for required elements
        if !program.contains("datatype Math") {
            return Err(DSLCompileError::Generic(
                "Missing required 'datatype Math' definition".to_string(),
            ));
        }

        Ok(())
    }

    /// Get information about available rule files
    pub fn list_available_rules(&self) -> Result<Vec<(RuleCategory, bool, String)>> {
        let mut rules_info = Vec::new();

        for category in RuleCategory::all() {
            let file_path = self.rules_dir.join(category.filename());
            let exists = file_path.exists();
            let description = category.description().to_string();
            rules_info.push((category, exists, description));
        }

        Ok(rules_info)
    }

    /// Check if all required rule files exist
    pub fn validate_rule_files(&self) -> Result<()> {
        for category in &self.config.categories {
            let file_path = self.rules_dir.join(category.filename());
            if !file_path.exists() {
                return Err(DSLCompileError::Generic(format!(
                    "Required rule file not found: {}",
                    file_path.display()
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_category_filename() {
        assert_eq!(RuleCategory::CoreDatatypes.filename(), "core_datatypes.egg");
        assert_eq!(
            RuleCategory::BasicArithmetic.filename(),
            "basic_arithmetic.egg"
        );
        assert_eq!(
            RuleCategory::Transcendental.filename(),
            "transcendental.egg"
        );
    }

    #[test]
    fn test_default_rule_config() {
        let config = RuleConfig::default();
        assert!(config.categories.contains(&RuleCategory::CoreDatatypes));
        assert!(config.categories.contains(&RuleCategory::BasicArithmetic));
        assert!(config.validate_syntax);
    }

    #[test]
    fn test_rule_loader_creation() {
        let loader = RuleLoader::default();
        // The path should be one of the valid candidates that actually exists
        let path_str = loader.rules_dir.to_string_lossy();
        assert!(
            path_str.ends_with("egglog_rules"),
            "Expected path to end with 'egglog_rules', got: {path_str}"
        );
    }

    #[test]
    fn test_syntax_validation() {
        let loader = RuleLoader::default();

        // Valid program
        let valid_program = "(datatype Math (Num f64))";
        assert!(loader.validate_program_syntax(valid_program).is_ok());

        // Invalid program - unbalanced parentheses
        let invalid_program = "(datatype Math (Num f64)";
        assert!(loader.validate_program_syntax(invalid_program).is_err());

        // Invalid program - missing datatype
        let missing_datatype = "(rewrite (Add ?x ?y) (Add ?y ?x))";
        assert!(loader.validate_program_syntax(missing_datatype).is_err());
    }
}
