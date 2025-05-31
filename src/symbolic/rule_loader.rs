//! Rule Loading System for MathCompile
//!
//! This module provides a flexible system for loading egglog rules from separate files,
//! enabling easy extension and maintenance of mathematical identities and optimizations.

use crate::error::MathCompileError;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Represents a collection of egglog rules from a specific domain
#[derive(Debug, Clone)]
pub struct RuleSet {
    pub name: String,
    pub description: String,
    pub rules: String,
    pub dependencies: Vec<String>,
    pub priority: u32,
}

/// Manages loading and organizing egglog rules from multiple sources
#[derive(Debug)]
pub struct RuleLoader {
    rule_sets: HashMap<String, RuleSet>,
    rule_directory: PathBuf,
}

impl RuleLoader {
    /// Create a new rule loader with the specified rules directory
    pub fn new<P: AsRef<Path>>(rule_directory: P) -> Self {
        Self {
            rule_sets: HashMap::new(),
            rule_directory: rule_directory.as_ref().to_path_buf(),
        }
    }

    /// Load all rule files from the rules directory
    pub fn load_all_rules(&mut self) -> Result<(), RuleLoadError> {
        // Define the standard rule files and their metadata
        let standard_rules = vec![
            (
                "core_arithmetic",
                "Core arithmetic operations and identities",
                100,
            ),
            (
                "trigonometric",
                "Trigonometric functions and identities",
                200,
            ),
            ("logarithmic", "Logarithmic and exponential functions", 300),
            ("hyperbolic", "Hyperbolic functions and identities", 400),
            ("calculus", "Differentiation and integration rules", 500),
            ("linear_algebra", "Matrix and vector operations", 600),
            ("special_functions", "Special mathematical functions", 700),
            (
                "number_theory",
                "Number theoretic functions and identities",
                800,
            ),
            ("complex_analysis", "Complex number operations", 900),
        ];

        for (name, description, priority) in standard_rules {
            if let Err(e) = self.load_rule_file(name, description, priority) {
                // Log warning but continue loading other files
                eprintln!("Warning: Failed to load rule file '{}': {}", name, e);
            }
        }

        // Load any custom rule files found in the directory
        self.discover_custom_rules()?;

        Ok(())
    }

    /// Load a specific rule file
    pub fn load_rule_file(
        &mut self,
        name: &str,
        description: &str,
        priority: u32,
    ) -> Result<(), RuleLoadError> {
        let file_path = self.rule_directory.join(format!("{}.egg", name));

        if !file_path.exists() {
            return Err(RuleLoadError::FileNotFound(file_path));
        }

        let content = fs::read_to_string(&file_path)
            .map_err(|e| RuleLoadError::IoError(file_path.clone(), e))?;

        let dependencies = self.extract_dependencies(&content);

        let rule_set = RuleSet {
            name: name.to_string(),
            description: description.to_string(),
            rules: content,
            dependencies,
            priority,
        };

        self.rule_sets.insert(name.to_string(), rule_set);
        Ok(())
    }

    /// Discover and load custom rule files
    fn discover_custom_rules(&mut self) -> Result<(), RuleLoadError> {
        if !self.rule_directory.exists() {
            return Ok(()); // No custom rules directory
        }

        let entries = fs::read_dir(&self.rule_directory)
            .map_err(|e| RuleLoadError::IoError(self.rule_directory.clone(), e))?;

        for entry in entries {
            let entry =
                entry.map_err(|e| RuleLoadError::IoError(self.rule_directory.clone(), e))?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("egg") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    // Skip if already loaded
                    if !self.rule_sets.contains_key(stem) {
                        let content = fs::read_to_string(&path)
                            .map_err(|e| RuleLoadError::IoError(path.clone(), e))?;

                        let dependencies = self.extract_dependencies(&content);

                        let rule_set = RuleSet {
                            name: stem.to_string(),
                            description: format!("Custom rules from {}", stem),
                            rules: content,
                            dependencies,
                            priority: 1000, // Custom rules have lower priority by default
                        };

                        self.rule_sets.insert(stem.to_string(), rule_set);
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract dependencies from rule content (simple heuristic)
    fn extract_dependencies(&self, content: &str) -> Vec<String> {
        let mut dependencies = Vec::new();

        // Look for comments indicating dependencies
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with(";; depends:") || line.starts_with(";; requires:") {
                let deps_str = line.split(':').nth(1).unwrap_or("").trim();
                for dep in deps_str.split(',') {
                    let dep = dep.trim();
                    if !dep.is_empty() {
                        dependencies.push(dep.to_string());
                    }
                }
            }
        }

        dependencies
    }

    /// Get all loaded rule sets
    pub fn get_rule_sets(&self) -> &HashMap<String, RuleSet> {
        &self.rule_sets
    }

    /// Get a specific rule set by name
    pub fn get_rule_set(&self, name: &str) -> Option<&RuleSet> {
        self.rule_sets.get(name)
    }

    /// Get rule sets sorted by priority (higher priority first)
    pub fn get_rule_sets_by_priority(&self) -> Vec<&RuleSet> {
        let mut rule_sets: Vec<&RuleSet> = self.rule_sets.values().collect();
        rule_sets.sort_by(|a, b| a.priority.cmp(&b.priority));
        rule_sets
    }

    /// Combine all rules into a single string for egglog
    pub fn combine_all_rules(&self) -> String {
        let mut combined = String::new();

        // Add header
        combined.push_str(";; Combined Rules for MathCompile\n");
        combined.push_str(";; Generated automatically from rule files\n\n");

        // Add rules in priority order
        for rule_set in self.get_rule_sets_by_priority() {
            combined.push_str(&format!(";; === {} ===\n", rule_set.name));
            combined.push_str(&format!(";; {}\n", rule_set.description));
            combined.push_str(&format!(";; Priority: {}\n", rule_set.priority));
            if !rule_set.dependencies.is_empty() {
                combined.push_str(&format!(
                    ";; Dependencies: {}\n",
                    rule_set.dependencies.join(", ")
                ));
            }
            combined.push_str("\n");
            combined.push_str(&rule_set.rules);
            combined.push_str("\n\n");
        }

        combined
    }

    /// Combine specific rule sets
    pub fn combine_rule_sets(&self, names: &[&str]) -> Result<String, RuleLoadError> {
        let mut combined = String::new();

        // Resolve dependencies and collect rule set names
        let mut selected_names = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for &name in names {
            self.collect_dependencies(name, &mut selected_names, &mut visited)?;
        }

        // Get rule sets and sort by priority
        let mut selected_sets: Vec<&RuleSet> = selected_names
            .iter()
            .filter_map(|name| self.rule_sets.get(name))
            .collect();
        selected_sets.sort_by(|a, b| a.priority.cmp(&b.priority));

        // Combine rules
        combined.push_str(";; Selected Rules for MathCompile\n\n");
        for rule_set in selected_sets {
            combined.push_str(&format!(";; === {} ===\n", rule_set.name));
            combined.push_str(&rule_set.rules);
            combined.push_str("\n\n");
        }

        Ok(combined)
    }

    /// Recursively collect dependencies
    fn collect_dependencies(
        &self,
        name: &str,
        collected: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
    ) -> Result<(), RuleLoadError> {
        if visited.contains(name) {
            return Ok(()); // Already processed
        }

        let rule_set = self
            .rule_sets
            .get(name)
            .ok_or_else(|| RuleLoadError::RuleSetNotFound(name.to_string()))?;

        visited.insert(name.to_string());

        // First collect dependencies
        for dep in &rule_set.dependencies {
            self.collect_dependencies(dep, collected, visited)?;
        }

        // Then add this rule set name
        collected.push(name.to_string());

        Ok(())
    }

    /// Validate that all dependencies are available
    pub fn validate_dependencies(&self) -> Result<(), RuleLoadError> {
        for rule_set in self.rule_sets.values() {
            for dep in &rule_set.dependencies {
                if !self.rule_sets.contains_key(dep) {
                    return Err(RuleLoadError::MissingDependency {
                        rule_set: rule_set.name.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Add a rule set programmatically
    pub fn add_rule_set(&mut self, rule_set: RuleSet) {
        self.rule_sets.insert(rule_set.name.clone(), rule_set);
    }

    /// Remove a rule set
    pub fn remove_rule_set(&mut self, name: &str) -> Option<RuleSet> {
        self.rule_sets.remove(name)
    }

    /// Check if a rule set exists
    pub fn has_rule_set(&self, name: &str) -> bool {
        self.rule_sets.contains_key(name)
    }

    /// Get statistics about loaded rules
    pub fn get_statistics(&self) -> RuleStatistics {
        let total_rule_sets = self.rule_sets.len();
        let total_rules = self
            .rule_sets
            .values()
            .map(|rs| {
                rs.rules
                    .lines()
                    .filter(|line| line.trim().starts_with("(rewrite"))
                    .count()
            })
            .sum();

        let rule_sets_by_priority: HashMap<u32, usize> =
            self.rule_sets.values().fold(HashMap::new(), |mut acc, rs| {
                *acc.entry(rs.priority).or_insert(0) += 1;
                acc
            });

        RuleStatistics {
            total_rule_sets,
            total_rules,
            rule_sets_by_priority,
        }
    }
}

/// Statistics about loaded rules
#[derive(Debug)]
pub struct RuleStatistics {
    pub total_rule_sets: usize,
    pub total_rules: usize,
    pub rule_sets_by_priority: HashMap<u32, usize>,
}

/// Errors that can occur during rule loading
#[derive(Debug)]
pub enum RuleLoadError {
    FileNotFound(PathBuf),
    IoError(PathBuf, io::Error),
    RuleSetNotFound(String),
    MissingDependency {
        rule_set: String,
        dependency: String,
    },
    CircularDependency(Vec<String>),
}

impl std::fmt::Display for RuleLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleLoadError::FileNotFound(path) => {
                write!(f, "Rule file not found: {}", path.display())
            }
            RuleLoadError::IoError(path, err) => {
                write!(f, "IO error reading {}: {}", path.display(), err)
            }
            RuleLoadError::RuleSetNotFound(name) => {
                write!(f, "Rule set '{}' not found", name)
            }
            RuleLoadError::MissingDependency {
                rule_set,
                dependency,
            } => {
                write!(
                    f,
                    "Rule set '{}' depends on missing rule set '{}'",
                    rule_set, dependency
                )
            }
            RuleLoadError::CircularDependency(cycle) => {
                write!(f, "Circular dependency detected: {}", cycle.join(" -> "))
            }
        }
    }
}

impl std::error::Error for RuleLoadError {}

/// Builder for creating custom rule sets
pub struct RuleSetBuilder {
    name: String,
    description: String,
    rules: Vec<String>,
    dependencies: Vec<String>,
    priority: u32,
}

impl RuleSetBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            rules: Vec::new(),
            dependencies: Vec::new(),
            priority: 1000,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn add_rule(mut self, rule: impl Into<String>) -> Self {
        self.rules.push(rule.into());
        self
    }

    pub fn add_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    pub fn build(self) -> RuleSet {
        RuleSet {
            name: self.name,
            description: self.description,
            rules: self.rules.join("\n"),
            dependencies: self.dependencies,
            priority: self.priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_rule_loader_basic() {
        let temp_dir = env::temp_dir().join("mathcompile_test_rules");
        let rules_dir = temp_dir.join("rules");
        fs::create_dir_all(&rules_dir).unwrap();

        // Create a test rule file
        let test_rules = r#"
;; Test rules
(rewrite (Add ?x (Const 0)) ?x)
(rewrite (Mul ?x (Const 1)) ?x)
"#;
        fs::write(rules_dir.join("test.egg"), test_rules).unwrap();

        let mut loader = RuleLoader::new(&rules_dir);
        loader.load_rule_file("test", "Test rules", 100).unwrap();

        assert!(loader.has_rule_set("test"));
        let rule_set = loader.get_rule_set("test").unwrap();
        assert_eq!(rule_set.name, "test");
        assert_eq!(rule_set.priority, 100);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_rule_set_builder() {
        let rule_set = RuleSetBuilder::new("custom")
            .description("Custom mathematical rules")
            .priority(500)
            .add_rule("(rewrite (Add ?x ?x) (Mul (Const 2) ?x))")
            .add_dependency("core_arithmetic")
            .build();

        assert_eq!(rule_set.name, "custom");
        assert_eq!(rule_set.priority, 500);
        assert_eq!(rule_set.dependencies, vec!["core_arithmetic"]);
        assert!(rule_set.rules.contains("(rewrite (Add ?x ?x)"));
    }
}
