//! Interval-Based Domain Analysis
//!
//! This module implements a mathematically rigorous domain representation using
//! interval endpoints, eliminating redundancy while providing full expressiveness.

use crate::ast::ASTRepr;
use std::{collections::HashMap, fmt};

/// Interval endpoint specification
#[derive(Debug, Clone, PartialEq)]
pub enum Endpoint<F> {
    /// Open endpoint: (value, ...)  or  (..., value)
    Open(F),
    /// Closed endpoint: [value, ...]  or  [..., value]
    Closed(F),
    /// Unbounded: (-∞, ...)  or  (..., +∞)
    Unbounded,
}

/// Abstract domain representation using intervals
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalDomain<F> {
    /// Bottom domain (empty set)
    Bottom,
    /// Top domain (all real numbers) = (-∞, +∞)
    Top,
    /// Single constant value = {c}
    Constant(F),
    /// General interval with flexible endpoints
    Interval {
        lower: Endpoint<F>,
        upper: Endpoint<F>,
    },
}

impl<F: Copy + PartialOrd> IntervalDomain<F> {
    /// Create common domain patterns

    /// Positive real numbers: (0, +∞)
    pub fn positive(zero: F) -> Self {
        Self::Interval {
            lower: Endpoint::Open(zero),
            upper: Endpoint::Unbounded,
        }
    }

    /// Non-negative real numbers: [0, +∞)
    pub fn non_negative(zero: F) -> Self {
        Self::Interval {
            lower: Endpoint::Closed(zero),
            upper: Endpoint::Unbounded,
        }
    }

    /// Negative real numbers: (-∞, 0)
    pub fn negative(zero: F) -> Self {
        Self::Interval {
            lower: Endpoint::Unbounded,
            upper: Endpoint::Open(zero),
        }
    }

    /// Non-positive real numbers: (-∞, 0]
    pub fn non_positive(zero: F) -> Self {
        Self::Interval {
            lower: Endpoint::Unbounded,
            upper: Endpoint::Closed(zero),
        }
    }

    /// Closed interval: [a, b]
    pub fn closed_interval(a: F, b: F) -> Self {
        if a > b {
            Self::Bottom
        } else if a == b {
            Self::Constant(a)
        } else {
            Self::Interval {
                lower: Endpoint::Closed(a),
                upper: Endpoint::Closed(b),
            }
        }
    }

    /// Open interval: (a, b)
    pub fn open_interval(a: F, b: F) -> Self {
        if a >= b {
            Self::Bottom
        } else {
            Self::Interval {
                lower: Endpoint::Open(a),
                upper: Endpoint::Open(b),
            }
        }
    }

    /// Check if a value is contained in this domain
    pub fn contains(&self, value: F) -> bool {
        match self {
            IntervalDomain::Bottom => false,
            IntervalDomain::Top => true,
            IntervalDomain::Constant(c) => value == *c,
            IntervalDomain::Interval { lower, upper } => {
                let lower_ok = match lower {
                    Endpoint::Unbounded => true,
                    Endpoint::Open(bound) => value > *bound,
                    Endpoint::Closed(bound) => value >= *bound,
                };
                let upper_ok = match upper {
                    Endpoint::Unbounded => true,
                    Endpoint::Open(bound) => value < *bound,
                    Endpoint::Closed(bound) => value <= *bound,
                };
                lower_ok && upper_ok
            }
        }
    }

    /// Check if this domain contains only positive values
    pub fn is_positive(&self, zero: F) -> bool {
        match self {
            IntervalDomain::Bottom => false,
            IntervalDomain::Top => false,
            IntervalDomain::Constant(c) => *c > zero,
            IntervalDomain::Interval { lower, upper: _ } => match lower {
                Endpoint::Open(bound) => *bound >= zero,
                Endpoint::Closed(bound) => *bound > zero,
                Endpoint::Unbounded => false,
            },
        }
    }

    /// Check if this domain contains only non-negative values
    pub fn is_non_negative(&self, zero: F) -> bool {
        match self {
            IntervalDomain::Bottom => false,
            IntervalDomain::Top => false,
            IntervalDomain::Constant(c) => *c >= zero,
            IntervalDomain::Interval { lower, upper: _ } => match lower {
                Endpoint::Open(bound) => *bound >= zero,
                Endpoint::Closed(bound) => *bound >= zero,
                Endpoint::Unbounded => false,
            },
        }
    }

    /// Check if this domain contains zero
    pub fn contains_zero(&self, zero: F) -> bool {
        self.contains(zero)
    }

    /// Compute the join (least upper bound) of two domains
    pub fn join(&self, other: &IntervalDomain<F>) -> IntervalDomain<F> {
        match (self, other) {
            (IntervalDomain::Bottom, d) | (d, IntervalDomain::Bottom) => d.clone(),
            (IntervalDomain::Top, _) | (_, IntervalDomain::Top) => IntervalDomain::Top,

            (IntervalDomain::Constant(a), IntervalDomain::Constant(b)) if a == b => {
                IntervalDomain::Constant(*a)
            }

            (
                IntervalDomain::Interval {
                    lower: l1,
                    upper: u1,
                },
                IntervalDomain::Interval {
                    lower: l2,
                    upper: u2,
                },
            ) => IntervalDomain::Interval {
                lower: min_endpoint(l1, l2),
                upper: max_endpoint(u1, u2),
            },

            // Convert constants to intervals and join
            (IntervalDomain::Constant(c), IntervalDomain::Interval { .. }) => {
                let const_interval = IntervalDomain::Interval {
                    lower: Endpoint::Closed(*c),
                    upper: Endpoint::Closed(*c),
                };
                const_interval.join(other)
            }
            (IntervalDomain::Interval { .. }, IntervalDomain::Constant(c)) => {
                let const_interval = IntervalDomain::Interval {
                    lower: Endpoint::Closed(*c),
                    upper: Endpoint::Closed(*c),
                };
                self.join(&const_interval)
            }

            // Conservative: fall back to Top for complex cases
            _ => IntervalDomain::Top,
        }
    }

    /// Compute the meet (greatest lower bound) of two domains
    pub fn meet(&self, other: &IntervalDomain<F>) -> IntervalDomain<F> {
        match (self, other) {
            (IntervalDomain::Bottom, _) | (_, IntervalDomain::Bottom) => IntervalDomain::Bottom,
            (IntervalDomain::Top, d) | (d, IntervalDomain::Top) => d.clone(),

            (IntervalDomain::Constant(a), IntervalDomain::Constant(b)) if a == b => {
                IntervalDomain::Constant(*a)
            }
            (IntervalDomain::Constant(_a), IntervalDomain::Constant(_)) => {
                IntervalDomain::Bottom // Different constants
            }

            (
                IntervalDomain::Interval {
                    lower: l1,
                    upper: u1,
                },
                IntervalDomain::Interval {
                    lower: l2,
                    upper: u2,
                },
            ) => {
                let new_lower = max_endpoint(l1, l2);
                let new_upper = min_endpoint(u1, u2);

                // Check if interval is valid
                if is_valid_interval(&new_lower, &new_upper) {
                    IntervalDomain::Interval {
                        lower: new_lower,
                        upper: new_upper,
                    }
                } else {
                    IntervalDomain::Bottom
                }
            }

            // Conservative: fall back to Bottom for complex cases
            _ => IntervalDomain::Bottom,
        }
    }
}

/// Helper function to find minimum endpoint (broader lower bound)
fn min_endpoint<F: Copy + PartialOrd>(a: &Endpoint<F>, b: &Endpoint<F>) -> Endpoint<F> {
    match (a, b) {
        (Endpoint::Unbounded, _) | (_, Endpoint::Unbounded) => Endpoint::Unbounded,
        (Endpoint::Open(x), Endpoint::Open(y)) => Endpoint::Open(if x < y { *x } else { *y }),
        (Endpoint::Closed(x), Endpoint::Closed(y)) => Endpoint::Closed(if x < y { *x } else { *y }),
        (Endpoint::Open(x), Endpoint::Closed(y)) | (Endpoint::Closed(y), Endpoint::Open(x)) => {
            if x < y {
                Endpoint::Open(*x)
            } else if x == y {
                Endpoint::Open(*x) // Open is broader than closed at same point
            } else {
                Endpoint::Closed(*y)
            }
        }
    }
}

/// Helper function to find maximum endpoint (broader upper bound)
fn max_endpoint<F: Copy + PartialOrd>(a: &Endpoint<F>, b: &Endpoint<F>) -> Endpoint<F> {
    match (a, b) {
        (Endpoint::Unbounded, _) | (_, Endpoint::Unbounded) => Endpoint::Unbounded,
        (Endpoint::Open(x), Endpoint::Open(y)) => Endpoint::Open(if x > y { *x } else { *y }),
        (Endpoint::Closed(x), Endpoint::Closed(y)) => Endpoint::Closed(if x > y { *x } else { *y }),
        (Endpoint::Open(x), Endpoint::Closed(y)) | (Endpoint::Closed(y), Endpoint::Open(x)) => {
            if x > y {
                Endpoint::Open(*x)
            } else if x == y {
                Endpoint::Open(*x) // Open is broader than closed at same point
            } else {
                Endpoint::Closed(*y)
            }
        }
    }
}

/// Check if an interval with given endpoints is valid (non-empty)
fn is_valid_interval<F: Copy + PartialOrd>(lower: &Endpoint<F>, upper: &Endpoint<F>) -> bool {
    match (lower, upper) {
        (Endpoint::Unbounded, _) | (_, Endpoint::Unbounded) => true,
        (Endpoint::Open(a), Endpoint::Open(b)) => a < b,
        (Endpoint::Closed(a), Endpoint::Closed(b)) => a <= b,
        (Endpoint::Open(a), Endpoint::Closed(b)) | (Endpoint::Closed(a), Endpoint::Open(b)) => {
            a < b
        }
    }
}

impl<F: fmt::Display> fmt::Display for IntervalDomain<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntervalDomain::Bottom => write!(f, "⊥"),
            IntervalDomain::Top => write!(f, "ℝ"),
            IntervalDomain::Constant(c) => write!(f, "{{{c}}}"),
            IntervalDomain::Interval { lower, upper } => {
                let (left_bracket, lower_val) = match lower {
                    Endpoint::Unbounded => ("(-∞".to_string(), String::new()),
                    Endpoint::Open(val) => ("(".to_string(), format!("{val}")),
                    Endpoint::Closed(val) => ("[".to_string(), format!("{val}")),
                };
                let (upper_val, right_bracket) = match upper {
                    Endpoint::Unbounded => (String::new(), "+∞)".to_string()),
                    Endpoint::Open(val) => (format!("{val}"), ")".to_string()),
                    Endpoint::Closed(val) => (format!("{val}"), "]".to_string()),
                };

                match (lower, upper) {
                    (Endpoint::Unbounded, Endpoint::Unbounded) => write!(f, "ℝ"),
                    (Endpoint::Unbounded, _) => write!(f, "(-∞, {upper_val}{right_bracket}"),
                    (_, Endpoint::Unbounded) => write!(f, "{left_bracket}{lower_val}, +∞)"),
                    (_, _) => write!(f, "{left_bracket}{lower_val}, {upper_val}{right_bracket}"),
                }
            }
        }
    }
}

/// Domain analyzer using endpoint-based intervals
#[derive(Debug, Clone)]
pub struct IntervalDomainAnalyzer<F> {
    /// Variable domains (maps variable index to domain)
    variable_domains: HashMap<usize, IntervalDomain<F>>,
    /// Cache for computed expression domains
    expression_cache: HashMap<String, IntervalDomain<F>>,
    /// Zero value for this numeric type
    zero: F,
}

impl<F: Copy + PartialOrd + fmt::Display + fmt::Debug> IntervalDomainAnalyzer<F> {
    /// Create a new interval domain analyzer
    pub fn new(zero: F) -> Self {
        Self {
            variable_domains: HashMap::new(),
            expression_cache: HashMap::new(),
            zero,
        }
    }

    /// Set the domain for a variable
    pub fn set_variable_domain(&mut self, var_index: usize, domain: IntervalDomain<F>) {
        self.variable_domains.insert(var_index, domain);
    }

    /// Get the domain for a variable (defaults to Top if not set)
    pub fn get_variable_domain(&self, var_index: usize) -> IntervalDomain<F> {
        self.variable_domains
            .get(&var_index)
            .cloned()
            .unwrap_or(IntervalDomain::Top)
    }

    /// Analyze the domain of an expression
    pub fn analyze_domain(&mut self, expr: &ASTRepr<F>) -> IntervalDomain<F>
    where
        F: Into<f64> + From<f64>,
    {
        // Check cache first
        let expr_key = format!("{expr:?}");
        if let Some(cached) = self.expression_cache.get(&expr_key) {
            return cached.clone();
        }

        let domain = self.analyze_domain_impl(expr);
        self.expression_cache.insert(expr_key, domain.clone());
        domain
    }

    /// Internal implementation of domain analysis
    fn analyze_domain_impl(&mut self, expr: &ASTRepr<F>) -> IntervalDomain<F>
    where
        F: Into<f64> + From<f64>,
    {
        match expr {
            ASTRepr::Constant(value) => IntervalDomain::Constant(*value),

            ASTRepr::Variable(index) => self.get_variable_domain(*index),

            ASTRepr::Add(left, right) => {
                let left_domain = self.analyze_domain(left);
                let right_domain = self.analyze_domain(right);
                self.analyze_addition(&left_domain, &right_domain)
            }

            ASTRepr::Ln(inner) => {
                let inner_domain = self.analyze_domain(inner);
                self.analyze_logarithm(&inner_domain)
            }

            ASTRepr::Exp(inner) => {
                let inner_domain = self.analyze_domain(inner);
                self.analyze_exponential(&inner_domain)
            }

            // Add other operations as needed
            _ => IntervalDomain::Top, // Conservative for unimplemented operations
        }
    }

    /// Analyze addition domain
    fn analyze_addition(
        &self,
        left: &IntervalDomain<F>,
        right: &IntervalDomain<F>,
    ) -> IntervalDomain<F>
    where
        F: Into<f64> + From<f64>,
    {
        match (left, right) {
            (IntervalDomain::Bottom, _) | (_, IntervalDomain::Bottom) => IntervalDomain::Bottom,

            (IntervalDomain::Constant(a), IntervalDomain::Constant(b)) => {
                let sum: f64 = (*a).into() + (*b).into();
                IntervalDomain::Constant(F::from(sum))
            }

            // Positive + Positive = Positive
            (a, b) if a.is_positive(self.zero) && b.is_positive(self.zero) => {
                IntervalDomain::positive(self.zero)
            }

            // Conservative approximation for complex cases
            _ => IntervalDomain::Top,
        }
    }

    /// Analyze logarithm domain
    fn analyze_logarithm(&self, inner: &IntervalDomain<F>) -> IntervalDomain<F>
    where
        F: Into<f64> + From<f64>,
    {
        match inner {
            IntervalDomain::Bottom => IntervalDomain::Bottom,
            domain if !domain.is_positive(self.zero) => IntervalDomain::Bottom, // ln only defined for positive values
            IntervalDomain::Constant(x) if *x > self.zero => {
                let ln_val: f64 = (*x).into().ln();
                IntervalDomain::Constant(F::from(ln_val))
            }
            _ if inner.is_positive(self.zero) => IntervalDomain::Top, // ln of positive domain is all reals
            _ => IntervalDomain::Bottom,
        }
    }

    /// Analyze exponential domain
    fn analyze_exponential(&self, _inner: &IntervalDomain<F>) -> IntervalDomain<F> {
        // exp is always positive for any real input
        IntervalDomain::positive(self.zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_domain_creation() {
        let positive = IntervalDomain::positive(0.0);
        let non_negative = IntervalDomain::non_negative(0.0);

        assert!(positive.is_positive(0.0));
        assert!(!positive.contains(0.0));
        assert!(non_negative.is_non_negative(0.0));
        assert!(non_negative.contains(0.0));
    }

    #[test]
    fn test_domain_operations() {
        let pos = IntervalDomain::positive(0.0);
        let neg = IntervalDomain::negative(0.0);
        let joined = pos.join(&neg);

        // Positive ∪ Negative should cover all non-zero reals
        assert!(joined.contains(1.0));
        assert!(joined.contains(-1.0));

        // The join of (0,+∞) and (-∞,0) produces (-∞,+∞) which is equivalent to ℝ
        match &joined {
            IntervalDomain::Interval {
                lower: Endpoint::Unbounded,
                upper: Endpoint::Unbounded,
            } => {
                // This is correct - unbounded interval is equivalent to Top
            }
            IntervalDomain::Top => {
                // This would also be correct
            }
            other => panic!("Expected unbounded interval or Top, got: {other:?}"),
        }

        // Test that it contains zero (since it's the full real line)
        assert!(joined.contains(0.0));
    }

    #[test]
    fn test_interval_validity() {
        let valid = IntervalDomain::closed_interval(1.0, 5.0);
        let invalid = IntervalDomain::closed_interval(5.0, 1.0);

        assert!(matches!(valid, IntervalDomain::Interval { .. }));
        assert!(matches!(invalid, IntervalDomain::Bottom));
    }

    #[test]
    fn test_interval_domain_analysis() {
        let mut analyzer = IntervalDomainAnalyzer::new(0.0);

        // Set x to be positive
        analyzer.set_variable_domain(0, IntervalDomain::positive(0.0));

        // Test ln(x) where x > 0
        let x = ASTRepr::Variable(0);
        let ln_x = ASTRepr::Ln(Box::new(x));
        let domain = analyzer.analyze_domain(&ln_x);

        // ln of positive domain should be Top (all reals)
        assert!(matches!(domain, IntervalDomain::Top));
    }
}
