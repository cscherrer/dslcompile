//! Mathematical Summation Support
//!
//! This module provides basic summation support types. The main summation functionality
//! has moved to `DynamicContext.sum()` which provides proven performance optimizations.

use crate::ast::ASTRepr;
use crate::error::Result;

/// Integer range for mathematical summations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntRange {
    start: i64,
    end: i64,
}

impl IntRange {
    #[must_use]
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    pub fn iter(&self) -> impl Iterator<Item = i64> {
        self.start..=self.end
    }

    #[must_use]
    pub fn len(&self) -> usize {
        if self.end >= self.start {
            (self.end - self.start + 1) as usize
        } else {
            0
        }
    }

    #[must_use]
    pub fn start(&self) -> i64 {
        self.start
    }

    #[must_use]
    pub fn end(&self) -> i64 {
        self.end
    }
}

/// Placeholder for `DirectEval` type (was in `final_tagless`)
pub struct DirectEval;

impl DirectEval {
    #[must_use]
    pub fn eval_with_vars(expr: &ASTRepr<f64>, vars: &[f64]) -> f64 {
        expr.eval_with_vars(vars)
    }
}
