//! Summation types for the unified sum() API
//!
//! This module provides the types needed for the unified summation API that can handle
//! both mathematical ranges and data vectors with the same interface.

/// Represents different types of summable ranges for the unified sum() API
#[derive(Debug, Clone)]
pub enum SummableRange {
    /// Mathematical range like 1..=10 for symbolic optimization
    MathematicalRange { start: i64, end: i64 },
    /// Data iteration for runtime values
    DataIteration { values: Vec<f64> },
}

/// Trait for converting different types into summable ranges
/// This enables the unified sum() API to handle both mathematical ranges and data vectors
pub trait IntoSummableRange {
    fn into_summable(self) -> SummableRange;
}

/// Implementation for mathematical ranges
impl IntoSummableRange for std::ops::RangeInclusive<i64> {
    fn into_summable(self) -> SummableRange {
        SummableRange::MathematicalRange {
            start: *self.start(),
            end: *self.end(),
        }
    }
}

/// Implementation for data vectors
impl IntoSummableRange for Vec<f64> {
    fn into_summable(self) -> SummableRange {
        SummableRange::DataIteration { values: self }
    }
}

/// Implementation for data slices
impl IntoSummableRange for &[f64] {
    fn into_summable(self) -> SummableRange {
        SummableRange::DataIteration {
            values: self.to_vec(),
        }
    }
}

/// Implementation for data vector references
impl IntoSummableRange for &Vec<f64> {
    fn into_summable(self) -> SummableRange {
        SummableRange::DataIteration {
            values: self.clone(),
        }
    }
} 