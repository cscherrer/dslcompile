//! Error types for `MathJIT`
//!
//! This module defines the error types used throughout the `MathJIT` library.

use std::fmt;

/// Result type alias for `MathJIT` operations
pub type Result<T> = std::result::Result<T, MathJITError>;

/// Main error type for `MathJIT` operations
#[derive(Debug, Clone)]
pub enum MathJITError {
    /// JIT compilation error
    #[cfg(feature = "jit")]
    JITError(String),

    /// Optimization error
    #[cfg(feature = "optimization")]
    Optimization(String),

    /// Variable not found error
    VariableNotFound(String),

    /// Invalid expression error
    InvalidExpression(String),

    /// Numeric computation error
    NumericError(String),

    /// Feature not enabled error
    FeatureNotEnabled(String),

    /// Generic error with message
    Generic(String),
}

impl fmt::Display for MathJITError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "jit")]
            MathJITError::JITError(msg) => write!(f, "JIT compilation error: {msg}"),

            #[cfg(feature = "optimization")]
            MathJITError::Optimization(msg) => write!(f, "Optimization error: {msg}"),

            MathJITError::VariableNotFound(var) => write!(f, "Variable not found: {var}"),
            MathJITError::InvalidExpression(msg) => write!(f, "Invalid expression: {msg}"),
            MathJITError::NumericError(msg) => write!(f, "Numeric error: {msg}"),
            MathJITError::FeatureNotEnabled(feature) => write!(f, "Feature not enabled: {feature}"),
            MathJITError::Generic(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl std::error::Error for MathJITError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

// JIT error conversion will be added when JIT support is implemented

impl From<String> for MathJITError {
    fn from(msg: String) -> Self {
        MathJITError::Generic(msg)
    }
}

impl From<&str> for MathJITError {
    fn from(msg: &str) -> Self {
        MathJITError::Generic(msg.to_string())
    }
}
