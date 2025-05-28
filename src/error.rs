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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display_formatting() {
        let variable_error = MathJITError::VariableNotFound("x".to_string());
        assert_eq!(variable_error.to_string(), "Variable not found: x");

        let invalid_expr_error = MathJITError::InvalidExpression("malformed".to_string());
        assert_eq!(invalid_expr_error.to_string(), "Invalid expression: malformed");

        let numeric_error = MathJITError::NumericError("division by zero".to_string());
        assert_eq!(numeric_error.to_string(), "Numeric error: division by zero");

        let feature_error = MathJITError::FeatureNotEnabled("jit".to_string());
        assert_eq!(feature_error.to_string(), "Feature not enabled: jit");

        let generic_error = MathJITError::Generic("something went wrong".to_string());
        assert_eq!(generic_error.to_string(), "Error: something went wrong");
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_error_display() {
        let jit_error = MathJITError::JITError("compilation failed".to_string());
        assert_eq!(jit_error.to_string(), "JIT compilation error: compilation failed");
    }

    #[test]
    #[cfg(feature = "optimization")]
    fn test_optimization_error_display() {
        let opt_error = MathJITError::Optimization("optimization failed".to_string());
        assert_eq!(opt_error.to_string(), "Optimization error: optimization failed");
    }

    #[test]
    fn test_error_source() {
        let error = MathJITError::Generic("test".to_string());
        assert!(error.source().is_none());
    }

    #[test]
    fn test_from_string_conversion() {
        let error: MathJITError = "test error".to_string().into();
        match error {
            MathJITError::Generic(msg) => assert_eq!(msg, "test error"),
            _ => panic!("Expected Generic error"),
        }
    }

    #[test]
    fn test_from_str_conversion() {
        let error: MathJITError = "test error".into();
        match error {
            MathJITError::Generic(msg) => assert_eq!(msg, "test error"),
            _ => panic!("Expected Generic error"),
        }
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = MathJITError::VariableNotFound("x".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("VariableNotFound"));
        assert!(debug_str.contains("x"));
    }

    #[test]
    fn test_error_clone() {
        let original = MathJITError::NumericError("overflow".to_string());
        let cloned = original.clone();
        
        match (original, cloned) {
            (MathJITError::NumericError(msg1), MathJITError::NumericError(msg2)) => {
                assert_eq!(msg1, msg2);
            }
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_result_type_alias() {
        fn test_function() -> Result<i32> {
            Ok(42)
        }
        
        let result = test_function();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_error_case() {
        fn failing_function() -> Result<i32> {
            Err(MathJITError::Generic("failed".to_string()))
        }
        
        let result = failing_function();
        assert!(result.is_err());
        match result.unwrap_err() {
            MathJITError::Generic(msg) => assert_eq!(msg, "failed"),
            _ => panic!("Expected Generic error"),
        }
    }
}
