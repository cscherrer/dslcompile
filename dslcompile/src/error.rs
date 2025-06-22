//! Error types for `DSLCompile`
//!
//! This module defines the error types used throughout the `DSLCompile` library.

use std::fmt;

/// Result type alias for `DSLCompile` operations
pub type Result<T> = std::result::Result<T, DSLCompileError>;

/// Main error type for `DSLCompile` operations
#[derive(Debug, Clone)]
pub enum DSLCompileError {
    /// JIT compilation error (generic)
    JITError(String),

    /// Compilation error (Rust codegen)
    CompilationError(String),

    /// Optimization error
    #[cfg(feature = "optimization")]
    Optimization(String),

    /// Variable not found error
    VariableNotFound(String),

    /// Invalid expression error
    InvalidExpression(String),

    /// Numeric computation error
    NumericError(String),

    /// Domain analysis error (e.g., ln of negative number)
    DomainError(String),

    /// Feature not enabled error
    FeatureNotEnabled(String),

    /// Invalid input error
    InvalidInput(String),

    /// Unsupported operation or AST node type
    UnsupportedOperation(String),

    /// Invalid variable name format
    InvalidVariableName(String),

    /// Unsupported expression type in context
    UnsupportedExpression(String),

    /// Invalid binding in let expression or lambda
    InvalidBinding(String),

    /// Invalid lambda structure or usage
    InvalidLambda(String),

    /// Generic error with message (avoid using - prefer specific variants)
    Generic(String),
}

impl fmt::Display for DSLCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DSLCompileError::JITError(msg) => write!(f, "JIT compilation error: {msg}"),

            DSLCompileError::CompilationError(msg) => write!(f, "Compilation error: {msg}"),

            #[cfg(feature = "optimization")]
            DSLCompileError::Optimization(msg) => write!(f, "Optimization error: {msg}"),

            DSLCompileError::VariableNotFound(var) => write!(f, "Variable not found: {var}"),
            DSLCompileError::InvalidExpression(msg) => write!(f, "Invalid expression: {msg}"),
            DSLCompileError::NumericError(msg) => write!(f, "Numeric error: {msg}"),
            DSLCompileError::DomainError(msg) => write!(f, "Domain error: {msg}"),
            DSLCompileError::FeatureNotEnabled(feature) => {
                write!(f, "Feature not enabled: {feature}")
            }
            DSLCompileError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            DSLCompileError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {op}"),
            DSLCompileError::InvalidVariableName(name) => write!(f, "Invalid variable name: {name}"),
            DSLCompileError::UnsupportedExpression(expr) => write!(f, "Unsupported expression: {expr}"),
            DSLCompileError::InvalidBinding(details) => write!(f, "Invalid binding: {details}"),
            DSLCompileError::InvalidLambda(details) => write!(f, "Invalid lambda: {details}"),
            DSLCompileError::Generic(msg) => write!(f, "Error: {msg}"),
        }
    }
}

impl std::error::Error for DSLCompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

// JIT error conversion will be added when JIT support is implemented

impl From<String> for DSLCompileError {
    fn from(msg: String) -> Self {
        DSLCompileError::Generic(msg)
    }
}

impl From<&str> for DSLCompileError {
    fn from(msg: &str) -> Self {
        DSLCompileError::Generic(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display_formatting() {
        let variable_error = DSLCompileError::VariableNotFound("x".to_string());
        assert_eq!(variable_error.to_string(), "Variable not found: x");

        let invalid_expr_error = DSLCompileError::InvalidExpression("malformed".to_string());
        assert_eq!(
            invalid_expr_error.to_string(),
            "Invalid expression: malformed"
        );

        let numeric_error = DSLCompileError::NumericError("division by zero".to_string());
        assert_eq!(numeric_error.to_string(), "Numeric error: division by zero");

        let feature_error = DSLCompileError::FeatureNotEnabled("jit".to_string());
        assert_eq!(feature_error.to_string(), "Feature not enabled: jit");

        let generic_error = DSLCompileError::Generic("something went wrong".to_string());
        assert_eq!(generic_error.to_string(), "Error: something went wrong");
    }

    #[test]
    fn test_jit_error_display() {
        let jit_error = DSLCompileError::JITError("compilation failed".to_string());
        assert_eq!(
            jit_error.to_string(),
            "JIT compilation error: compilation failed"
        );
    }

    #[test]
    #[cfg(feature = "optimization")]
    fn test_optimization_error_display() {
        let opt_error = DSLCompileError::Optimization("optimization failed".to_string());
        assert_eq!(
            opt_error.to_string(),
            "Optimization error: optimization failed"
        );
    }

    #[test]
    fn test_error_source() {
        let error = DSLCompileError::Generic("test".to_string());
        assert!(error.source().is_none());
    }

    #[test]
    fn test_from_string_conversion() {
        let error: DSLCompileError = "test error".to_string().into();
        match error {
            DSLCompileError::Generic(msg) => assert_eq!(msg, "test error"),
            _ => panic!("Expected Generic error"),
        }
    }

    #[test]
    fn test_from_str_conversion() {
        let error: DSLCompileError = "test error".into();
        match error {
            DSLCompileError::Generic(msg) => assert_eq!(msg, "test error"),
            _ => panic!("Expected Generic error"),
        }
    }

    #[test]
    fn test_error_debug_formatting() {
        let error = DSLCompileError::VariableNotFound("x".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("VariableNotFound"));
        assert!(debug_str.contains('x'));
    }

    #[test]
    fn test_error_clone() {
        let original = DSLCompileError::NumericError("overflow".to_string());
        let cloned = original.clone();

        match (original, cloned) {
            (DSLCompileError::NumericError(msg1), DSLCompileError::NumericError(msg2)) => {
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
            Err(DSLCompileError::Generic("failed".to_string()))
        }

        let result = failing_function();
        assert!(result.is_err());
        match result.unwrap_err() {
            DSLCompileError::Generic(msg) => assert_eq!(msg, "failed"),
            _ => panic!("Expected Generic error"),
        }
    }
}
