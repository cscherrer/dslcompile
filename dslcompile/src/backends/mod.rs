//! Compilation Backends for `DSLCompile`
//!
//! This module provides different compilation backends for mathematical expressions:
//! - **Rust Codegen**: Hot-loading compiled Rust dynamic libraries (primary backend)
//! - **Future backends**: LLVM, GPU compilation, etc.

// Rust code generation and compilation backend (primary)
pub mod rust_codegen;

// Re-export commonly used types from the Rust backend
pub use rust_codegen::{CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel};

/// Trait for compilation backends
pub trait CompilationBackend {
    /// The type representing a compiled function
    type CompiledFunction;
    /// The error type for compilation failures
    type Error;

    /// Compile an expression to a native function
    fn compile(
        &mut self,
        expr: &crate::ast::ASTRepr<f64>,
    ) -> Result<Self::CompiledFunction, Self::Error>;
}

/// Backend selection based on compilation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// Use Rust hot-loading compilation (primary and default)
    RustHotLoad,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::RustHotLoad
    }
}
