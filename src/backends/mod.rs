//! Compilation Backends for `MathJIT`
//!
//! This module provides different compilation backends for mathematical expressions:
//! - **Rust Codegen**: Hot-loading compiled Rust dynamic libraries (primary backend)
//! - **Cranelift JIT**: Fast JIT compilation using Cranelift (optional)
//! - **Future backends**: LLVM, GPU compilation, etc.

// Rust code generation and compilation backend (primary)
pub mod rust_codegen;

// Cranelift JIT backend (optional)
#[cfg(feature = "cranelift")]
pub mod cranelift;

// Re-export commonly used types from each backend
pub use rust_codegen::{CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel};

#[cfg(feature = "cranelift")]
pub use cranelift::{CompilationStats, JITCompiler, JITFunction, JITSignature};

/// Trait for compilation backends
pub trait CompilationBackend {
    /// The type representing a compiled function
    type CompiledFunction;
    /// The error type for compilation failures
    type Error;

    /// Compile an expression to a native function
    fn compile(
        &mut self,
        expr: &crate::final_tagless::ASTRepr<f64>,
    ) -> Result<Self::CompiledFunction, Self::Error>;
}

/// Backend selection based on compilation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// Use Rust hot-loading compilation (primary)
    RustHotLoad,
    /// Use Cranelift JIT compilation (optional)
    #[cfg(feature = "cranelift")]
    Cranelift,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::RustHotLoad
    }
}
