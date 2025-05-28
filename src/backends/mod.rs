//! Compilation Backends for `MathJIT`
//!
//! This module provides different compilation backends for mathematical expressions:
//! - **Cranelift JIT**: Fast JIT compilation using Cranelift
//! - **Rust Codegen**: Hot-loading compiled Rust dynamic libraries
//! - **Future backends**: LLVM, GPU compilation, etc.

// Cranelift JIT backend
#[cfg(feature = "jit")]
pub mod cranelift;

// Rust code generation and compilation backend
pub mod rust_codegen;

// Re-export commonly used types from each backend
#[cfg(feature = "jit")]
pub use cranelift::{CompilationStats, JITCompiler, JITFunction, JITSignature};

pub use rust_codegen::{RustCodeGenerator, RustCompiler};

/// Trait for compilation backends
pub trait CompilationBackend {
    /// The type representing a compiled function
    type CompiledFunction;
    /// The error type for compilation failures
    type Error;

    /// Compile an expression to a native function
    fn compile(
        &mut self,
        expr: &crate::final_tagless::JITRepr<f64>,
    ) -> Result<Self::CompiledFunction, Self::Error>;
}

/// Backend selection based on compilation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// Use Cranelift JIT compilation
    Cranelift,
    /// Use Rust hot-loading compilation
    RustHotLoad,
}

impl Default for BackendType {
    fn default() -> Self {
        #[cfg(feature = "jit")]
        return Self::Cranelift;

        #[cfg(not(feature = "jit"))]
        return Self::RustHotLoad;
    }
}
