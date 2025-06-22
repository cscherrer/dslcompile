//! Compilation Backends for `DSLCompile`
//!
//! This module provides different compilation backends for mathematical expressions:
//! - **Rust Codegen**: Hot-loading compiled Rust dynamic libraries (primary backend)
//! - **Static Compiler**: Zero-overhead inline code generation
//! - **LLVM JIT**: Direct JIT compilation using LLVM (fastest execution)
//! - **Future backends**: GPU compilation, etc.

// Rust code generation and compilation backend (primary)
pub mod rust_codegen;

// Static compilation backend (zero overhead)
pub mod static_compiler;

// LLVM JIT compilation backend (maximum performance)
#[cfg(feature = "llvm_jit")]
pub mod llvm_jit;

// Re-export commonly used types from the Rust backend
pub use rust_codegen::{CompiledRustFunction, RustCodeGenerator, RustCompiler, RustOptLevel};

// Re-export static compilation types
pub use static_compiler::{StaticCompilable, StaticCompiler};

// Re-export LLVM JIT types when feature is enabled
#[cfg(feature = "llvm_jit")]
pub use llvm_jit::LLVMJITCompiler;

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
    /// Use LLVM JIT compilation (fastest execution)
    #[cfg(feature = "llvm_jit")]
    LlvmJit,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::RustHotLoad
    }
}
