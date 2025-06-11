use dslcompile::{
    ast::{
        advanced::{pretty_ast},
        runtime::VariableRegistry,
        ASTRepr,
    },
    frunk::hlist,
    interval_domain::{IntervalDomain, IntervalDomainAnalyzer},
    symbolic::symbolic::SymbolicOptimizer,
    DSLCompileError,
}; 