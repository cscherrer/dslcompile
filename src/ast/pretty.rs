//! Pretty-printers for `ASTRepr` and `ANFExpr`
//!
//! These functions pretty-print *existing* expression trees (`ASTRepr`, `ANFExpr`),
//! using variable names from a `VariableRegistry`.
//!
//! This is different from the `PrettyPrint` final tagless instance, which builds up
//! a string as you construct an expression. These pretty-printers work on any tree,
//! regardless of how it was constructed (e.g., parsing, property-based generation, etc).

use crate::final_tagless::{ASTRepr, NumericType, VariableRegistry};
use crate::symbolic::anf::{ANFAtom, ANFComputation, ANFExpr, VarRef};

/// Pretty-print an `ASTRepr` using infix notation and variable names from the registry.
pub fn pretty_ast<T: NumericType>(expr: &ASTRepr<T>, registry: &VariableRegistry) -> String {
    match expr {
        ASTRepr::Constant(val) => format!("{val}"),
        ASTRepr::Variable(idx) => registry.get_name(*idx).unwrap_or("?").to_string(),
        ASTRepr::Add(a, b) => format!(
            "({} + {})",
            pretty_ast(a, registry),
            pretty_ast(b, registry)
        ),
        ASTRepr::Sub(a, b) => format!(
            "({} - {})",
            pretty_ast(a, registry),
            pretty_ast(b, registry)
        ),
        ASTRepr::Mul(a, b) => format!(
            "({} * {})",
            pretty_ast(a, registry),
            pretty_ast(b, registry)
        ),
        ASTRepr::Div(a, b) => format!(
            "({} / {})",
            pretty_ast(a, registry),
            pretty_ast(b, registry)
        ),
        ASTRepr::Pow(a, b) => format!(
            "({} ^ {})",
            pretty_ast(a, registry),
            pretty_ast(b, registry)
        ),
        ASTRepr::Neg(a) => format!("-{}", pretty_ast(a, registry)),
        ASTRepr::Ln(a) => format!("ln({})", pretty_ast(a, registry)),
        ASTRepr::Exp(a) => format!("exp({})", pretty_ast(a, registry)),
        ASTRepr::Sqrt(a) => format!("sqrt({})", pretty_ast(a, registry)),
        ASTRepr::Sin(a) => format!("sin({})", pretty_ast(a, registry)),
        ASTRepr::Cos(a) => format!("cos({})", pretty_ast(a, registry)),
    }
}

/// Pretty-print an `ANFExpr` as indented let-bindings, using variable names from the registry.
pub fn pretty_anf<T: NumericType>(expr: &ANFExpr<T>, registry: &VariableRegistry) -> String {
    fn atom<T: NumericType>(a: &ANFAtom<T>, registry: &VariableRegistry) -> String {
        match a {
            ANFAtom::Constant(val) => format!("{val}"),
            ANFAtom::Variable(var) => match var {
                VarRef::User(idx) => registry.get_name(*idx).unwrap_or("?").to_string(),
                VarRef::Bound(id) => format!("t{id}"),
            },
        }
    }
    fn comp<T: NumericType>(c: &ANFComputation<T>, registry: &VariableRegistry) -> String {
        use ANFComputation::{Add, Cos, Div, Exp, Ln, Mul, Neg, Pow, Sin, Sqrt, Sub};
        match c {
            Add(a, b) => format!("{} + {}", atom(a, registry), atom(b, registry)),
            Sub(a, b) => format!("{} - {}", atom(a, registry), atom(b, registry)),
            Mul(a, b) => format!("{} * {}", atom(a, registry), atom(b, registry)),
            Div(a, b) => format!("{} / {}", atom(a, registry), atom(b, registry)),
            Pow(a, b) => format!("{} ^ {}", atom(a, registry), atom(b, registry)),
            Neg(a) => format!("-{}", atom(a, registry)),
            Ln(a) => format!("ln({})", atom(a, registry)),
            Exp(a) => format!("exp({})", atom(a, registry)),
            Sin(a) => format!("sin({})", atom(a, registry)),
            Cos(a) => format!("cos({})", atom(a, registry)),
            Sqrt(a) => format!("sqrt({})", atom(a, registry)),
        }
    }
    fn go<T: NumericType>(e: &ANFExpr<T>, registry: &VariableRegistry, indent: usize) -> String {
        let pad = |n| "  ".repeat(n);
        match e {
            ANFExpr::Atom(a) => format!("{}{}", pad(indent), atom(a, registry)),
            ANFExpr::Let(var, c, body) => {
                let vname = match var {
                    VarRef::User(idx) => registry.get_name(*idx).unwrap_or("?").to_string(),
                    VarRef::Bound(id) => format!("t{id}"),
                };
                let comp_str = comp(c, registry);
                let body_str = go(body, registry, indent + 1);
                format!("{}let {} = {}\n{}", pad(indent), vname, comp_str, body_str)
            }
        }
    }
    go(expr, registry, 0)
}
