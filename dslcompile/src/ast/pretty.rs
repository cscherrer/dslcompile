//! Pretty-printers for `ASTRepr` and `ANFExpr`
//!
//! These functions pretty-print *existing* expression trees (`ASTRepr`, `ANFExpr`),
//! using variable names from a `VariableRegistry`.
//!
//! This is different from the `PrettyPrint` final tagless instance, which builds up
//! a string as you construct an expression. These pretty-printers work on any tree,
//! regardless of how it was constructed (e.g., parsing, property-based generation, etc).

use crate::final_tagless::{ASTRepr, NumericType, VariableRegistry};
use crate::symbolic::anf::{ANFAtom, ANFComputation, ANFExpr};

/// Pretty-print an `ASTRepr` using infix notation and variable names from the registry.
pub fn pretty_ast<T: NumericType>(expr: &ASTRepr<T>, registry: &VariableRegistry) -> String {
    match expr {
        ASTRepr::Constant(val) => format!("{val}"),
        ASTRepr::Variable(idx) => {
            if *idx < registry.len() {
                registry.debug_name(*idx)
            } else {
                format!("var_{idx}")
            }
        }
        ASTRepr::Add(left, right) => {
            format!(
                "({} + {})",
                pretty_ast(left, registry),
                pretty_ast(right, registry)
            )
        }
        ASTRepr::Sub(left, right) => {
            format!(
                "({} - {})",
                pretty_ast(left, registry),
                pretty_ast(right, registry)
            )
        }
        ASTRepr::Mul(left, right) => {
            format!(
                "({} * {})",
                pretty_ast(left, registry),
                pretty_ast(right, registry)
            )
        }
        ASTRepr::Div(left, right) => {
            format!(
                "({} / {})",
                pretty_ast(left, registry),
                pretty_ast(right, registry)
            )
        }
        ASTRepr::Pow(base, exp) => {
            format!(
                "({}^{})",
                pretty_ast(base, registry),
                pretty_ast(exp, registry)
            )
        }
        ASTRepr::Neg(inner) => format!("(-{})", pretty_ast(inner, registry)),
        ASTRepr::Ln(inner) => format!("ln({})", pretty_ast(inner, registry)),
        ASTRepr::Exp(inner) => format!("exp({})", pretty_ast(inner, registry)),
        ASTRepr::Sin(inner) => format!("sin({})", pretty_ast(inner, registry)),
        ASTRepr::Cos(inner) => format!("cos({})", pretty_ast(inner, registry)),
        ASTRepr::Sqrt(inner) => format!("sqrt({})", pretty_ast(inner, registry)),
    }
}

/// Pretty-print an `ANFExpr` as indented let-bindings, using variable names from the registry.
pub fn pretty_anf<T: NumericType>(expr: &ANFExpr<T>, registry: &VariableRegistry) -> String {
    fn atom<T: NumericType>(a: &ANFAtom<T>, registry: &VariableRegistry) -> String {
        match a {
            ANFAtom::Constant(v) => format!("{v}"),
            ANFAtom::Variable(var_ref) => var_ref.debug_name(registry),
        }
    }
    fn comp<T: NumericType>(c: &ANFComputation<T>, registry: &VariableRegistry) -> String {
        match c {
            ANFComputation::Add(a, b) => format!("{} + {}", atom(a, registry), atom(b, registry)),
            ANFComputation::Sub(a, b) => format!("{} - {}", atom(a, registry), atom(b, registry)),
            ANFComputation::Mul(a, b) => format!("{} * {}", atom(a, registry), atom(b, registry)),
            ANFComputation::Div(a, b) => format!("{} / {}", atom(a, registry), atom(b, registry)),
            ANFComputation::Pow(a, b) => format!("{}^{}", atom(a, registry), atom(b, registry)),
            ANFComputation::Neg(a) => format!("-{}", atom(a, registry)),
            ANFComputation::Ln(a) => format!("ln({})", atom(a, registry)),
            ANFComputation::Exp(a) => format!("exp({})", atom(a, registry)),
            ANFComputation::Sin(a) => format!("sin({})", atom(a, registry)),
            ANFComputation::Cos(a) => format!("cos({})", atom(a, registry)),
            ANFComputation::Sqrt(a) => format!("sqrt({})", atom(a, registry)),
        }
    }
    fn go<T: NumericType>(e: &ANFExpr<T>, registry: &VariableRegistry, indent: usize) -> String {
        let spaces = "  ".repeat(indent);
        match e {
            ANFExpr::Atom(a) => atom(a, registry),
            ANFExpr::Let(var, computation, body) => {
                format!(
                    "let {} = {} in\n{}{}",
                    var.debug_name(registry),
                    comp(computation, registry),
                    spaces,
                    go(body, registry, indent)
                )
            }
        }
    }
    go(expr, registry, 0)
}
