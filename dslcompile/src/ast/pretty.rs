//! Pretty-printers for `ASTRepr` and `ANFExpr`
//!
//! These functions pretty-print *existing* expression trees (`ASTRepr`, `ANFExpr`),
//! using variable names from a `VariableRegistry`.
//!
//! This is different from the `PrettyPrint` final tagless instance, which builds up
//! a string as you construct an expression. These pretty-printers work on any tree,
//! regardless of how it was constructed (e.g., parsing, property-based generation, etc).

use crate::ast::ASTRepr;
use crate::ast::Scalar;
use crate::ast::runtime::typed_registry::VariableRegistry;
use crate::symbolic::anf::{ANFAtom, ANFComputation, ANFExpr};

/// Generate a pretty-printed string representation of an AST
pub fn pretty_ast<T>(ast: &ASTRepr<T>, registry: &VariableRegistry) -> String
where
    T: std::fmt::Display,
{
    match ast {
        ASTRepr::Variable(index) => {
            format!("x_{index}")
        }
        ASTRepr::Constant(value) => value.to_string(),
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
                "({})^({})",
                pretty_ast(base, registry),
                pretty_ast(exp, registry)
            )
        }
        ASTRepr::Neg(inner) => {
            format!("-({})", pretty_ast(inner, registry))
        }
        ASTRepr::Sin(inner) => {
            format!("sin({})", pretty_ast(inner, registry))
        }
        ASTRepr::Cos(inner) => {
            format!("cos({})", pretty_ast(inner, registry))
        }
        ASTRepr::Exp(inner) => {
            format!("exp({})", pretty_ast(inner, registry))
        }
        ASTRepr::Ln(inner) => {
            format!("ln({})", pretty_ast(inner, registry))
        }
        ASTRepr::Sqrt(inner) => {
            format!("sqrt({})", pretty_ast(inner, registry))
        }
        ASTRepr::Sum(_collection) => {
            // TODO: Pretty print Collection format
            "Σ(Collection)".to_string() // Placeholder until Collection pretty printing is implemented
        }
    }
}

/// Pretty-print an `ASTRepr` with proper indentation and newlines for complex expressions
pub fn pretty_ast_indented<T: Scalar>(expr: &ASTRepr<T>, registry: &VariableRegistry) -> String {
    pretty_ast_indented_impl(expr, registry, 0, false)
}

/// Internal implementation for indented pretty printing with depth tracking
fn pretty_ast_indented_impl<T: Scalar>(
    expr: &ASTRepr<T>,
    registry: &VariableRegistry,
    depth: usize,
    is_function_arg: bool,
) -> String {
    let indent = "  ".repeat(depth);
    let next_indent = "  ".repeat(depth + 1);

    match expr {
        // Simple expressions - no indentation needed
        ASTRepr::Constant(val) => format!("{val}"),
        ASTRepr::Variable(idx) => {
            if *idx < registry.len() {
                registry.debug_name(*idx)
            } else {
                format!("var_{idx}")
            }
        }

        // Binary operations - add newlines for complex sub-expressions
        ASTRepr::Add(left, right) => {
            let left_str = pretty_ast_indented_impl(left, registry, depth + 1, false);
            let right_str = pretty_ast_indented_impl(right, registry, depth + 1, false);

            if should_multiline(left, right) {
                format!("(\n{next_indent}{left_str} +\n{next_indent}{right_str}\n{indent})")
            } else {
                format!("({left_str} + {right_str})")
            }
        }

        ASTRepr::Sub(left, right) => {
            let left_str = pretty_ast_indented_impl(left, registry, depth + 1, false);
            let right_str = pretty_ast_indented_impl(right, registry, depth + 1, false);

            if should_multiline(left, right) {
                format!("(\n{next_indent}{left_str} -\n{next_indent}{right_str}\n{indent})")
            } else {
                format!("({left_str} - {right_str})")
            }
        }

        ASTRepr::Mul(left, right) => {
            let left_str = pretty_ast_indented_impl(left, registry, depth + 1, false);
            let right_str = pretty_ast_indented_impl(right, registry, depth + 1, false);

            if should_multiline(left, right) {
                format!("(\n{next_indent}{left_str} *\n{next_indent}{right_str}\n{indent})")
            } else {
                format!("({left_str} * {right_str})")
            }
        }

        ASTRepr::Div(left, right) => {
            let left_str = pretty_ast_indented_impl(left, registry, depth + 1, false);
            let right_str = pretty_ast_indented_impl(right, registry, depth + 1, false);

            if should_multiline(left, right) {
                format!("(\n{next_indent}{left_str} /\n{next_indent}{right_str}\n{indent})")
            } else {
                format!("({left_str} / {right_str})")
            }
        }

        ASTRepr::Pow(base, exp) => {
            let base_str = pretty_ast_indented_impl(base, registry, depth + 1, false);
            let exp_str = pretty_ast_indented_impl(exp, registry, depth + 1, false);

            if should_multiline(base, exp) {
                format!("(\n{next_indent}{base_str}^\n{next_indent}{exp_str}\n{indent})")
            } else {
                format!("({base_str}^{exp_str})")
            }
        }

        // Unary operations - function-style formatting
        ASTRepr::Neg(inner) => {
            let inner_str = pretty_ast_indented_impl(inner, registry, depth + 1, true);
            if is_complex_expr(inner) && !is_function_arg {
                format!("(-\n{next_indent}{inner_str})")
            } else {
                format!("(-{inner_str})")
            }
        }

        ASTRepr::Ln(inner) => {
            let inner_str = pretty_ast_indented_impl(inner, registry, depth + 1, true);
            if is_complex_expr(inner) && !is_function_arg {
                format!("ln(\n{next_indent}{inner_str}\n{indent})")
            } else {
                format!("ln({inner_str})")
            }
        }

        ASTRepr::Exp(inner) => {
            let inner_str = pretty_ast_indented_impl(inner, registry, depth + 1, true);
            if is_complex_expr(inner) && !is_function_arg {
                format!("exp(\n{next_indent}{inner_str}\n{indent})")
            } else {
                format!("exp({inner_str})")
            }
        }

        ASTRepr::Sin(inner) => {
            let inner_str = pretty_ast_indented_impl(inner, registry, depth + 1, true);
            if is_complex_expr(inner) && !is_function_arg {
                format!("sin(\n{next_indent}{inner_str}\n{indent})")
            } else {
                format!("sin({inner_str})")
            }
        }

        ASTRepr::Cos(inner) => {
            let inner_str = pretty_ast_indented_impl(inner, registry, depth + 1, true);
            if is_complex_expr(inner) && !is_function_arg {
                format!("cos(\n{next_indent}{inner_str}\n{indent})")
            } else {
                format!("cos({inner_str})")
            }
        }

        ASTRepr::Sqrt(inner) => {
            let inner_str = pretty_ast_indented_impl(inner, registry, depth + 1, true);
            if is_complex_expr(inner) && !is_function_arg {
                format!("sqrt(\n{next_indent}{inner_str}\n{indent})")
            } else {
                format!("sqrt({inner_str})")
            }
        }

        // Sum operation - special formatting for summations
        ASTRepr::Sum(_collection) => {
            // TODO: Pretty print Collection format with proper indentation
            format!("{indent}Σ(Collection)") // Placeholder until Collection pretty printing is implemented
        }
    }
}

/// Check if binary operation should be formatted across multiple lines
fn should_multiline<T: Scalar>(left: &ASTRepr<T>, right: &ASTRepr<T>) -> bool {
    // Use multiline if either operand is complex or if both are non-trivial
    is_complex_expr(left)
        || is_complex_expr(right)
        || (is_nontrivial_expr(left) && is_nontrivial_expr(right))
}

/// Check if expression is complex enough to warrant indentation
fn is_complex_expr<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => false,
        ASTRepr::Add(_, _)
        | ASTRepr::Sub(_, _)
        | ASTRepr::Mul(_, _)
        | ASTRepr::Div(_, _)
        | ASTRepr::Pow(_, _)
        | ASTRepr::Sum(_) => true,
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => is_nontrivial_expr(inner),
    }
}

/// Check if expression is non-trivial (not just constant or variable)
fn is_nontrivial_expr<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    !matches!(expr, ASTRepr::Constant(_) | ASTRepr::Variable(_))
}

/// Pretty-print an `ANFExpr` as indented let-bindings, using variable names from the registry.
pub fn pretty_anf<T: Scalar>(expr: &ANFExpr<T>, registry: &VariableRegistry) -> String {
    fn atom<T: Scalar>(a: &ANFAtom<T>, registry: &VariableRegistry) -> String {
        match a {
            ANFAtom::Constant(v) => format!("{v}"),
            ANFAtom::Variable(var_ref) => var_ref.debug_name(registry),
        }
    }
    fn comp<T: Scalar>(c: &ANFComputation<T>, registry: &VariableRegistry) -> String {
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
    fn go<T: Scalar>(e: &ANFExpr<T>, registry: &VariableRegistry, indent: usize) -> String {
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
