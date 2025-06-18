//! Pretty-printers for `ASTRepr`
//!
//! These functions pretty-print *existing* expression trees (`ASTRepr`),
//! using variable names from a `VariableRegistry`.
//!
//! This is different from the `PrettyPrint` final tagless instance, which builds up
//! a string as you construct an expression. These pretty-printers work on any tree,
//! regardless of how it was constructed (e.g., parsing, property-based generation, etc).

use crate::{
    ast::{ASTRepr, Scalar},
    contexts::VariableRegistry,
};

/// Generate a pretty-printed string representation of an AST
pub fn pretty_ast<T: Scalar>(ast: &ASTRepr<T>, registry: &VariableRegistry) -> String {
    match ast {
        ASTRepr::Variable(index) => {
            format!("x_{index}")
        }
        ASTRepr::Constant(value) => value.to_string(),
        ASTRepr::Add(terms) => {
            if terms.is_empty() {
                "0".to_string()
            } else if terms.len() == 1 {
                pretty_ast(terms.elements().next().unwrap(), registry)
            } else {
                let term_strs: Vec<String> = terms
                    .elements()
                    .map(|term| pretty_ast(term, registry))
                    .collect();
                format!("({})", term_strs.join(" + "))
            }
        }
        ASTRepr::Sub(left, right) => {
            format!(
                "({} - {})",
                pretty_ast(left, registry),
                pretty_ast(right, registry)
            )
        }
        ASTRepr::Mul(factors) => {
            if factors.is_empty() {
                "1".to_string()
            } else if factors.len() == 1 {
                pretty_ast(factors.elements().next().unwrap(), registry)
            } else {
                let factor_strs: Vec<String> = factors
                    .elements()
                    .map(|factor| pretty_ast(factor, registry))
                    .collect();
                format!("({})", factor_strs.join(" * "))
            }
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

        // Lambda expressions - format as λ(vars).body
        ASTRepr::Lambda(lambda) => {
            let var_list = if lambda.var_indices.is_empty() {
                "_".to_string()
            } else {
                lambda
                    .var_indices
                    .iter()
                    .map(|idx| format!("x_{idx}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!("λ({}).{}", var_list, pretty_ast(&lambda.body, registry))
        }

        ASTRepr::BoundVar(index) => {
            format!("bound_{index}")
        }
        ASTRepr::Let(binding_id, expr, body) => {
            let expr_str = pretty_ast(expr, registry);
            let body_str = pretty_ast(body, registry);
            format!("let bound_{binding_id} = {expr_str} in {body_str}")
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
        ASTRepr::Add(terms) => {
            if terms.is_empty() {
                "0".to_string()
            } else if terms.len() == 1 {
                pretty_ast_indented_impl(terms.elements().next().unwrap(), registry, depth, is_function_arg)
            } else {
                let term_strs: Vec<String> = terms
                    .elements()
                    .map(|term| pretty_ast_indented_impl(term, registry, depth + 1, false))
                    .collect();

                let has_complex = terms.elements().any(|t| is_complex_expr(t));
                if has_complex {
                    format!(
                        "(\n{}{}\n{})",
                        next_indent,
                        term_strs.join(&format!(" +\n{next_indent}")),
                        indent
                    )
                } else {
                    format!("({})", term_strs.join(" + "))
                }
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

        ASTRepr::Mul(factors) => {
            if factors.is_empty() {
                "1".to_string()
            } else if factors.len() == 1 {
                pretty_ast_indented_impl(factors.elements().next().unwrap(), registry, depth, is_function_arg)
            } else {
                let factor_strs: Vec<String> = factors
                    .elements()
                    .map(|factor| pretty_ast_indented_impl(factor, registry, depth + 1, false))
                    .collect();

                let has_complex = factors.elements().any(|f| is_complex_expr(f));
                if has_complex {
                    format!(
                        "(\n{}{}\n{})",
                        next_indent,
                        factor_strs.join(&format!(" *\n{next_indent}")),
                        indent
                    )
                } else {
                    format!("({})", factor_strs.join(" * "))
                }
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

        // Lambda expressions - format with proper indentation
        ASTRepr::Lambda(lambda) => {
            let var_list = if lambda.var_indices.is_empty() {
                "_".to_string()
            } else {
                lambda
                    .var_indices
                    .iter()
                    .map(|idx| format!("x_{idx}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let body_str = pretty_ast_indented_impl(&lambda.body, registry, depth + 1, false);
            if is_complex_expr(&lambda.body) && !is_function_arg {
                format!("λ({var_list}).\n{next_indent}{body_str}")
            } else {
                format!("λ({var_list}).{body_str}")
            }
        }

        ASTRepr::BoundVar(index) => {
            format!("bound_{index}")
        }
        ASTRepr::Let(binding_id, expr, body) => {
            let expr_str = pretty_ast_indented_impl(expr, registry, depth + 1, false);
            let body_str = pretty_ast_indented_impl(body, registry, depth + 1, false);
            if is_complex_expr(expr) || is_complex_expr(body) {
                format!(
                    "let bound_{binding_id} =\n{next_indent}{expr_str}\n{indent}in\n{next_indent}{body_str}"
                )
            } else {
                format!("let bound_{binding_id} = {expr_str} in {body_str}")
            }
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
        ASTRepr::Add(_)
        | ASTRepr::Sub(_, _)
        | ASTRepr::Mul(_)
        | ASTRepr::Div(_, _)
        | ASTRepr::Pow(_, _)
        | ASTRepr::Sum(_) => true,
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => is_nontrivial_expr(inner),

        // Lambda expressions are complex if their body is complex
        ASTRepr::Lambda(lambda) => is_complex_expr(&lambda.body),

        ASTRepr::BoundVar(_) => {
            // BoundVar behaves like Variable for complexity analysis
            false
        }
        ASTRepr::Let(_, expr, body) => {
            // Let expressions are complex if either the bound expression or body is complex
            is_complex_expr(expr) || is_complex_expr(body)
        }
    }
}

/// Check if expression is non-trivial (not just constant or variable)
fn is_nontrivial_expr<T: Scalar>(expr: &ASTRepr<T>) -> bool {
    !matches!(expr, ASTRepr::Constant(_) | ASTRepr::Variable(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contexts::VariableRegistry;

    #[test]
    fn test_basic_pretty_printing() {
        let registry = VariableRegistry::new();
        // Test constants
        let const_expr = ASTRepr::<f64>::Constant(42.0);
        assert_eq!(pretty_ast(&const_expr, &registry), "42");

        let const_expr_float = ASTRepr::<f64>::Constant(1.23456);
        assert_eq!(pretty_ast(&const_expr_float, &registry), "1.23456");

        // Test variables
        let var_expr = ASTRepr::<f64>::Variable(0);
        assert_eq!(pretty_ast(&var_expr, &registry), "x_0");

        let var_expr_1 = ASTRepr::<f64>::Variable(1);
        assert_eq!(pretty_ast(&var_expr_1, &registry), "x_1");
    }

    #[test]
    fn test_arithmetic_operations_pretty_printing() {
        let registry = VariableRegistry::new();
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let const_2 = ASTRepr::<f64>::Constant(2.0);

        // Test addition
        use crate::ast::multiset::MultiSet;
        let add_expr = ASTRepr::Add(MultiSet::from_iter([x.clone(), y.clone()]));
        assert_eq!(pretty_ast(&add_expr, &registry), "(x_0 + x_1)");

        // Test subtraction
        let sub_expr = ASTRepr::Sub(Box::new(x.clone()), Box::new(const_2.clone()));
        assert_eq!(pretty_ast(&sub_expr, &registry), "(x_0 - 2)");

        // Test multiplication
        let mul_expr = ASTRepr::Mul(MultiSet::from_iter([const_2.clone(), x.clone()]));
        assert_eq!(pretty_ast(&mul_expr, &registry), "(2 * x_0)");

        // Test division
        let div_expr = ASTRepr::Div(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(pretty_ast(&div_expr, &registry), "(x_0 / x_1)");

        // Test negation
        let neg_expr = ASTRepr::Neg(Box::new(x.clone()));
        assert_eq!(pretty_ast(&neg_expr, &registry), "-(x_0)");
    }

    #[test]
    fn test_transcendental_functions_pretty_printing() {
        let registry = VariableRegistry::new();
        let x = ASTRepr::<f64>::Variable(0);

        // Test sin
        let sin_expr = ASTRepr::Sin(Box::new(x.clone()));
        assert_eq!(pretty_ast(&sin_expr, &registry), "sin(x_0)");

        let cos_expr = ASTRepr::Cos(Box::new(x.clone()));
        assert_eq!(pretty_ast(&cos_expr, &registry), "cos(x_0)");

        // Note: Tan is not available in ASTRepr, so we skip that test

        // Test exp
        let exp_expr = ASTRepr::Exp(Box::new(x.clone()));
        assert_eq!(pretty_ast(&exp_expr, &registry), "exp(x_0)");

        let ln_expr = ASTRepr::Ln(Box::new(x.clone()));
        assert_eq!(pretty_ast(&ln_expr, &registry), "ln(x_0)");

        // Test power
        let const_2 = ASTRepr::<f64>::Constant(2.0);
        let pow_expr = ASTRepr::Pow(Box::new(x.clone()), Box::new(const_2));
        assert_eq!(pretty_ast(&pow_expr, &registry), "(x_0)^(2)");
    }

    #[test]
    fn test_complex_nested_expressions() {
        use crate::ast::multiset::MultiSet;
        let registry = VariableRegistry::new();
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let const_2 = ASTRepr::<f64>::Constant(2.0);

        // Test sin(x + y) * 2
        let add_expr = ASTRepr::Add(MultiSet::from_iter([x.clone(), y.clone()]));
        let sin_expr = ASTRepr::Sin(Box::new(add_expr));
        let complex_expr = sin_expr * const_2;
        assert_eq!(
            pretty_ast(&complex_expr, &registry),
            "(sin((x_0 + x_1)) * 2)"
        );

        // Test exp(ln(x) + cos(y))
        let ln_x = ASTRepr::Ln(Box::new(x.clone()));
        let cos_y = ASTRepr::Cos(Box::new(y.clone()));
        let add_ln_cos = ln_x + cos_y;
        let exp_complex = ASTRepr::Exp(Box::new(add_ln_cos));
        assert_eq!(
            pretty_ast(&exp_complex, &registry),
            "exp((ln(x_0) + cos(x_1)))"
        );
    }

    #[test]
    fn test_power_expressions() {
        use crate::ast::multiset::MultiSet;
        let registry = VariableRegistry::new();
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let const_2 = ASTRepr::<f64>::Constant(2.0);

        // Test x^2
        let x_squared = ASTRepr::Pow(Box::new(x.clone()), Box::new(const_2.clone()));
        assert_eq!(pretty_ast(&x_squared, &registry), "(x_0)^(2)");

        // Test x^y
        let x_pow_y = ASTRepr::Pow(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(pretty_ast(&x_pow_y, &registry), "(x_0)^(x_1)");

        // Test (x + 1)^2
        let const_1 = ASTRepr::<f64>::Constant(1.0);
        let x_plus_1 = ASTRepr::Add(MultiSet::from_iter([x.clone(), const_1]));
        let nested_pow = ASTRepr::Pow(Box::new(x_plus_1), Box::new(const_2));
        assert_eq!(pretty_ast(&nested_pow, &registry), "((x_0 + 1))^(2)");
    }

    #[test]
    fn test_special_values() {
        let registry = VariableRegistry::new();

        let zero = ASTRepr::<f64>::Constant(0.0);
        assert_eq!(pretty_ast(&zero, &registry), "0");

        let one = ASTRepr::<f64>::Constant(1.0);
        assert_eq!(pretty_ast(&one, &registry), "1");

        let neg_val = ASTRepr::<f64>::Constant(-1.23456);
        assert_eq!(pretty_ast(&neg_val, &registry), "-1.23456");

        let small = ASTRepr::<f64>::Constant(1e-10);
        assert_eq!(pretty_ast(&small, &registry), "0.0000000001");
    }

    #[test]
    fn test_operator_precedence_in_pretty_printing() {
        let registry = VariableRegistry::new();
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let z = ASTRepr::<f64>::Variable(2);

        // Test x + y * z (should show precedence with parentheses)
        use crate::ast::multiset::MultiSet;
        let mul_y_z = ASTRepr::Mul(MultiSet::from_iter([y.clone(), z.clone()]));
        let add_x_mul = ASTRepr::Add(MultiSet::from_iter([x.clone(), mul_y_z]));
        assert_eq!(pretty_ast(&add_x_mul, &registry), "(x_0 + (x_1 * x_2))");

        // Test (x + y) * z
        let add_x_y = ASTRepr::Add(MultiSet::from_iter([x.clone(), y.clone()]));
        let mul_add_z = ASTRepr::Mul(MultiSet::from_iter([add_x_y, z.clone()]));
        assert_eq!(pretty_ast(&mul_add_z, &registry), "((x_0 + x_1) * x_2)");
    }

    #[test]
    fn test_variable_indexing() {
        let registry = VariableRegistry::new();

        // Test various variable indices
        for i in 0..10 {
            let var = ASTRepr::<f64>::Variable(i);
            assert_eq!(pretty_ast(&var, &registry), format!("x_{i}"));
        }

        // Test high index
        let high_var = ASTRepr::<f64>::Variable(999);
        assert_eq!(pretty_ast(&high_var, &registry), "x_999");
    }

    #[test]
    fn test_edge_cases() {
        let registry = VariableRegistry::new();
        let x = ASTRepr::<f64>::Variable(0);

        // Test double negation
        let neg_x = ASTRepr::Neg(Box::new(x.clone()));
        let double_neg = ASTRepr::Neg(Box::new(neg_x));
        assert_eq!(pretty_ast(&double_neg, &registry), "-(-(x_0))");

        // Test nested functions
        let sin_x = ASTRepr::Sin(Box::new(x.clone()));
        let cos_sin_x = ASTRepr::Cos(Box::new(sin_x));
        assert_eq!(pretty_ast(&cos_sin_x, &registry), "cos(sin(x_0))");

        // Test nested powers
        let const_2 = ASTRepr::<f64>::Constant(2.0);
        let const_3 = ASTRepr::<f64>::Constant(3.0);
        let x_pow_2 = ASTRepr::Pow(Box::new(x.clone()), Box::new(const_2));
        let pow_of_pow = ASTRepr::Pow(Box::new(x_pow_2), Box::new(const_3));
        assert_eq!(pretty_ast(&pow_of_pow, &registry), "((x_0)^(2))^(3)");
    }

    #[test]
    fn test_floating_point_formatting() {
        let registry = VariableRegistry::new();

        let int_float = ASTRepr::<f64>::Constant(5.0);
        assert_eq!(pretty_ast(&int_float, &registry), "5");

        let decimal_float = ASTRepr::<f64>::Constant(3.14);
        assert_eq!(pretty_ast(&decimal_float, &registry), "3.14");

        let precise_float = ASTRepr::<f64>::Constant(2.718281828459045);
        assert_eq!(pretty_ast(&precise_float, &registry), "2.718281828459045");
    }
}
