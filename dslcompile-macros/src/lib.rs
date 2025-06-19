//! Procedural macros for compile-time mathematical optimization
//!
//! This crate provided procedural macros for compile-time optimization.
//! Mathematical expressions can be simplified at compile time for better performance.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, Ident, Token, parse_macro_input};

// Note: Compile-time optimization using egg is handled at runtime
// This module provides basic compile-time mathematical simplifications

/// Procedural macro for compile-time optimization with scoped variables
///
/// This macro uses the modern scoped variables syntax for cleaner, more natural expressions.
/// The macro automatically detects variable dependencies and generates optimized code.
///
/// # Example
/// ```ignore
/// let result = optimize_math!(a, b, c; a.sin().mul(b).add(c.pow(2.0)));
/// ```
#[proc_macro]
pub fn optimize_math(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as OptimizeMathInput);

    // Convert the expression to our internal AST
    let ast = match expr_to_ast(&input.expr, &input.vars) {
        Ok(ast) => ast,
        Err(e) => {
            return syn::Error::new_spanned(&input.expr, e)
                .to_compile_error()
                .into();
        }
    };

    // Run basic compile-time optimization
    let optimized_ast = apply_basic_optimizations(&ast);

    // Generate direct Rust code
    let generated_code = ast_to_rust_expr(&optimized_ast, &input.vars);

    // Return the optimized expression
    generated_code.into()
}

/// Input structure for the compile-time optimization macro
struct OptimizeMathInput {
    vars: Vec<Ident>,
    expr: Expr,
}

impl syn::parse::Parse for OptimizeMathInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut vars = Vec::new();

        // Parse variable list before semicolon
        loop {
            if input.peek(Token![;]) {
                break;
            }

            vars.push(input.parse::<Ident>()?);

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else if input.peek(Token![;]) {
                break;
            } else {
                return Err(input.error("Expected ',' or ';'"));
            }
        }

        // Parse semicolon
        input.parse::<Token![;]>()?;

        // Parse expression
        let expr = input.parse::<Expr>()?;

        Ok(OptimizeMathInput { vars, expr })
    }
}

/// Internal AST representation for compile-time optimization
#[derive(Debug, Clone)]
enum CompileTimeAST {
    Variable(usize),
    Constant(f64),
    Add(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Mul(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Sub(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Sin(Box<CompileTimeAST>),
    Cos(Box<CompileTimeAST>),
    Exp(Box<CompileTimeAST>),
    Ln(Box<CompileTimeAST>),
    Pow(Box<CompileTimeAST>, Box<CompileTimeAST>),
}

impl CompileTimeAST {
    // Note: Egglog-specific conversion removed - optimization now handled at runtime with egg
}

/// Convert Rust expression to our internal AST representation
fn expr_to_ast(expr: &Expr, vars: &[Ident]) -> Result<CompileTimeAST, String> {
    match expr {
        // Method calls like var::<0>().sin().add(...)
        Expr::MethodCall(method_call) => {
            let receiver_ast = expr_to_ast(&method_call.receiver, vars)?;

            match method_call.method.to_string().as_str() {
                "add" => {
                    if method_call.args.len() != 1 {
                        return Err("add() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0], vars)?;
                    Ok(CompileTimeAST::Add(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "mul" => {
                    if method_call.args.len() != 1 {
                        return Err("mul() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0], vars)?;
                    Ok(CompileTimeAST::Mul(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "sub" => {
                    if method_call.args.len() != 1 {
                        return Err("sub() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0], vars)?;
                    Ok(CompileTimeAST::Sub(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "pow" => {
                    if method_call.args.len() != 1 {
                        return Err("pow() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0], vars)?;
                    Ok(CompileTimeAST::Pow(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "sin" => {
                    if !method_call.args.is_empty() {
                        return Err("sin() takes no arguments".to_string());
                    }
                    Ok(CompileTimeAST::Sin(Box::new(receiver_ast)))
                }
                "cos" => {
                    if !method_call.args.is_empty() {
                        return Err("cos() takes no arguments".to_string());
                    }
                    Ok(CompileTimeAST::Cos(Box::new(receiver_ast)))
                }
                "exp" => {
                    if !method_call.args.is_empty() {
                        return Err("exp() takes no arguments".to_string());
                    }
                    Ok(CompileTimeAST::Exp(Box::new(receiver_ast)))
                }
                "ln" => {
                    if !method_call.args.is_empty() {
                        return Err("ln() takes no arguments".to_string());
                    }
                    Ok(CompileTimeAST::Ln(Box::new(receiver_ast)))
                }
                _ => Err(format!("Unsupported method: {}", method_call.method)),
            }
        }

        // Variable references
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                if let Some(pos) = vars.iter().position(|v| v == ident) {
                    Ok(CompileTimeAST::Variable(pos))
                } else {
                    Err(format!("Unknown variable: {ident}"))
                }
            } else {
                Err("Complex paths not supported".to_string())
            }
        }

        // Literal numbers
        Expr::Lit(lit) => {
            if let syn::Lit::Float(float_lit) = &lit.lit {
                let value: f64 = float_lit
                    .base10_parse()
                    .map_err(|_| "Invalid float literal".to_string())?;
                Ok(CompileTimeAST::Constant(value))
            } else if let syn::Lit::Int(int_lit) = &lit.lit {
                let value: f64 = int_lit
                    .base10_parse()
                    .map_err(|_| "Invalid integer literal".to_string())?;
                Ok(CompileTimeAST::Constant(value))
            } else {
                Err("Only numeric literals supported".to_string())
            }
        }

        // Macro calls like var::<0>()
        Expr::Macro(macro_call) => {
            let macro_name = macro_call
                .mac
                .path
                .segments
                .last()
                .ok_or("Empty macro path")?
                .ident
                .to_string();

            if macro_name == "var" {
                // Parse var::<N>() to get variable index
                let tokens = macro_call.mac.tokens.to_string();
                let cleaned = tokens.trim_start_matches("::< ").trim_end_matches(" >");

                if let Ok(index) = cleaned.parse::<usize>() {
                    if index < vars.len() {
                        Ok(CompileTimeAST::Variable(index))
                    } else {
                        Err(format!("Variable index {index} out of range"))
                    }
                } else {
                    Err("Invalid variable index in var::<>()".to_string())
                }
            } else {
                Err("Unsupported macro".to_string())
            }
        }

        _ => Err("Unsupported expression type".to_string()),
    }
}

/// Apply basic optimization rules
fn apply_basic_optimizations(ast: &CompileTimeAST) -> CompileTimeAST {
    match ast {
        // x + 0 -> x
        CompileTimeAST::Add(left, right) => {
            let left_opt = apply_basic_optimizations(left);
            let right_opt = apply_basic_optimizations(right);

            if let CompileTimeAST::Constant(0.0) = right_opt {
                left_opt
            } else if let CompileTimeAST::Constant(0.0) = left_opt {
                right_opt
            } else {
                CompileTimeAST::Add(Box::new(left_opt), Box::new(right_opt))
            }
        }
        // x * 1 -> x, x * 0 -> 0
        CompileTimeAST::Mul(left, right) => {
            let left_opt = apply_basic_optimizations(left);
            let right_opt = apply_basic_optimizations(right);

            if let CompileTimeAST::Constant(1.0) = right_opt {
                left_opt
            } else if let CompileTimeAST::Constant(1.0) = left_opt {
                right_opt
            } else if let CompileTimeAST::Constant(0.0) = right_opt {
                CompileTimeAST::Constant(0.0)
            } else if let CompileTimeAST::Constant(0.0) = left_opt {
                CompileTimeAST::Constant(0.0)
            } else {
                CompileTimeAST::Mul(Box::new(left_opt), Box::new(right_opt))
            }
        }
        // ln(exp(x)) -> x
        CompileTimeAST::Ln(inner) => {
            let inner_opt = apply_basic_optimizations(inner);
            if let CompileTimeAST::Exp(exp_inner) = &inner_opt {
                (**exp_inner).clone()
            } else {
                CompileTimeAST::Ln(Box::new(inner_opt))
            }
        }
        // Recursively optimize other expressions
        CompileTimeAST::Sub(left, right) => CompileTimeAST::Sub(
            Box::new(apply_basic_optimizations(left)),
            Box::new(apply_basic_optimizations(right)),
        ),
        CompileTimeAST::Pow(base, exp) => CompileTimeAST::Pow(
            Box::new(apply_basic_optimizations(base)),
            Box::new(apply_basic_optimizations(exp)),
        ),
        CompileTimeAST::Sin(inner) => {
            CompileTimeAST::Sin(Box::new(apply_basic_optimizations(inner)))
        }
        CompileTimeAST::Cos(inner) => {
            CompileTimeAST::Cos(Box::new(apply_basic_optimizations(inner)))
        }
        CompileTimeAST::Exp(inner) => {
            CompileTimeAST::Exp(Box::new(apply_basic_optimizations(inner)))
        }
        // Leaf nodes
        CompileTimeAST::Variable(_) | CompileTimeAST::Constant(_) => ast.clone(),
    }
}

/// Convert optimized AST back to Rust expression tokens
fn ast_to_rust_expr(ast: &CompileTimeAST, vars: &[Ident]) -> TokenStream2 {
    match ast {
        CompileTimeAST::Variable(index) => {
            if let Some(var_name) = vars.get(*index) {
                quote! { #var_name }
            } else {
                // Fallback - shouldn't happen with valid input
                quote! { compile_error!("Invalid variable index") }
            }
        }
        CompileTimeAST::Constant(value) => {
            quote! { #value }
        }
        CompileTimeAST::Add(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr + #right_expr) }
        }
        CompileTimeAST::Mul(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr * #right_expr) }
        }
        CompileTimeAST::Sub(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr - #right_expr) }
        }
        CompileTimeAST::Pow(base, exp) => {
            let base_expr = ast_to_rust_expr(base, vars);
            let exp_expr = ast_to_rust_expr(exp, vars);
            quote! { (#base_expr).powf(#exp_expr) }
        }
        CompileTimeAST::Sin(inner) => {
            let inner_expr = ast_to_rust_expr(inner, vars);
            quote! { (#inner_expr).sin() }
        }
        CompileTimeAST::Cos(inner) => {
            let inner_expr = ast_to_rust_expr(inner, vars);
            quote! { (#inner_expr).cos() }
        }
        CompileTimeAST::Exp(inner) => {
            let inner_expr = ast_to_rust_expr(inner, vars);
            quote! { (#inner_expr).exp() }
        }
        CompileTimeAST::Ln(inner) => {
            let inner_expr = ast_to_rust_expr(inner, vars);
            quote! { (#inner_expr).ln() }
        }
    }
}
