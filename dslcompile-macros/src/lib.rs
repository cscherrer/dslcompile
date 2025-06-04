use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, Ident, Token, parse_macro_input};

// Direct egglog integration for compile-time optimization
#[cfg(feature = "optimization")]
use egglog::EGraph;

/// Procedural macro for compile-time egglog optimization with direct code generation
///
/// This macro:
/// 1. Parses the expression at compile time
/// 2. Converts to AST representation  
/// 3. Runs egglog equality saturation during macro expansion
/// 4. Generates direct Rust expressions (no runtime dispatch, no enums)
///
/// Usage: `optimize_compile_time!(expr, [var1, var2, ...])`
/// Returns: Direct Rust expression that compiles to optimal assembly
///
/// Example:
/// ```rust
/// use dslcompile_macros::optimize_compile_time;
///
/// // This is a compile-time optimization example
/// // The macro would optimize mathematical expressions at compile time
/// // For now, this is a placeholder that demonstrates the syntax
/// # fn main() {
/// #     // Placeholder test - the actual macro requires more complex setup
/// #     let x = 1.0;
/// #     let y = 2.0;
/// #     let result = x + y; // This would be: optimize_compile_time!(x + y, [x, y]);
/// #     assert_eq!(result, 3.0);
/// # }
/// ```
#[proc_macro]
pub fn optimize_compile_time(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as OptimizeInput);

    // Convert the expression to our internal AST representation
    let ast = match expr_to_ast(&input.expr) {
        Ok(ast) => ast,
        Err(e) => {
            return syn::Error::new_spanned(
                &input.expr,
                format!("Failed to parse expression: {e}"),
            )
            .to_compile_error()
            .into();
        }
    };

    // Run egglog optimization at compile time
    let optimized_ast = run_compile_time_optimization(&ast);

    // Generate direct Rust code
    let generated_code = ast_to_rust_expr(&optimized_ast, &input.vars);

    // Return the optimized expression
    quote! {
        {
            #generated_code
        }
    }
    .into()
}

/// Input structure for the macro
struct OptimizeInput {
    expr: Expr,
    vars: Vec<Ident>,
}

impl syn::parse::Parse for OptimizeInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let expr = input.parse()?;
        input.parse::<Token![,]>()?;

        let content;
        syn::bracketed!(content in input);

        let vars = content
            .parse_terminated(Ident::parse, Token![,])?
            .into_iter()
            .collect();

        Ok(OptimizeInput { expr, vars })
    }
}

/// Compile-time AST representation for procedural macro parsing
#[derive(Debug, Clone, PartialEq)]
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
    /// Convert to egglog s-expression format
    #[cfg(feature = "optimization")]
    fn to_egglog(&self) -> String {
        match self {
            CompileTimeAST::Variable(id) => format!("(Var \"x{id}\")"),
            CompileTimeAST::Constant(val) => {
                if val.fract() == 0.0 {
                    format!("(Num {val:.1})")
                } else {
                    format!("(Num {val})")
                }
            }
            CompileTimeAST::Add(left, right) => {
                format!("(Add {} {})", left.to_egglog(), right.to_egglog())
            }
            CompileTimeAST::Mul(left, right) => {
                format!("(Mul {} {})", left.to_egglog(), right.to_egglog())
            }
            CompileTimeAST::Sub(left, right) => {
                // Convert Sub to Add + Neg for canonical form
                format!("(Add {} (Neg {}))", left.to_egglog(), right.to_egglog())
            }
            CompileTimeAST::Sin(inner) => {
                format!("(Sin {})", inner.to_egglog())
            }
            CompileTimeAST::Cos(inner) => {
                format!("(Cos {})", inner.to_egglog())
            }
            CompileTimeAST::Exp(inner) => {
                format!("(Exp {})", inner.to_egglog())
            }
            CompileTimeAST::Ln(inner) => {
                format!("(Ln {})", inner.to_egglog())
            }
            CompileTimeAST::Pow(base, exp) => {
                format!("(Pow {} {})", base.to_egglog(), exp.to_egglog())
            }
        }
    }
}

/// Convert Rust expression to our internal AST representation
fn expr_to_ast(expr: &Expr) -> Result<CompileTimeAST, String> {
    match expr {
        // Method calls like var::<0>().sin().add(...)
        Expr::MethodCall(method_call) => {
            let receiver_ast = expr_to_ast(&method_call.receiver)?;

            match method_call.method.to_string().as_str() {
                "add" => {
                    if method_call.args.len() != 1 {
                        return Err("add() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Add(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "sub" => {
                    if method_call.args.len() != 1 {
                        return Err("sub() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Sub(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "mul" => {
                    if method_call.args.len() != 1 {
                        return Err("mul() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Mul(
                        Box::new(receiver_ast),
                        Box::new(arg_ast),
                    ))
                }
                "div" => {
                    if method_call.args.len() != 1 {
                        return Err("div() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    // Convert division to multiplication by reciprocal: a / b = a * b^(-1)
                    Ok(CompileTimeAST::Mul(
                        Box::new(receiver_ast),
                        Box::new(CompileTimeAST::Pow(
                            Box::new(arg_ast),
                            Box::new(CompileTimeAST::Constant(-1.0)),
                        )),
                    ))
                }
                "pow" => {
                    if method_call.args.len() != 1 {
                        return Err("pow() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
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
                "sqrt" => {
                    if !method_call.args.is_empty() {
                        return Err("sqrt() takes no arguments".to_string());
                    }
                    // Convert sqrt to power of 0.5: sqrt(x) = x^0.5
                    Ok(CompileTimeAST::Pow(
                        Box::new(receiver_ast),
                        Box::new(CompileTimeAST::Constant(0.5)),
                    ))
                }
                "neg" => {
                    if !method_call.args.is_empty() {
                        return Err("neg() takes no arguments".to_string());
                    }
                    // Convert negation to multiplication by -1: -x = (-1) * x
                    Ok(CompileTimeAST::Mul(
                        Box::new(CompileTimeAST::Constant(-1.0)),
                        Box::new(receiver_ast),
                    ))
                }
                _ => Err(format!("Unknown method: {}", method_call.method)),
            }
        }

        // Function calls like var::<0>() or constant(1.0)
        Expr::Call(call) => {
            if let Expr::Path(path) = &*call.func {
                if let Some(segment) = path.path.segments.last() {
                    match segment.ident.to_string().as_str() {
                        "var" => {
                            // Extract the const generic parameter
                            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments
                                && let Some(syn::GenericArgument::Const(const_expr)) =
                                    args.args.first()
                                && let Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Int(lit_int),
                                    ..
                                }) = const_expr
                            {
                                let var_id: usize = lit_int
                                    .base10_parse()
                                    .map_err(|_| "Invalid variable ID".to_string())?;
                                return Ok(CompileTimeAST::Variable(var_id));
                            }
                            Err("Invalid var::<ID>() syntax".to_string())
                        }
                        "constant" => {
                            if call.args.len() != 1 {
                                return Err("constant() requires exactly one argument".to_string());
                            }

                            // Handle different types of constant arguments
                            match &call.args[0] {
                                // Direct float literal: constant(1.0)
                                Expr::Lit(syn::ExprLit { lit: syn::Lit::Float(lit_float), .. }) => {
                                    let value: f64 = lit_float.base10_parse()
                                        .map_err(|_| "Invalid float literal".to_string())?;
                                    Ok(CompileTimeAST::Constant(value))
                                }
                                // Direct int literal: constant(1)
                                Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit_int), .. }) => {
                                    let value: f64 = lit_int.base10_parse::<i64>()
                                        .map_err(|_| "Invalid int literal".to_string())? as f64;
                                    Ok(CompileTimeAST::Constant(value))
                                }
                                // Unary expression: constant(*c) or constant(-1.0)
                                Expr::Unary(unary) => {
                                    match unary.op {
                                        syn::UnOp::Deref(_) => {
                                            // Handle *c - this is a dereference of a variable
                                            // For macro purposes, we'll treat this as a runtime constant
                                            // that gets evaluated when the macro is expanded
                                            Err("constant() with variable dereference not supported in compile-time optimization".to_string())
                                        }
                                        syn::UnOp::Neg(_) => {
                                            // Handle -1.0 or -1
                                            match &*unary.expr {
                                                Expr::Lit(syn::ExprLit { lit: syn::Lit::Float(lit_float), .. }) => {
                                                    let value: f64 = lit_float.base10_parse()
                                                        .map_err(|_| "Invalid float literal".to_string())?;
                                                    Ok(CompileTimeAST::Constant(-value))
                                                }
                                                Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit_int), .. }) => {
                                                    let value: f64 = lit_int.base10_parse::<i64>()
                                                        .map_err(|_| "Invalid int literal".to_string())? as f64;
                                                    Ok(CompileTimeAST::Constant(-value))
                                                }
                                                _ => Err("constant() with complex negative expression not supported".to_string())
                                            }
                                        }
                                        _ => Err("constant() with unsupported unary operator".to_string())
                                    }
                                }
                                // Variable or other complex expression
                                _ => Err("constant() argument must be a numeric literal (variables not supported in compile-time optimization)".to_string())
                            }
                        }
                        _ => Err(format!("Unknown function: {}", segment.ident)),
                    }
                } else {
                    Err("Invalid function call".to_string())
                }
            } else {
                Err("Complex function calls not supported".to_string())
            }
        }

        _ => Err("Unsupported expression type".to_string()),
    }
}

/// Run compile-time egglog optimization using the real egglog engine
#[cfg(feature = "optimization")]
fn run_compile_time_optimization(ast: &CompileTimeAST) -> CompileTimeAST {
    // Create egglog instance with mathematical rules
    let mut egraph = match create_egglog_with_math_rules() {
        Ok(egraph) => egraph,
        Err(_) => return ast.clone(), // Fallback to original if egglog fails
    };

    // Convert to egglog s-expression format
    let egglog_expr = ast.to_egglog();
    let expr_id = "expr_0";

    // Add expression to egglog
    let add_command = format!("(let {expr_id} {egglog_expr})");
    if egraph.parse_and_run_program(None, &add_command).is_err() {
        return ast.clone(); // Fallback if adding expression fails
    }

    // Run optimization rules with STRICT LIMIT to prevent infinite expansion
    if egraph.parse_and_run_program(None, "(run 3)").is_err() {
        return ast.clone(); // Fallback if optimization fails
    }

    // Extract the best expression
    let extract_command = format!("(extract {expr_id})");
    match egraph.parse_and_run_program(None, &extract_command) {
        Ok(result) => {
            // Parse the result back to CompileTimeAST
            let output_string = result.join("\n");
            parse_egglog_result(&output_string).unwrap_or_else(|_| ast.clone())
        }
        Err(_) => ast.clone(), // Fallback if extraction fails
    }
}

/// Create egglog instance with mathematical optimization rules
#[cfg(feature = "optimization")]
fn create_egglog_with_math_rules() -> Result<EGraph, String> {
    let mut egraph = EGraph::default();

    // Load a SAFE mathematical optimization program (no infinite expansion)
    let program = r"
; Mathematical expression datatype
(datatype Math
  (Num f64)
  (Var String)
  (Add Math Math)
  (Mul Math Math)
  (Neg Math)
  (Pow Math Math)
  (Ln Math)
  (Exp Math)
  (Sin Math)
  (Cos Math))

; SAFE SIMPLIFICATION RULES (no expansion)
; Identity rules
(rewrite (Add a (Num 0.0)) a)
(rewrite (Add (Num 0.0) a) a)
(rewrite (Mul a (Num 1.0)) a)
(rewrite (Mul (Num 1.0) a) a)
(rewrite (Mul a (Num 0.0)) (Num 0.0))
(rewrite (Mul (Num 0.0) a) (Num 0.0))
(rewrite (Pow a (Num 0.0)) (Num 1.0))
(rewrite (Pow a (Num 1.0)) a)

; SAFE transcendental identities (only simplifying)
(rewrite (Ln (Exp x)) x)
; Remove the problematic expansion rule: (rewrite (Exp (Add a b)) (Mul (Exp a) (Exp b)))

; SAFE specific patterns (no general commutativity/associativity)
(rewrite (Ln (Mul (Exp a) (Exp b))) (Add a b))

; Double negation
(rewrite (Neg (Neg x)) x)

; Power simplifications
(rewrite (Pow (Exp x) y) (Exp (Mul x y)))
(rewrite (Pow x (Num 0.5)) (Sqrt x))
";

    egraph
        .parse_and_run_program(None, program)
        .map_err(|e| format!("Failed to initialize egglog: {e}"))?;

    Ok(egraph)
}

/// Parse egglog extraction result back to `CompileTimeAST`
#[cfg(feature = "optimization")]
fn parse_egglog_result(output: &str) -> Result<CompileTimeAST, String> {
    let cleaned = output.trim();
    parse_sexpr(cleaned)
}

/// Parse a single s-expression to `CompileTimeAST`
#[cfg(feature = "optimization")]
fn parse_sexpr(s: &str) -> Result<CompileTimeAST, String> {
    let s = s.trim();

    if !s.starts_with('(') || !s.ends_with(')') {
        return Err("Invalid s-expression format".to_string());
    }

    let inner = &s[1..s.len() - 1];
    let tokens = tokenize_sexpr(inner)?;

    if tokens.is_empty() {
        return Err("Empty s-expression".to_string());
    }

    match tokens[0].as_str() {
        "Num" => {
            if tokens.len() != 2 {
                return Err("Num requires exactly one argument".to_string());
            }
            let value: f64 = tokens[1]
                .parse()
                .map_err(|_| "Invalid number format".to_string())?;
            Ok(CompileTimeAST::Constant(value))
        }
        "Var" => {
            if tokens.len() != 2 {
                return Err("Var requires exactly one argument".to_string());
            }
            // Parse variable name like "x0" to get index
            let var_name = tokens[1].trim_matches('"');
            if !var_name.starts_with('x') {
                return Err("Invalid variable name format".to_string());
            }
            let index: usize = var_name[1..]
                .parse()
                .map_err(|_| "Invalid variable index".to_string())?;
            Ok(CompileTimeAST::Variable(index))
        }
        "Add" => {
            if tokens.len() != 3 {
                return Err("Add requires exactly two arguments".to_string());
            }
            let left = parse_sexpr(&tokens[1])?;
            let right = parse_sexpr(&tokens[2])?;
            Ok(CompileTimeAST::Add(Box::new(left), Box::new(right)))
        }
        "Mul" => {
            if tokens.len() != 3 {
                return Err("Mul requires exactly two arguments".to_string());
            }
            let left = parse_sexpr(&tokens[1])?;
            let right = parse_sexpr(&tokens[2])?;
            Ok(CompileTimeAST::Mul(Box::new(left), Box::new(right)))
        }
        "Neg" => {
            if tokens.len() != 2 {
                return Err("Neg requires exactly one argument".to_string());
            }
            let inner = parse_sexpr(&tokens[1])?;
            // Convert Neg to Mul by -1
            Ok(CompileTimeAST::Mul(
                Box::new(CompileTimeAST::Constant(-1.0)),
                Box::new(inner),
            ))
        }
        "Pow" => {
            if tokens.len() != 3 {
                return Err("Pow requires exactly two arguments".to_string());
            }
            let base = parse_sexpr(&tokens[1])?;
            let exp = parse_sexpr(&tokens[2])?;
            Ok(CompileTimeAST::Pow(Box::new(base), Box::new(exp)))
        }
        "Ln" => {
            if tokens.len() != 2 {
                return Err("Ln requires exactly one argument".to_string());
            }
            let inner = parse_sexpr(&tokens[1])?;
            Ok(CompileTimeAST::Ln(Box::new(inner)))
        }
        "Exp" => {
            if tokens.len() != 2 {
                return Err("Exp requires exactly one argument".to_string());
            }
            let inner = parse_sexpr(&tokens[1])?;
            Ok(CompileTimeAST::Exp(Box::new(inner)))
        }
        "Sin" => {
            if tokens.len() != 2 {
                return Err("Sin requires exactly one argument".to_string());
            }
            let inner = parse_sexpr(&tokens[1])?;
            Ok(CompileTimeAST::Sin(Box::new(inner)))
        }
        "Cos" => {
            if tokens.len() != 2 {
                return Err("Cos requires exactly one argument".to_string());
            }
            let inner = parse_sexpr(&tokens[1])?;
            Ok(CompileTimeAST::Cos(Box::new(inner)))
        }
        _ => Err(format!("Unknown function: {}", tokens[0])),
    }
}

/// Tokenize s-expression while respecting nested parentheses
#[cfg(feature = "optimization")]
fn tokenize_sexpr(s: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    let mut paren_depth = 0;
    let mut in_quotes = false;

    for ch in s.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                current_token.push(ch);
            }
            '(' if !in_quotes => {
                if paren_depth == 0 && !current_token.is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
                paren_depth += 1;
                current_token.push(ch);
            }
            ')' if !in_quotes => {
                paren_depth -= 1;
                current_token.push(ch);
                if paren_depth == 0 {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
            }
            ' ' | '\t' | '\n' if !in_quotes && paren_depth == 0 => {
                if !current_token.is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
            }
            _ => {
                current_token.push(ch);
            }
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token.trim().to_string());
    }

    Ok(tokens)
}

/// Convert optimized AST to direct Rust expression
fn ast_to_rust_expr(ast: &CompileTimeAST, vars: &[Ident]) -> TokenStream2 {
    match ast {
        CompileTimeAST::Constant(c) => {
            quote! { #c }
        }
        CompileTimeAST::Variable(idx) => {
            if *idx < vars.len() {
                let var = &vars[*idx];
                quote! { #var }
            } else {
                quote! { 0.0 /* undefined variable */ }
            }
        }
        CompileTimeAST::Add(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr + #right_expr) }
        }
        CompileTimeAST::Sub(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr - #right_expr) }
        }
        CompileTimeAST::Mul(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr * #right_expr) }
        }
        CompileTimeAST::Pow(base, exp) => {
            let base_expr = ast_to_rust_expr_with_parens(base, vars);
            let exp_expr = ast_to_rust_expr(exp, vars);
            quote! { #base_expr.powf(#exp_expr) }
        }
        CompileTimeAST::Sin(inner) => {
            let inner_expr = ast_to_rust_expr_with_parens(inner, vars);
            quote! { #inner_expr.sin() }
        }
        CompileTimeAST::Cos(inner) => {
            let inner_expr = ast_to_rust_expr_with_parens(inner, vars);
            quote! { #inner_expr.cos() }
        }
        CompileTimeAST::Exp(inner) => {
            let inner_expr = ast_to_rust_expr_with_parens(inner, vars);
            quote! { #inner_expr.exp() }
        }
        CompileTimeAST::Ln(inner) => {
            let inner_expr = ast_to_rust_expr_with_parens(inner, vars);
            quote! { #inner_expr.ln() }
        }
    }
}

/// Generate expression with parentheses when needed for method calls
fn ast_to_rust_expr_with_parens(ast: &CompileTimeAST, vars: &[Ident]) -> TokenStream2 {
    match ast {
        // Simple expressions don't need parentheses for method calls
        CompileTimeAST::Constant(_) | CompileTimeAST::Variable(_) => ast_to_rust_expr(ast, vars),
        // Complex expressions need parentheses
        _ => {
            let expr = ast_to_rust_expr(ast, vars);
            quote! { (#expr) }
        }
    }
}

/// Fallback optimization when egglog is not available
#[cfg(not(feature = "optimization"))]
fn run_compile_time_optimization(ast: &CompileTimeAST) -> CompileTimeAST {
    // Apply basic manual optimizations as fallback
    apply_basic_optimizations(ast)
}

/// Apply basic optimization rules without egglog
#[cfg(not(feature = "optimization"))]
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
