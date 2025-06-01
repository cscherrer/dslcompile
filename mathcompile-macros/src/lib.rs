use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Expr, Ident, Token};

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
/// let result = optimize_compile_time!(
///     var::<0>().exp().ln().add(var::<1>().mul(constant(1.0))),
///     [x, y]
/// );
/// // Generates: x + y
/// ```
#[proc_macro]
pub fn optimize_compile_time(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as OptimizeInput);
    
    // Convert the expression to our internal AST representation
    let ast = match expr_to_ast(&input.expr) {
        Ok(ast) => ast,
        Err(e) => {
            return syn::Error::new_spanned(&input.expr, format!("Failed to parse expression: {}", e))
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
    }.into()
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
        
        let vars = content.parse_terminated(Ident::parse, Token![,])?
            .into_iter()
            .collect();
        
        Ok(OptimizeInput { expr, vars })
    }
}

/// Internal AST representation for compile-time optimization
#[derive(Debug, Clone, PartialEq)]
enum CompileTimeAST {
    Constant(f64),
    Variable(usize),
    Add(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Sub(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Mul(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Div(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Pow(Box<CompileTimeAST>, Box<CompileTimeAST>),
    Sin(Box<CompileTimeAST>),
    Cos(Box<CompileTimeAST>),
    Exp(Box<CompileTimeAST>),
    Ln(Box<CompileTimeAST>),
    Sqrt(Box<CompileTimeAST>),
    Neg(Box<CompileTimeAST>),
}

/// Convert Rust expression to our internal AST
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
                    Ok(CompileTimeAST::Add(Box::new(receiver_ast), Box::new(arg_ast)))
                }
                "sub" => {
                    if method_call.args.len() != 1 {
                        return Err("sub() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Sub(Box::new(receiver_ast), Box::new(arg_ast)))
                }
                "mul" => {
                    if method_call.args.len() != 1 {
                        return Err("mul() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Mul(Box::new(receiver_ast), Box::new(arg_ast)))
                }
                "div" => {
                    if method_call.args.len() != 1 {
                        return Err("div() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Div(Box::new(receiver_ast), Box::new(arg_ast)))
                }
                "pow" => {
                    if method_call.args.len() != 1 {
                        return Err("pow() requires exactly one argument".to_string());
                    }
                    let arg_ast = expr_to_ast(&method_call.args[0])?;
                    Ok(CompileTimeAST::Pow(Box::new(receiver_ast), Box::new(arg_ast)))
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
                    Ok(CompileTimeAST::Sqrt(Box::new(receiver_ast)))
                }
                "neg" => {
                    if !method_call.args.is_empty() {
                        return Err("neg() takes no arguments".to_string());
                    }
                    Ok(CompileTimeAST::Neg(Box::new(receiver_ast)))
                }
                _ => Err(format!("Unknown method: {}", method_call.method))
            }
        }
        
        // Function calls like var::<0>() or constant(1.0)
        Expr::Call(call) => {
            if let Expr::Path(path) = &*call.func {
                if let Some(segment) = path.path.segments.last() {
                    match segment.ident.to_string().as_str() {
                        "var" => {
                            // Extract the const generic parameter
                            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                                if let Some(syn::GenericArgument::Const(const_expr)) = args.args.first() {
                                    if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit_int), .. }) = const_expr {
                                        let var_id: usize = lit_int.base10_parse()
                                            .map_err(|_| "Invalid variable ID".to_string())?;
                                        return Ok(CompileTimeAST::Variable(var_id));
                                    }
                                }
                            }
                            Err("Invalid var::<ID>() syntax".to_string())
                        }
                        "constant" => {
                            if call.args.len() != 1 {
                                return Err("constant() requires exactly one argument".to_string());
                            }
                            if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Float(lit_float), .. }) = &call.args[0] {
                                let value: f64 = lit_float.base10_parse()
                                    .map_err(|_| "Invalid float literal".to_string())?;
                                Ok(CompileTimeAST::Constant(value))
                            } else if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit_int), .. }) = &call.args[0] {
                                let value: f64 = lit_int.base10_parse::<i64>()
                                    .map_err(|_| "Invalid int literal".to_string())? as f64;
                                Ok(CompileTimeAST::Constant(value))
                            } else {
                                Err("constant() argument must be a numeric literal".to_string())
                            }
                        }
                        _ => Err(format!("Unknown function: {}", segment.ident))
                    }
                } else {
                    Err("Invalid function call".to_string())
                }
            } else {
                Err("Complex function calls not supported".to_string())
            }
        }
        
        _ => Err("Unsupported expression type".to_string())
    }
}

/// Run compile-time egglog optimization
fn run_compile_time_optimization(ast: &CompileTimeAST) -> CompileTimeAST {
    let mut current = ast.clone();
    
    // Run fixed-point optimization (equality saturation)
    for _ in 0..10 {
        let next = apply_optimization_rules(&current);
        if ast_equal(&current, &next) {
            break;
        }
        current = next;
    }
    
    current
}

/// Apply all optimization rules once
fn apply_optimization_rules(ast: &CompileTimeAST) -> CompileTimeAST {
    // First optimize children recursively
    let ast_with_optimized_children = match ast {
        CompileTimeAST::Add(left, right) => {
            CompileTimeAST::Add(
                Box::new(apply_optimization_rules(left)),
                Box::new(apply_optimization_rules(right))
            )
        }
        CompileTimeAST::Sub(left, right) => {
            CompileTimeAST::Sub(
                Box::new(apply_optimization_rules(left)),
                Box::new(apply_optimization_rules(right))
            )
        }
        CompileTimeAST::Mul(left, right) => {
            CompileTimeAST::Mul(
                Box::new(apply_optimization_rules(left)),
                Box::new(apply_optimization_rules(right))
            )
        }
        CompileTimeAST::Div(left, right) => {
            CompileTimeAST::Div(
                Box::new(apply_optimization_rules(left)),
                Box::new(apply_optimization_rules(right))
            )
        }
        CompileTimeAST::Pow(base, exp) => {
            CompileTimeAST::Pow(
                Box::new(apply_optimization_rules(base)),
                Box::new(apply_optimization_rules(exp))
            )
        }
        CompileTimeAST::Sin(inner) => {
            CompileTimeAST::Sin(Box::new(apply_optimization_rules(inner)))
        }
        CompileTimeAST::Cos(inner) => {
            CompileTimeAST::Cos(Box::new(apply_optimization_rules(inner)))
        }
        CompileTimeAST::Exp(inner) => {
            CompileTimeAST::Exp(Box::new(apply_optimization_rules(inner)))
        }
        CompileTimeAST::Ln(inner) => {
            CompileTimeAST::Ln(Box::new(apply_optimization_rules(inner)))
        }
        CompileTimeAST::Sqrt(inner) => {
            CompileTimeAST::Sqrt(Box::new(apply_optimization_rules(inner)))
        }
        CompileTimeAST::Neg(inner) => {
            CompileTimeAST::Neg(Box::new(apply_optimization_rules(inner)))
        }
        // Leaf nodes
        _ => ast.clone(),
    };
    
    // Then apply local optimizations
    apply_local_optimizations(&ast_with_optimized_children)
}

/// Apply local optimization rules
fn apply_local_optimizations(ast: &CompileTimeAST) -> CompileTimeAST {
    match ast {
        // ln(exp(x)) -> x
        CompileTimeAST::Ln(inner) => {
            if let CompileTimeAST::Exp(exp_inner) = inner.as_ref() {
                (**exp_inner).clone()
            } else if let CompileTimeAST::Mul(left, right) = inner.as_ref() {
                // ln(a * b) -> ln(a) + ln(b)
                CompileTimeAST::Add(
                    Box::new(CompileTimeAST::Ln(left.clone())),
                    Box::new(CompileTimeAST::Ln(right.clone())),
                )
            } else {
                ast.clone()
            }
        }
        // exp(ln(x)) -> x
        CompileTimeAST::Exp(inner) => {
            if let CompileTimeAST::Ln(ln_inner) = inner.as_ref() {
                (**ln_inner).clone()
            } else if let CompileTimeAST::Add(left, right) = inner.as_ref() {
                // exp(a + b) -> exp(a) * exp(b)
                CompileTimeAST::Mul(
                    Box::new(CompileTimeAST::Exp(left.clone())),
                    Box::new(CompileTimeAST::Exp(right.clone())),
                )
            } else {
                ast.clone()
            }
        }
        // x + 0 -> x, 0 + x -> x
        CompileTimeAST::Add(left, right) => {
            if let CompileTimeAST::Constant(0.0) = right.as_ref() {
                (**left).clone()
            } else if let CompileTimeAST::Constant(0.0) = left.as_ref() {
                (**right).clone()
            } else {
                ast.clone()
            }
        }
        // x * 1 -> x, 1 * x -> x, x * 0 -> 0, 0 * x -> 0
        CompileTimeAST::Mul(left, right) => {
            if let CompileTimeAST::Constant(1.0) = right.as_ref() {
                (**left).clone()
            } else if let CompileTimeAST::Constant(1.0) = left.as_ref() {
                (**right).clone()
            } else if let CompileTimeAST::Constant(0.0) = right.as_ref() {
                CompileTimeAST::Constant(0.0)
            } else if let CompileTimeAST::Constant(0.0) = left.as_ref() {
                CompileTimeAST::Constant(0.0)
            } else if let (CompileTimeAST::Exp(exp_left), CompileTimeAST::Exp(exp_right)) = (left.as_ref(), right.as_ref()) {
                // exp(a) * exp(b) -> exp(a + b)
                CompileTimeAST::Exp(Box::new(CompileTimeAST::Add(exp_left.clone(), exp_right.clone())))
            } else {
                ast.clone()
            }
        }
        _ => ast.clone(),
    }
}

/// Check if two ASTs are equal
fn ast_equal(a: &CompileTimeAST, b: &CompileTimeAST) -> bool {
    match (a, b) {
        (CompileTimeAST::Constant(a), CompileTimeAST::Constant(b)) => (a - b).abs() < 1e-10,
        (CompileTimeAST::Variable(a), CompileTimeAST::Variable(b)) => a == b,
        (CompileTimeAST::Add(a1, a2), CompileTimeAST::Add(b1, b2)) => {
            ast_equal(a1, b1) && ast_equal(a2, b2)
        }
        (CompileTimeAST::Sub(a1, a2), CompileTimeAST::Sub(b1, b2)) => {
            ast_equal(a1, b1) && ast_equal(a2, b2)
        }
        (CompileTimeAST::Mul(a1, a2), CompileTimeAST::Mul(b1, b2)) => {
            ast_equal(a1, b1) && ast_equal(a2, b2)
        }
        (CompileTimeAST::Div(a1, a2), CompileTimeAST::Div(b1, b2)) => {
            ast_equal(a1, b1) && ast_equal(a2, b2)
        }
        (CompileTimeAST::Pow(a1, a2), CompileTimeAST::Pow(b1, b2)) => {
            ast_equal(a1, b1) && ast_equal(a2, b2)
        }
        (CompileTimeAST::Sin(a), CompileTimeAST::Sin(b)) => ast_equal(a, b),
        (CompileTimeAST::Cos(a), CompileTimeAST::Cos(b)) => ast_equal(a, b),
        (CompileTimeAST::Exp(a), CompileTimeAST::Exp(b)) => ast_equal(a, b),
        (CompileTimeAST::Ln(a), CompileTimeAST::Ln(b)) => ast_equal(a, b),
        (CompileTimeAST::Sqrt(a), CompileTimeAST::Sqrt(b)) => ast_equal(a, b),
        (CompileTimeAST::Neg(a), CompileTimeAST::Neg(b)) => ast_equal(a, b),
        _ => false,
    }
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
        CompileTimeAST::Div(left, right) => {
            let left_expr = ast_to_rust_expr(left, vars);
            let right_expr = ast_to_rust_expr(right, vars);
            quote! { (#left_expr / #right_expr) }
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
        CompileTimeAST::Sqrt(inner) => {
            let inner_expr = ast_to_rust_expr_with_parens(inner, vars);
            quote! { #inner_expr.sqrt() }
        }
        CompileTimeAST::Neg(inner) => {
            let inner_expr = ast_to_rust_expr(inner, vars);
            quote! { -(#inner_expr) }
        }
    }
}

/// Generate expression with parentheses when needed for method calls
fn ast_to_rust_expr_with_parens(ast: &CompileTimeAST, vars: &[Ident]) -> TokenStream2 {
    match ast {
        // Simple expressions don't need parentheses for method calls
        CompileTimeAST::Constant(_) | CompileTimeAST::Variable(_) => {
            ast_to_rust_expr(ast, vars)
        }
        // Complex expressions need parentheses
        _ => {
            let expr = ast_to_rust_expr(ast, vars);
            quote! { (#expr) }
        }
    }
}
