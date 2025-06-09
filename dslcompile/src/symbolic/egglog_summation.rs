//! Egglog-based Summation System
//!
//! This module implements the new egglog-based summation system that integrates
//! with the unified ctx.var::<T>() API. It provides powerful mathematical
//! optimization for summation expressions using egglog's equality saturation.

use crate::ast::ASTRepr;
use crate::error::{DSLCompileError, Result};
use std::collections::HashMap;

#[cfg(feature = "optimization")]
use egglog::EGraph;

/// Egglog-based summation optimizer
#[cfg(feature = "optimization")]
pub struct EgglogSummationOptimizer {
    /// The egglog EGraph with summation rules
    egraph: EGraph,
    /// Variable counter for generating unique names
    var_counter: usize,
    /// Cache for expression mappings
    expr_cache: HashMap<String, ASTRepr<f64>>,
}

#[cfg(feature = "optimization")]
impl EgglogSummationOptimizer {
    /// Create a new egglog summation optimizer
    pub fn new() -> Result<Self> {
        let mut egraph = EGraph::default();

        // Load the summation egglog program
        let program = Self::create_summation_program();

        egraph.parse_and_run_program(None, &program).map_err(|e| {
            DSLCompileError::Optimization(format!(
                "Failed to initialize egglog summation optimizer: {e}"
            ))
        })?;

        Ok(Self {
            egraph,
            var_counter: 0,
            expr_cache: HashMap::new(),
        })
    }

    /// Create the egglog program with summation rules
    fn create_summation_program() -> String {
        // Include the base mathematical rules plus summation-specific rules
        let base_math_rules = r"
; Mathematical expression datatype
(datatype Math
  (Num f64)
  (Var String)
  (Add Math Math)
  (Sub Math Math)
  (Mul Math Math)
  (Div Math Math)
  (Pow Math Math)
  (Neg Math)
  (Ln Math)
  (Exp Math)
  (Sin Math)
  (Cos Math)
  (Sqrt Math))

; Basic mathematical rules
(rewrite (Add a (Num 0.0)) a)
(rewrite (Add (Num 0.0) a) a)
(rewrite (Mul a (Num 1.0)) a)
(rewrite (Mul (Num 1.0) a) a)
(rewrite (Mul a (Num 0.0)) (Num 0.0))
(rewrite (Mul (Num 0.0) a) (Num 0.0))
(rewrite (Add a b) (Add b a))
(rewrite (Mul a b) (Mul b a))
";

        let summation_rules = include_str!("../egglog_rules/summation_unified.egg");

        format!("{base_math_rules}\n{summation_rules}")
    }

    /// Optimize a summation expression using egglog
    pub fn optimize_summation(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // Convert expression to egglog format
        let egglog_expr = self.ast_to_egglog(expr)?;
        let expr_id = format!("sum_expr_{}", self.var_counter);
        self.var_counter += 1;

        // Store original expression
        self.expr_cache.insert(expr_id.clone(), expr.clone());

        // Add expression to egglog
        let add_command = format!("(let {expr_id} {egglog_expr})");
        self.egraph
            .parse_and_run_program(None, &add_command)
            .map_err(|e| {
                DSLCompileError::Optimization(format!(
                    "Failed to add summation expression to egglog: {e}"
                ))
            })?;

        // Run summation optimization rules
        self.egraph
            .parse_and_run_program(None, "(run 20)")
            .map_err(|e| {
                DSLCompileError::Optimization(format!(
                    "Failed to run summation optimization rules: {e}"
                ))
            })?;

        // Extract the best expression
        self.extract_best(&expr_id)
    }

    /// Convert AST to egglog format with summation support
    fn ast_to_egglog(&self, expr: &ASTRepr<f64>) -> Result<String> {
        match expr {
            ASTRepr::Constant(value) => {
                if value.fract() == 0.0 {
                    Ok(format!("(Num {value:.1})"))
                } else {
                    Ok(format!("(Num {value})"))
                }
            }
            ASTRepr::Variable(index) => Ok(format!("(Var \"x{index}\")")),
            ASTRepr::Add(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Add {left_s} {right_s})"))
            }
            ASTRepr::Sub(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Sub {left_s} {right_s})"))
            }
            ASTRepr::Mul(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Mul {left_s} {right_s})"))
            }
            ASTRepr::Div(left, right) => {
                let left_s = self.ast_to_egglog(left)?;
                let right_s = self.ast_to_egglog(right)?;
                Ok(format!("(Div {left_s} {right_s})"))
            }
            ASTRepr::Pow(base, exp) => {
                let base_s = self.ast_to_egglog(base)?;
                let exp_s = self.ast_to_egglog(exp)?;
                Ok(format!("(Pow {base_s} {exp_s})"))
            }
            ASTRepr::Neg(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Neg {inner_s})"))
            }
            ASTRepr::Ln(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Ln {inner_s})"))
            }
            ASTRepr::Exp(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Exp {inner_s})"))
            }
            ASTRepr::Sin(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Sin {inner_s})"))
            }
            ASTRepr::Cos(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Cos {inner_s})"))
            }
            ASTRepr::Sqrt(inner) => {
                let inner_s = self.ast_to_egglog(inner)?;
                Ok(format!("(Sqrt {inner_s})"))
            }
            ASTRepr::Sum {
                range,
                body,
                iter_var,
            } => {
                use crate::ast::ast_repr::SumRange;

                let body_s = self.ast_to_egglog(body)?;

                match range {
                    SumRange::Mathematical { start, end } => {
                        let start_s = self.ast_to_egglog(start)?;
                        let end_s = self.ast_to_egglog(end)?;
                        Ok(format!(
                            "(SumExpr (MathRange {start_s} {end_s}) {body_s} {iter_var})"
                        ))
                    }
                    SumRange::DataParameter { data_var } => {
                        Ok(format!(
                            "(SumExpr (DataRange {data_var}) {body_s} {iter_var})"
                        ))
                    }
                }
            }
        }
    }

    /// Extract the best expression using egglog extraction
    fn extract_best(&mut self, expr_id: &str) -> Result<ASTRepr<f64>> {
        let extract_command = format!("(extract {expr_id})");

        let extract_result = self
            .egraph
            .parse_and_run_program(None, &extract_command)
            .map_err(|e| {
                DSLCompileError::Optimization(format!(
                    "Failed to extract optimized summation expression: {e}"
                ))
            })?;

        let output_string = extract_result.join("\n");

        // Parse the extraction result back to AST
        match self.parse_egglog_output(&output_string) {
            Ok(optimized) => Ok(optimized),
            Err(_) => {
                // Fall back to original expression if parsing fails
                self.expr_cache.get(expr_id).cloned().ok_or_else(|| {
                    DSLCompileError::Optimization("Expression not found in cache".to_string())
                })
            }
        }
    }

    /// Parse egglog output back to ASTRepr
    fn parse_egglog_output(&self, output: &str) -> Result<ASTRepr<f64>> {
        let cleaned = output.trim();
        self.parse_sexpr(cleaned)
    }

    /// Parse a single s-expression
    fn parse_sexpr(&self, s: &str) -> Result<ASTRepr<f64>> {
        let s = s.trim();

        // Handle constants
        if let Ok(value) = s.parse::<f64>() {
            return Ok(ASTRepr::Constant(value));
        }

        // Handle parenthesized expressions
        if s.starts_with('(') && s.ends_with(')') {
            let inner = &s[1..s.len() - 1];
            let tokens = self.tokenize_sexpr(inner);

            if tokens.is_empty() {
                return Err(DSLCompileError::Optimization("Empty expression".to_string()));
            }

            match tokens[0].as_str() {
                "Num" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Num expression".to_string(),
                        ));
                    }
                    let value = tokens[1].parse::<f64>().map_err(|_| {
                        DSLCompileError::Optimization("Invalid number".to_string())
                    })?;
                    Ok(ASTRepr::Constant(value))
                }
                "Var" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Var expression".to_string(),
                        ));
                    }
                    // Extract variable index from "x0", "x1", etc.
                    let var_name = tokens[1].trim_matches('"');
                    if let Some(index_str) = var_name.strip_prefix('x') {
                        let index = index_str.parse::<usize>().map_err(|_| {
                            DSLCompileError::Optimization("Invalid variable index".to_string())
                        })?;
                        Ok(ASTRepr::Variable(index))
                    } else {
                        Err(DSLCompileError::Optimization(
                            "Invalid variable name".to_string(),
                        ))
                    }
                }
                "Add" => {
                    if tokens.len() != 3 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Add expression".to_string(),
                        ));
                    }
                    let left = self.parse_sexpr(&tokens[1])?;
                    let right = self.parse_sexpr(&tokens[2])?;
                    Ok(ASTRepr::Add(Box::new(left), Box::new(right)))
                }
                "Sub" => {
                    if tokens.len() != 3 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Sub expression".to_string(),
                        ));
                    }
                    let left = self.parse_sexpr(&tokens[1])?;
                    let right = self.parse_sexpr(&tokens[2])?;
                    Ok(ASTRepr::Sub(Box::new(left), Box::new(right)))
                }
                "Mul" => {
                    if tokens.len() != 3 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Mul expression".to_string(),
                        ));
                    }
                    let left = self.parse_sexpr(&tokens[1])?;
                    let right = self.parse_sexpr(&tokens[2])?;
                    Ok(ASTRepr::Mul(Box::new(left), Box::new(right)))
                }
                "Div" => {
                    if tokens.len() != 3 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Div expression".to_string(),
                        ));
                    }
                    let left = self.parse_sexpr(&tokens[1])?;
                    let right = self.parse_sexpr(&tokens[2])?;
                    Ok(ASTRepr::Div(Box::new(left), Box::new(right)))
                }
                "Pow" => {
                    if tokens.len() != 3 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Pow expression".to_string(),
                        ));
                    }
                    let base = self.parse_sexpr(&tokens[1])?;
                    let exp = self.parse_sexpr(&tokens[2])?;
                    Ok(ASTRepr::Pow(Box::new(base), Box::new(exp)))
                }
                "Neg" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Neg expression".to_string(),
                        ));
                    }
                    let inner = self.parse_sexpr(&tokens[1])?;
                    Ok(ASTRepr::Neg(Box::new(inner)))
                }
                "Ln" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Ln expression".to_string(),
                        ));
                    }
                    let inner = self.parse_sexpr(&tokens[1])?;
                    Ok(ASTRepr::Ln(Box::new(inner)))
                }
                "Exp" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Exp expression".to_string(),
                        ));
                    }
                    let inner = self.parse_sexpr(&tokens[1])?;
                    Ok(ASTRepr::Exp(Box::new(inner)))
                }
                "Sin" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Sin expression".to_string(),
                        ));
                    }
                    let inner = self.parse_sexpr(&tokens[1])?;
                    Ok(ASTRepr::Sin(Box::new(inner)))
                }
                "Cos" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Cos expression".to_string(),
                        ));
                    }
                    let inner = self.parse_sexpr(&tokens[1])?;
                    Ok(ASTRepr::Cos(Box::new(inner)))
                }
                "Sqrt" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid Sqrt expression".to_string(),
                        ));
                    }
                    let inner = self.parse_sexpr(&tokens[1])?;
                    Ok(ASTRepr::Sqrt(Box::new(inner)))
                }
                "SumExpr" => {
                    // Parse summation expressions
                    if tokens.len() != 4 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid SumExpr expression".to_string(),
                        ));
                    }
                    
                    // Parse range
                    let range = self.parse_sum_range(&tokens[1])?;
                    
                    // Parse body
                    let body = self.parse_sexpr(&tokens[2])?;
                    
                    // Parse iterator variable
                    let iter_var = tokens[3].parse::<usize>().map_err(|_| {
                        DSLCompileError::Optimization("Invalid iterator variable".to_string())
                    })?;
                    
                    Ok(ASTRepr::Sum {
                        range,
                        body: Box::new(body),
                        iter_var,
                    })
                }
                "DataSum" => {
                    // Parse data summation expressions (special runtime node)
                    if tokens.len() != 4 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid DataSum expression".to_string(),
                        ));
                    }
                    
                    let data_var = tokens[1].parse::<usize>().map_err(|_| {
                        DSLCompileError::Optimization("Invalid data variable".to_string())
                    })?;
                    
                    let body = self.parse_sexpr(&tokens[2])?;
                    
                    let iter_var = tokens[3].parse::<usize>().map_err(|_| {
                        DSLCompileError::Optimization("Invalid iterator variable".to_string())
                    })?;
                    
                    // Convert DataSum back to regular Sum with DataParameter range
                    Ok(ASTRepr::Sum {
                        range: crate::ast::ast_repr::SumRange::DataParameter { data_var },
                        body: Box::new(body),
                        iter_var,
                    })
                }
                _ => Err(DSLCompileError::Optimization(format!(
                    "Unknown expression type: {}",
                    tokens[0]
                ))),
            }
        } else {
            Err(DSLCompileError::Optimization(
                "Invalid s-expression format".to_string(),
            ))
        }
    }

    /// Parse a summation range from egglog format
    fn parse_sum_range(&self, range_str: &str) -> Result<crate::ast::ast_repr::SumRange<f64>> {
        let range_str = range_str.trim();
        
        if range_str.starts_with('(') && range_str.ends_with(')') {
            let inner = &range_str[1..range_str.len() - 1];
            let tokens = self.tokenize_sexpr(inner);
            
            if tokens.is_empty() {
                return Err(DSLCompileError::Optimization("Empty range".to_string()));
            }
            
            match tokens[0].as_str() {
                "MathRange" => {
                    if tokens.len() != 3 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid MathRange".to_string(),
                        ));
                    }
                    let start = self.parse_sexpr(&tokens[1])?;
                    let end = self.parse_sexpr(&tokens[2])?;
                    Ok(crate::ast::ast_repr::SumRange::Mathematical {
                        start: Box::new(start),
                        end: Box::new(end),
                    })
                }
                "DataRange" => {
                    if tokens.len() != 2 {
                        return Err(DSLCompileError::Optimization(
                            "Invalid DataRange".to_string(),
                        ));
                    }
                    let data_var = tokens[1].parse::<usize>().map_err(|_| {
                        DSLCompileError::Optimization("Invalid data variable".to_string())
                    })?;
                    Ok(crate::ast::ast_repr::SumRange::DataParameter { data_var })
                }
                _ => Err(DSLCompileError::Optimization(format!(
                    "Unknown range type: {}",
                    tokens[0]
                ))),
            }
        } else {
            Err(DSLCompileError::Optimization(
                "Invalid range format".to_string(),
            ))
        }
    }

    /// Tokenize an s-expression into components
    fn tokenize_sexpr(&self, s: &str) -> Vec<String> {
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

        tokens
    }
}

/// Fallback implementation when optimization feature is not enabled
#[cfg(not(feature = "optimization"))]
pub struct EgglogSummationOptimizer;

#[cfg(not(feature = "optimization"))]
impl EgglogSummationOptimizer {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn optimize_summation(&mut self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        Ok(expr.clone())
    }
}

/// Helper function to optimize summation expressions using egglog
pub fn optimize_summation_with_egglog(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
    let mut optimizer = EgglogSummationOptimizer::new()?;
    optimizer.optimize_summation(expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ast_repr::SumRange;

    #[test]
    fn test_egglog_summation_optimizer_creation() {
        let result = EgglogSummationOptimizer::new();
        if let Err(e) = &result {
            println!("Error creating EgglogSummationOptimizer: {e}");
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_summation_ast_to_egglog() {
        let optimizer = EgglogSummationOptimizer::new().unwrap();

        // Test mathematical range summation
        let sum_expr = ASTRepr::Sum {
            range: SumRange::Mathematical {
                start: Box::new(ASTRepr::Constant(1.0)),
                end: Box::new(ASTRepr::Constant(10.0)),
            },
            body: Box::new(ASTRepr::Variable(0)),
            iter_var: 0,
        };

        let egglog_str = optimizer.ast_to_egglog(&sum_expr).unwrap();
        assert!(egglog_str.contains("SumExpr"));
        assert!(egglog_str.contains("MathRange"));
        assert!(egglog_str.contains("(Num 1.0)"));
        assert!(egglog_str.contains("(Num 10.0)"));
    }

    #[test]
    fn test_data_range_summation() {
        let optimizer = EgglogSummationOptimizer::new().unwrap();

        // Test data parameter summation
        let sum_expr = ASTRepr::Sum {
            range: SumRange::DataParameter { data_var: 2 },
            body: Box::new(ASTRepr::Ln(Box::new(ASTRepr::Variable(1002)))),
            iter_var: 1002,
        };

        let egglog_str = optimizer.ast_to_egglog(&sum_expr).unwrap();
        assert!(egglog_str.contains("SumExpr"));
        assert!(egglog_str.contains("DataRange"));
        assert!(egglog_str.contains("2"));
        assert!(egglog_str.contains("Ln"));
    }
} 