//! ANF↔Egglog Bridge Module
//!
//! This module provides bidirectional conversion between ANF expressions and egglog Math expressions.
//! It leverages the existing VarRef system to ensure collision-free variable management across
//! the conversion boundary.
//!
//! # Architecture
//!
//! ```text
//! ASTRepr<f64> → ANF → EgglogMath → Optimized EgglogMath → ANF → Generated Code
//!             ↑      ↑           ↑                      ↑      ↑
//!             │      │           │                      │      │
//!             │      │           └── CSE Rules ────────┘      │
//!             │      │                                        │
//!             │      └── Collision-Free Variables ───────────┘
//!             │
//!             └── Current Pipeline
//! ```
//!
//! # Key Features
//!
//! - **Collision-Free Variables**: Uses VarRef system from ANF
//! - **Bidirectional**: ANF ↔ EgglogMath conversions
//! - **CSE Integration**: Supports Common Subexpression Elimination
//! - **Performance**: Zero-overhead abstractions where possible
//!
//! # Example Usage
//!
//! ```rust
//! use dslcompile::symbolic::anf::convert_to_anf;
//! use dslcompile::symbolic::egglog_anf_bridge::ANFEgglogBridge;
//!
//! // Convert AST → ANF → EgglogMath → Optimized → ANF
//! let anf_expr = convert_to_anf(&ast_expr)?;
//! let egglog_math = ANFEgglogBridge::anf_to_egglog(&anf_expr);
//! let optimized_math = run_egglog_cse_rules(egglog_math);
//! let optimized_anf = ANFEgglogBridge::egglog_to_anf(&optimized_math)?;
//! ```

use crate::{
    error::Result,
    symbolic::anf::{ANFAtom, ANFComputation, ANFExpr, VarRef},
};
use std::collections::HashMap;

/// Egglog Math expression representation that maps to our egglog datatype
///
/// This enum provides a Rust representation of the egglog Math datatype
/// with collision-free variable support using the VarRef system.
#[derive(Debug, Clone, PartialEq)]
pub enum EgglogMath {
    // Basic values
    Num(f64),
    UserVar(usize), // Maps to VarRef::User(usize)
    BoundVar(u32),  // Maps to VarRef::Bound(u32)

    // Binary arithmetic operations
    Add(Box<EgglogMath>, Box<EgglogMath>),
    Sub(Box<EgglogMath>, Box<EgglogMath>),
    Mul(Box<EgglogMath>, Box<EgglogMath>),
    Div(Box<EgglogMath>, Box<EgglogMath>),
    Pow(Box<EgglogMath>, Box<EgglogMath>),

    // Unary operations
    Neg(Box<EgglogMath>),
    Abs(Box<EgglogMath>),

    // Transcendental functions
    Ln(Box<EgglogMath>),
    Exp(Box<EgglogMath>),

    // Trigonometric functions
    Sin(Box<EgglogMath>),
    Cos(Box<EgglogMath>),

    // Other functions
    Sqrt(Box<EgglogMath>),

    // Let bindings with collision-free variables
    Let(u32, Box<EgglogMath>, Box<EgglogMath>), // (bound_id, expr, body)
}

impl EgglogMath {
    /// Create a constant expression
    pub fn constant(value: f64) -> Self {
        EgglogMath::Num(value)
    }

    /// Create a user variable reference
    pub fn user_var(index: usize) -> Self {
        EgglogMath::UserVar(index)
    }

    /// Create a bound variable reference
    pub fn bound_var(id: u32) -> Self {
        EgglogMath::BoundVar(id)
    }

    /// Create a let binding
    pub fn let_binding(bound_id: u32, expr: EgglogMath, body: EgglogMath) -> Self {
        EgglogMath::Let(bound_id, Box::new(expr), Box::new(body))
    }

    /// Count the number of let bindings in this expression
    pub fn let_count(&self) -> usize {
        match self {
            EgglogMath::Let(_, _, body) => 1 + body.let_count(),
            EgglogMath::Add(a, b)
            | EgglogMath::Sub(a, b)
            | EgglogMath::Mul(a, b)
            | EgglogMath::Div(a, b)
            | EgglogMath::Pow(a, b) => a.let_count() + b.let_count(),
            EgglogMath::Neg(a)
            | EgglogMath::Abs(a)
            | EgglogMath::Ln(a)
            | EgglogMath::Exp(a)
            | EgglogMath::Sin(a)
            | EgglogMath::Cos(a)
            | EgglogMath::Sqrt(a) => a.let_count(),
            _ => 0,
        }
    }

    /// Collect all user variables referenced in this expression
    pub fn user_variables(&self) -> Vec<usize> {
        let mut vars = Vec::new();
        self.collect_user_variables(&mut vars);
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    fn collect_user_variables(&self, vars: &mut Vec<usize>) {
        match self {
            EgglogMath::UserVar(idx) => vars.push(*idx),
            EgglogMath::Let(_, expr, body) => {
                expr.collect_user_variables(vars);
                body.collect_user_variables(vars);
            }
            EgglogMath::Add(a, b)
            | EgglogMath::Sub(a, b)
            | EgglogMath::Mul(a, b)
            | EgglogMath::Div(a, b)
            | EgglogMath::Pow(a, b) => {
                a.collect_user_variables(vars);
                b.collect_user_variables(vars);
            }
            EgglogMath::Neg(a)
            | EgglogMath::Abs(a)
            | EgglogMath::Ln(a)
            | EgglogMath::Exp(a)
            | EgglogMath::Sin(a)
            | EgglogMath::Cos(a)
            | EgglogMath::Sqrt(a) => {
                a.collect_user_variables(vars);
            }
            _ => {} // Constants and bound variables don't contribute user variables
        }
    }
}

/// Bridge for converting between ANF and Egglog representations
///
/// This struct manages the bidirectional conversion while maintaining
/// variable consistency and scope safety.
pub struct ANFEgglogBridge {
    /// Counter for generating fresh bound variable IDs
    next_bound_id: u32,
    /// Mapping from ANF bound IDs to egglog bound IDs for consistency
    bound_id_mapping: HashMap<u32, u32>,
}

impl ANFEgglogBridge {
    /// Create a new bridge instance
    pub fn new() -> Self {
        Self {
            next_bound_id: 0,
            bound_id_mapping: HashMap::new(),
        }
    }

    /// Convert ANF expression to Egglog Math
    ///
    /// This preserves the variable structure and let bindings from ANF,
    /// ensuring that CSE optimizations in egglog will be compatible.
    pub fn anf_to_egglog(&mut self, anf: &ANFExpr<f64>) -> EgglogMath {
        match anf {
            ANFExpr::Atom(atom) => self.anf_atom_to_egglog(atom),
            ANFExpr::Let(var_ref, computation, body) => {
                // Convert the computation to an egglog expression
                let expr = self.anf_computation_to_egglog(computation);

                // Get or create the bound ID for this let binding
                let bound_id = match var_ref {
                    VarRef::Bound(anf_id) => {
                        *self.bound_id_mapping.entry(*anf_id).or_insert_with(|| {
                            let id = self.next_bound_id;
                            self.next_bound_id += 1;
                            id
                        })
                    }
                    VarRef::User(_) => {
                        // This shouldn't happen in normal ANF, but handle gracefully
                        let id = self.next_bound_id;
                        self.next_bound_id += 1;
                        id
                    }
                };

                // Convert the body
                let body_expr = self.anf_to_egglog(body);

                EgglogMath::Let(bound_id, Box::new(expr), Box::new(body_expr))
            }
        }
    }

    /// Convert ANF atom to Egglog Math
    fn anf_atom_to_egglog(&self, atom: &ANFAtom<f64>) -> EgglogMath {
        match atom {
            ANFAtom::Constant(value) => EgglogMath::Num(*value),
            ANFAtom::Variable(var_ref) => match var_ref {
                VarRef::User(idx) => EgglogMath::UserVar(*idx),
                VarRef::Bound(id) => {
                    // Look up the mapped bound ID, or use the original if not found
                    let mapped_id = self.bound_id_mapping.get(id).copied().unwrap_or(*id);
                    EgglogMath::BoundVar(mapped_id)
                }
            },
        }
    }

    /// Convert ANF computation to Egglog Math
    fn anf_computation_to_egglog(&self, comp: &ANFComputation<f64>) -> EgglogMath {
        match comp {
            ANFComputation::Add(a, b) => EgglogMath::Add(
                Box::new(self.anf_atom_to_egglog(a)),
                Box::new(self.anf_atom_to_egglog(b)),
            ),
            ANFComputation::Sub(a, b) => EgglogMath::Sub(
                Box::new(self.anf_atom_to_egglog(a)),
                Box::new(self.anf_atom_to_egglog(b)),
            ),
            ANFComputation::Mul(a, b) => EgglogMath::Mul(
                Box::new(self.anf_atom_to_egglog(a)),
                Box::new(self.anf_atom_to_egglog(b)),
            ),
            ANFComputation::Div(a, b) => EgglogMath::Div(
                Box::new(self.anf_atom_to_egglog(a)),
                Box::new(self.anf_atom_to_egglog(b)),
            ),
            ANFComputation::Pow(a, b) => EgglogMath::Pow(
                Box::new(self.anf_atom_to_egglog(a)),
                Box::new(self.anf_atom_to_egglog(b)),
            ),
            ANFComputation::Neg(a) => EgglogMath::Neg(Box::new(self.anf_atom_to_egglog(a))),
            ANFComputation::Ln(a) => EgglogMath::Ln(Box::new(self.anf_atom_to_egglog(a))),
            ANFComputation::Exp(a) => EgglogMath::Exp(Box::new(self.anf_atom_to_egglog(a))),
            ANFComputation::Sin(a) => EgglogMath::Sin(Box::new(self.anf_atom_to_egglog(a))),
            ANFComputation::Cos(a) => EgglogMath::Cos(Box::new(self.anf_atom_to_egglog(a))),
            ANFComputation::Sqrt(a) => EgglogMath::Sqrt(Box::new(self.anf_atom_to_egglog(a))),
        }
    }

    /// Convert Egglog Math back to ANF expression
    ///
    /// This is the reverse transformation, allowing optimized egglog expressions
    /// to be converted back to ANF for code generation.
    pub fn egglog_to_anf(&mut self, math: &EgglogMath) -> Result<ANFExpr<f64>> {
        match math {
            EgglogMath::Num(value) => Ok(ANFExpr::Atom(ANFAtom::Constant(*value))),
            EgglogMath::UserVar(idx) => Ok(ANFExpr::Atom(ANFAtom::Variable(VarRef::User(*idx)))),
            EgglogMath::BoundVar(id) => Ok(ANFExpr::Atom(ANFAtom::Variable(VarRef::Bound(*id)))),

            // Let bindings
            EgglogMath::Let(bound_id, expr, body) => {
                // Convert the expression to a computation
                let computation = self.egglog_to_anf_computation(expr)?;

                // Convert the body
                let body_anf = self.egglog_to_anf(body)?;

                Ok(ANFExpr::Let(
                    VarRef::Bound(*bound_id),
                    computation,
                    Box::new(body_anf),
                ))
            }

            // Binary operations - these need to be converted to let bindings if they're not atomic
            EgglogMath::Add(a, b) => self.egglog_binary_to_anf(a, b, ANFComputation::Add),
            EgglogMath::Sub(a, b) => self.egglog_binary_to_anf(a, b, ANFComputation::Sub),
            EgglogMath::Mul(a, b) => self.egglog_binary_to_anf(a, b, ANFComputation::Mul),
            EgglogMath::Div(a, b) => self.egglog_binary_to_anf(a, b, ANFComputation::Div),
            EgglogMath::Pow(a, b) => self.egglog_binary_to_anf(a, b, ANFComputation::Pow),

            // Unary operations
            EgglogMath::Neg(a) => self.egglog_unary_to_anf(a, ANFComputation::Neg),
            EgglogMath::Abs(a) => {
                // Abs is not directly supported in ANF, would need extension
                Err("Abs operation not supported in ANF conversion".into())
            }
            EgglogMath::Ln(a) => self.egglog_unary_to_anf(a, ANFComputation::Ln),
            EgglogMath::Exp(a) => self.egglog_unary_to_anf(a, ANFComputation::Exp),
            EgglogMath::Sin(a) => self.egglog_unary_to_anf(a, ANFComputation::Sin),
            EgglogMath::Cos(a) => self.egglog_unary_to_anf(a, ANFComputation::Cos),
            EgglogMath::Sqrt(a) => self.egglog_unary_to_anf(a, ANFComputation::Sqrt),
        }
    }

    /// Convert binary egglog operation to ANF
    fn egglog_binary_to_anf<F>(
        &mut self,
        left: &EgglogMath,
        right: &EgglogMath,
        op_constructor: F,
    ) -> Result<ANFExpr<f64>>
    where
        F: Fn(ANFAtom<f64>, ANFAtom<f64>) -> ANFComputation<f64>,
    {
        // Try to convert operands to atoms
        let (left_lets, left_atom) = self.egglog_to_anf_atom(left)?;
        let (right_lets, right_atom) = self.egglog_to_anf_atom(right)?;

        // Generate fresh bound variable for the result
        let result_var = VarRef::Bound(self.next_bound_id);
        self.next_bound_id += 1;

        let computation = op_constructor(left_atom, right_atom);
        let result_expr = ANFExpr::Let(
            result_var,
            computation,
            Box::new(ANFExpr::Atom(ANFAtom::Variable(result_var))),
        );

        // Chain any necessary let bindings
        Ok(self.chain_anf_lets(left_lets, right_lets, result_expr))
    }

    /// Convert unary egglog operation to ANF
    fn egglog_unary_to_anf<F>(
        &mut self,
        operand: &EgglogMath,
        op_constructor: F,
    ) -> Result<ANFExpr<f64>>
    where
        F: Fn(ANFAtom<f64>) -> ANFComputation<f64>,
    {
        let (operand_lets, operand_atom) = self.egglog_to_anf_atom(operand)?;

        let result_var = VarRef::Bound(self.next_bound_id);
        self.next_bound_id += 1;

        let computation = op_constructor(operand_atom);
        let result_expr = ANFExpr::Let(
            result_var,
            computation,
            Box::new(ANFExpr::Atom(ANFAtom::Variable(result_var))),
        );

        Ok(self.wrap_anf_lets(operand_lets, result_expr))
    }

    /// Convert egglog expression to ANF atom, returning any necessary let bindings
    fn egglog_to_anf_atom(
        &mut self,
        math: &EgglogMath,
    ) -> Result<(Option<ANFExpr<f64>>, ANFAtom<f64>)> {
        match math {
            EgglogMath::Num(value) => Ok((None, ANFAtom::Constant(*value))),
            EgglogMath::UserVar(idx) => Ok((None, ANFAtom::Variable(VarRef::User(*idx)))),
            EgglogMath::BoundVar(id) => Ok((None, ANFAtom::Variable(VarRef::Bound(*id)))),
            _ => {
                // Complex expression - convert to ANF and extract the result variable
                let anf_expr = self.egglog_to_anf(math)?;
                let result_var = self.extract_anf_result_var(&anf_expr);
                Ok((Some(anf_expr), ANFAtom::Variable(result_var)))
            }
        }
    }

    /// Extract the result variable from an ANF expression
    fn extract_anf_result_var(&self, expr: &ANFExpr<f64>) -> VarRef {
        match expr {
            ANFExpr::Atom(ANFAtom::Variable(var_ref)) => *var_ref,
            ANFExpr::Let(var_ref, _, _) => *var_ref,
            ANFExpr::Atom(ANFAtom::Constant(_)) => {
                // This shouldn't happen in well-formed ANF, but handle gracefully
                VarRef::Bound(0)
            }
        }
    }

    /// Convert egglog math to ANF computation (for let binding RHS)
    fn egglog_to_anf_computation(&mut self, math: &EgglogMath) -> Result<ANFComputation<f64>> {
        match math {
            // These should be converted to atoms first, not computations
            EgglogMath::Num(_) | EgglogMath::UserVar(_) | EgglogMath::BoundVar(_) => {
                Err("Atomic expressions cannot be converted to computations".into())
            }

            EgglogMath::Add(a, b) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                let (_, b_atom) = self.egglog_to_anf_atom(b)?;
                Ok(ANFComputation::Add(a_atom, b_atom))
            }
            EgglogMath::Sub(a, b) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                let (_, b_atom) = self.egglog_to_anf_atom(b)?;
                Ok(ANFComputation::Sub(a_atom, b_atom))
            }
            EgglogMath::Mul(a, b) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                let (_, b_atom) = self.egglog_to_anf_atom(b)?;
                Ok(ANFComputation::Mul(a_atom, b_atom))
            }
            EgglogMath::Div(a, b) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                let (_, b_atom) = self.egglog_to_anf_atom(b)?;
                Ok(ANFComputation::Div(a_atom, b_atom))
            }
            EgglogMath::Pow(a, b) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                let (_, b_atom) = self.egglog_to_anf_atom(b)?;
                Ok(ANFComputation::Pow(a_atom, b_atom))
            }
            EgglogMath::Neg(a) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                Ok(ANFComputation::Neg(a_atom))
            }
            EgglogMath::Ln(a) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                Ok(ANFComputation::Ln(a_atom))
            }
            EgglogMath::Exp(a) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                Ok(ANFComputation::Exp(a_atom))
            }
            EgglogMath::Sin(a) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                Ok(ANFComputation::Sin(a_atom))
            }
            EgglogMath::Cos(a) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                Ok(ANFComputation::Cos(a_atom))
            }
            EgglogMath::Sqrt(a) => {
                let (_, a_atom) = self.egglog_to_anf_atom(a)?;
                Ok(ANFComputation::Sqrt(a_atom))
            }

            // Nested let bindings and other complex expressions need special handling
            EgglogMath::Let(_, _, _) => {
                Err("Let expressions cannot be directly converted to computations".into())
            }
            EgglogMath::Abs(_) => Err("Abs operation not supported in ANF computations".into()),
        }
    }

    /// Chain two optional ANF let expressions with a final expression
    fn chain_anf_lets(
        &self,
        first: Option<ANFExpr<f64>>,
        second: Option<ANFExpr<f64>>,
        final_expr: ANFExpr<f64>,
    ) -> ANFExpr<f64> {
        match (first, second) {
            (None, None) => final_expr,
            (Some(first_lets), None) => self.wrap_anf_lets(Some(first_lets), final_expr),
            (None, Some(second_lets)) => self.wrap_anf_lets(Some(second_lets), final_expr),
            (Some(first_lets), Some(second_lets)) => {
                let combined = self.wrap_anf_lets(Some(second_lets), final_expr);
                self.wrap_anf_lets(Some(first_lets), combined)
            }
        }
    }

    /// Wrap an expression with optional let bindings
    fn wrap_anf_lets(&self, wrapper: Option<ANFExpr<f64>>, body: ANFExpr<f64>) -> ANFExpr<f64> {
        match wrapper {
            None => body,
            Some(ANFExpr::Let(var, comp, _)) => {
                // Replace the body of the let with our new body
                ANFExpr::Let(var, comp, Box::new(body))
            }
            Some(other) => {
                // If it's not a let, just return the body
                // This shouldn't happen in normal usage
                body
            }
        }
    }

    /// Reset the bridge state for a new conversion
    pub fn reset(&mut self) {
        self.next_bound_id = 0;
        self.bound_id_mapping.clear();
    }
}

impl Default for ANFEgglogBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for standalone conversions
impl ANFEgglogBridge {
    /// Convert ANF to Egglog Math (standalone function)
    pub fn convert_anf_to_egglog(anf: &ANFExpr<f64>) -> EgglogMath {
        let mut bridge = ANFEgglogBridge::new();
        bridge.anf_to_egglog(anf)
    }

    /// Convert Egglog Math to ANF (standalone function)
    pub fn convert_egglog_to_anf(math: &EgglogMath) -> Result<ANFExpr<f64>> {
        let mut bridge = ANFEgglogBridge::new();
        bridge.egglog_to_anf(math)
    }
}
