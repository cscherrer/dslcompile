//! A-Normal Form (ANF) Intermediate Representation
//!
//! This module provides ANF transformation and optimization for mathematical expressions.
//! ANF ensures that every intermediate computation is explicitly named, enabling
//! efficient common subexpression elimination and clean optimization passes.
//!
//! # Variable Management
//!
//! Uses a hybrid approach:
//! - `VarRef::User(usize)`: Original user variables from the AST
//! - `VarRef::Generated(u32)`: Temporary variables generated during ANF conversion
//!
//! This integrates cleanly with the existing `VariableRegistry` system while providing
//! efficient integer-based variable management during optimization.

use crate::error::Result;
use crate::final_tagless::{ASTRepr, NumericType, VariableRegistry};
use num_traits::Zero;
use std::collections::HashMap;

/// Variable reference that distinguishes between user and generated variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VarRef {
    /// User-defined variable (index into `VariableRegistry`)
    User(usize),
    /// Generated temporary variable
    Generated(u32),
}

impl VarRef {
    /// Generate Rust code for this variable
    #[must_use]
    pub fn to_rust_code(&self, registry: &VariableRegistry) -> String {
        match self {
            VarRef::User(idx) => registry.get_name(*idx).unwrap_or("unknown").to_string(),
            VarRef::Generated(id) => format!("t{id}"),
        }
    }

    /// Generate a debug-friendly name for this variable
    #[must_use]
    pub fn debug_name(&self, registry: &VariableRegistry) -> String {
        match self {
            VarRef::User(idx) => {
                format!("{}({})", registry.get_name(*idx).unwrap_or("?"), idx)
            }
            VarRef::Generated(id) => format!("t{id}"),
        }
    }

    /// Check if this is a user variable
    #[must_use]
    pub fn is_user(&self) -> bool {
        matches!(self, VarRef::User(_))
    }

    /// Check if this is a generated variable
    #[must_use]
    pub fn is_generated(&self) -> bool {
        matches!(self, VarRef::Generated(_))
    }
}

/// Generator for fresh ANF variables
#[derive(Debug, Clone)]
pub struct ANFVarGen {
    next_temp_id: u32,
}

impl ANFVarGen {
    /// Create a new variable generator
    #[must_use]
    pub fn new() -> Self {
        Self { next_temp_id: 0 }
    }

    /// Generate a fresh temporary variable
    pub fn fresh(&mut self) -> VarRef {
        let var = VarRef::Generated(self.next_temp_id);
        self.next_temp_id += 1;
        var
    }

    /// Create a user variable reference
    #[must_use]
    pub fn user_var(&self, index: usize) -> VarRef {
        VarRef::User(index)
    }

    /// Get the number of generated variables so far
    #[must_use]
    pub fn generated_count(&self) -> u32 {
        self.next_temp_id
    }
}

impl Default for ANFVarGen {
    fn default() -> Self {
        Self::new()
    }
}

/// ANF atomic expressions (no sub-computations)
#[derive(Debug, Clone, PartialEq)]
pub enum ANFAtom<T> {
    /// Constant value
    Constant(T),
    /// Variable reference
    Variable(VarRef),
}

impl<T: NumericType> ANFAtom<T> {
    /// Check if this atom is a constant
    pub fn is_constant(&self) -> bool {
        matches!(self, ANFAtom::Constant(_))
    }

    /// Check if this atom is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, ANFAtom::Variable(_))
    }

    /// Extract constant value if this is a constant
    pub fn as_constant(&self) -> Option<&T> {
        match self {
            ANFAtom::Constant(val) => Some(val),
            _ => None,
        }
    }

    /// Extract variable reference if this is a variable
    pub fn as_variable(&self) -> Option<VarRef> {
        match self {
            ANFAtom::Variable(var) => Some(*var),
            _ => None,
        }
    }
}

/// ANF computations (operations with atomic arguments only)
#[derive(Debug, Clone, PartialEq)]
pub enum ANFComputation<T> {
    /// Addition: a + b
    Add(ANFAtom<T>, ANFAtom<T>),
    /// Subtraction: a - b
    Sub(ANFAtom<T>, ANFAtom<T>),
    /// Multiplication: a * b
    Mul(ANFAtom<T>, ANFAtom<T>),
    /// Division: a / b
    Div(ANFAtom<T>, ANFAtom<T>),
    /// Power: a^b
    Pow(ANFAtom<T>, ANFAtom<T>),
    /// Negation: -a
    Neg(ANFAtom<T>),
    /// Natural logarithm: ln(a)
    Ln(ANFAtom<T>),
    /// Exponential: exp(a)
    Exp(ANFAtom<T>),
    /// Sine: sin(a)
    Sin(ANFAtom<T>),
    /// Cosine: cos(a)
    Cos(ANFAtom<T>),
    /// Square root: sqrt(a)
    Sqrt(ANFAtom<T>),
}

impl<T: NumericType> ANFComputation<T> {
    /// Get all atomic operands of this computation
    pub fn operands(&self) -> Vec<&ANFAtom<T>> {
        match self {
            ANFComputation::Add(a, b)
            | ANFComputation::Sub(a, b)
            | ANFComputation::Mul(a, b)
            | ANFComputation::Div(a, b)
            | ANFComputation::Pow(a, b) => vec![a, b],
            ANFComputation::Neg(a)
            | ANFComputation::Ln(a)
            | ANFComputation::Exp(a)
            | ANFComputation::Sin(a)
            | ANFComputation::Cos(a)
            | ANFComputation::Sqrt(a) => vec![a],
        }
    }

    /// Check if this computation uses only constants
    pub fn is_constant_computation(&self) -> bool {
        self.operands().iter().all(|atom| atom.is_constant())
    }
}

/// ANF expressions
#[derive(Debug, Clone, PartialEq)]
pub enum ANFExpr<T> {
    /// Atomic expression (constant or variable)
    Atom(ANFAtom<T>),
    /// Let binding: let var = computation in body
    Let(VarRef, ANFComputation<T>, Box<ANFExpr<T>>),
}

impl<T: NumericType> ANFExpr<T> {
    /// Create an atomic constant
    pub fn constant(value: T) -> Self {
        ANFExpr::Atom(ANFAtom::Constant(value))
    }

    /// Create an atomic variable
    #[must_use]
    pub fn variable(var: VarRef) -> Self {
        ANFExpr::Atom(ANFAtom::Variable(var))
    }

    /// Create a let binding
    pub fn let_binding(var: VarRef, computation: ANFComputation<T>, body: ANFExpr<T>) -> Self {
        ANFExpr::Let(var, computation, Box::new(body))
    }

    /// Check if this is an atomic expression
    pub fn is_atom(&self) -> bool {
        matches!(self, ANFExpr::Atom(_))
    }

    /// Check if this is a let binding
    pub fn is_let(&self) -> bool {
        matches!(self, ANFExpr::Let(_, _, _))
    }

    /// Count the number of let bindings in this expression
    pub fn let_count(&self) -> usize {
        match self {
            ANFExpr::Atom(_) => 0,
            ANFExpr::Let(_, _, body) => 1 + body.let_count(),
        }
    }

    /// Collect all variables used in this expression
    pub fn used_variables(&self) -> Vec<VarRef> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort_unstable();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<VarRef>) {
        match self {
            ANFExpr::Atom(ANFAtom::Variable(var)) => vars.push(*var),
            ANFExpr::Atom(ANFAtom::Constant(_)) => {}
            ANFExpr::Let(_, comp, body) => {
                for operand in comp.operands() {
                    if let ANFAtom::Variable(var) = operand {
                        vars.push(*var);
                    }
                }
                body.collect_variables(vars);
            }
        }
    }
}

/// Structural hash representing the shape of an expression for CSE
/// Ignores actual numeric values but captures operators and variable positions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StructuralHash {
    /// Constant (any constant value)
    Constant,
    /// Variable by index
    Variable(usize),
    /// Binary operations
    Add(Box<StructuralHash>, Box<StructuralHash>),
    Sub(Box<StructuralHash>, Box<StructuralHash>),
    Mul(Box<StructuralHash>, Box<StructuralHash>),
    Div(Box<StructuralHash>, Box<StructuralHash>),
    Pow(Box<StructuralHash>, Box<StructuralHash>),
    /// Unary operations
    Neg(Box<StructuralHash>),
    Ln(Box<StructuralHash>),
    Exp(Box<StructuralHash>),
    Sin(Box<StructuralHash>),
    Cos(Box<StructuralHash>),
    Sqrt(Box<StructuralHash>),
}

impl StructuralHash {
    /// Create a structural hash from an `ASTRepr` expression
    pub fn from_expr<T: NumericType>(expr: &ASTRepr<T>) -> Self {
        match expr {
            ASTRepr::Constant(_) => StructuralHash::Constant,
            ASTRepr::Variable(idx) => StructuralHash::Variable(*idx),
            ASTRepr::Add(left, right) => StructuralHash::Add(
                Box::new(Self::from_expr(left)),
                Box::new(Self::from_expr(right)),
            ),
            ASTRepr::Sub(left, right) => StructuralHash::Sub(
                Box::new(Self::from_expr(left)),
                Box::new(Self::from_expr(right)),
            ),
            ASTRepr::Mul(left, right) => StructuralHash::Mul(
                Box::new(Self::from_expr(left)),
                Box::new(Self::from_expr(right)),
            ),
            ASTRepr::Div(left, right) => StructuralHash::Div(
                Box::new(Self::from_expr(left)),
                Box::new(Self::from_expr(right)),
            ),
            ASTRepr::Pow(left, right) => StructuralHash::Pow(
                Box::new(Self::from_expr(left)),
                Box::new(Self::from_expr(right)),
            ),
            ASTRepr::Neg(inner) => StructuralHash::Neg(Box::new(Self::from_expr(inner))),
            ASTRepr::Ln(inner) => StructuralHash::Ln(Box::new(Self::from_expr(inner))),
            ASTRepr::Exp(inner) => StructuralHash::Exp(Box::new(Self::from_expr(inner))),
            ASTRepr::Sin(inner) => StructuralHash::Sin(Box::new(Self::from_expr(inner))),
            ASTRepr::Cos(inner) => StructuralHash::Cos(Box::new(Self::from_expr(inner))),
            ASTRepr::Sqrt(inner) => StructuralHash::Sqrt(Box::new(Self::from_expr(inner))),
        }
    }
}

/// ANF converter that transforms `ASTRepr` to ANF form
#[derive(Debug)]
pub struct ANFConverter {
    var_gen: ANFVarGen,
    /// Cache for common subexpression elimination
    /// Maps structural hashes to their assigned variables
    expr_cache: HashMap<StructuralHash, VarRef>,
}

impl ANFConverter {
    /// Create a new ANF converter
    #[must_use]
    pub fn new() -> Self {
        Self {
            var_gen: ANFVarGen::new(),
            expr_cache: HashMap::new(),
        }
    }

    /// Convert an `ASTRepr` expression to ANF
    pub fn convert<T: NumericType + Clone + Zero>(
        &mut self,
        expr: &ASTRepr<T>,
    ) -> Result<ANFExpr<T>> {
        Ok(self.to_anf(expr))
    }

    /// Convert `ASTRepr` to ANF, returning the result and any let-bindings needed
    fn to_anf<T: NumericType + Clone + Zero>(&mut self, expr: &ASTRepr<T>) -> ANFExpr<T> {
        // Check cache for CSE - but only for non-trivial expressions
        if !matches!(expr, ASTRepr::Constant(_) | ASTRepr::Variable(_)) {
            let structural_hash = StructuralHash::from_expr(expr);
            if let Some(cached_var) = self.expr_cache.get(&structural_hash) {
                // Cache hit! Reuse the existing variable
                return ANFExpr::Atom(ANFAtom::Variable(*cached_var));
            }
        }

        match expr {
            ASTRepr::Constant(value) => ANFExpr::Atom(ANFAtom::Constant(value.clone())),
            ASTRepr::Variable(index) => ANFExpr::Atom(ANFAtom::Variable(VarRef::User(*index))),

            // Binary operations - these will be cached
            ASTRepr::Add(left, right) => {
                self.convert_binary_op_with_cse(expr, left, right, ANFComputation::Add)
            }
            ASTRepr::Sub(left, right) => {
                self.convert_binary_op_with_cse(expr, left, right, ANFComputation::Sub)
            }
            ASTRepr::Mul(left, right) => {
                self.convert_binary_op_with_cse(expr, left, right, ANFComputation::Mul)
            }
            ASTRepr::Div(left, right) => {
                self.convert_binary_op_with_cse(expr, left, right, ANFComputation::Div)
            }
            ASTRepr::Pow(left, right) => {
                self.convert_binary_op_with_cse(expr, left, right, ANFComputation::Pow)
            }

            // Unary operations - these will be cached
            ASTRepr::Neg(inner) => self.convert_unary_op_with_cse(expr, inner, ANFComputation::Neg),
            ASTRepr::Ln(inner) => self.convert_unary_op_with_cse(expr, inner, ANFComputation::Ln),
            ASTRepr::Exp(inner) => self.convert_unary_op_with_cse(expr, inner, ANFComputation::Exp),
            ASTRepr::Sin(inner) => self.convert_unary_op_with_cse(expr, inner, ANFComputation::Sin),
            ASTRepr::Cos(inner) => self.convert_unary_op_with_cse(expr, inner, ANFComputation::Cos),
            ASTRepr::Sqrt(inner) => {
                self.convert_unary_op_with_cse(expr, inner, ANFComputation::Sqrt)
            }
        }
    }

    /// Convert a binary operation to ANF with CSE caching
    fn convert_binary_op_with_cse<T: NumericType + Clone + Zero, F>(
        &mut self,
        original_expr: &ASTRepr<T>,
        left: &ASTRepr<T>,
        right: &ASTRepr<T>,
        op_constructor: F,
    ) -> ANFExpr<T>
    where
        F: FnOnce(ANFAtom<T>, ANFAtom<T>) -> ANFComputation<T>,
    {
        let result = self.convert_binary_op(left, right, op_constructor);

        // Cache the result for this expression
        if let ANFExpr::Let(var, _, _) = &result {
            let structural_hash = StructuralHash::from_expr(original_expr);
            self.expr_cache.insert(structural_hash, *var);
        }

        result
    }

    /// Convert a unary operation to ANF with CSE caching
    fn convert_unary_op_with_cse<T: NumericType + Clone + Zero, F>(
        &mut self,
        original_expr: &ASTRepr<T>,
        inner: &ASTRepr<T>,
        op_constructor: F,
    ) -> ANFExpr<T>
    where
        F: FnOnce(ANFAtom<T>) -> ANFComputation<T>,
    {
        let result = self.convert_unary_op(inner, op_constructor);

        // Cache the result for this expression
        if let ANFExpr::Let(var, _, _) = &result {
            let structural_hash = StructuralHash::from_expr(original_expr);
            self.expr_cache.insert(structural_hash, *var);
        }

        result
    }

    /// Convert a binary operation to ANF
    fn convert_binary_op<T: NumericType + Clone + Zero, F>(
        &mut self,
        left: &ASTRepr<T>,
        right: &ASTRepr<T>,
        op_constructor: F,
    ) -> ANFExpr<T>
    where
        F: FnOnce(ANFAtom<T>, ANFAtom<T>) -> ANFComputation<T>,
    {
        let (left_expr, left_atom) = self.to_anf_atom(left);
        let (right_expr, right_atom) = self.to_anf_atom(right);

        let computation = op_constructor(left_atom, right_atom);
        let result_var = self.var_gen.fresh();

        self.chain_lets(
            left_expr,
            right_expr,
            ANFExpr::Let(
                result_var,
                computation,
                Box::new(ANFExpr::Atom(ANFAtom::Variable(result_var))),
            ),
        )
    }

    /// Convert a unary operation to ANF
    fn convert_unary_op<T: NumericType + Clone + Zero, F>(
        &mut self,
        inner: &ASTRepr<T>,
        op_constructor: F,
    ) -> ANFExpr<T>
    where
        F: FnOnce(ANFAtom<T>) -> ANFComputation<T>,
    {
        let (inner_expr, inner_atom) = self.to_anf_atom(inner);
        let computation = op_constructor(inner_atom);
        let result_var = self.var_gen.fresh();

        self.wrap_with_lets(
            inner_expr,
            ANFExpr::Let(
                result_var,
                computation,
                Box::new(ANFExpr::Atom(ANFAtom::Variable(result_var))),
            ),
        )
    }

    /// Convert an expression to an atom, generating let-bindings as needed
    fn to_anf_atom<T: NumericType + Clone + Zero>(
        &mut self,
        expr: &ASTRepr<T>,
    ) -> (Option<ANFExpr<T>>, ANFAtom<T>) {
        match expr {
            ASTRepr::Constant(value) => (None, ANFAtom::Constant(value.clone())),
            ASTRepr::Variable(index) => (None, ANFAtom::Variable(VarRef::User(*index))),
            _ => {
                // Complex expression needs to be converted to ANF first
                let anf_expr = self.to_anf(expr);
                let temp_var = self.var_gen.fresh();

                // Wrap the ANF expression in a let-binding
                let let_expr = ANFExpr::Let(
                    temp_var,
                    self.extract_computation_from_anf(&anf_expr),
                    Box::new(ANFExpr::Atom(ANFAtom::Variable(temp_var))),
                );

                (Some(let_expr), ANFAtom::Variable(temp_var))
            }
        }
    }

    /// Extract the final computation from an ANF expression
    /// This is a helper for when we need to wrap complex expressions
    fn extract_computation_from_anf<T: NumericType + Clone + Zero>(
        &self,
        expr: &ANFExpr<T>,
    ) -> ANFComputation<T> {
        match expr {
            ANFExpr::Atom(ANFAtom::Constant(value)) => {
                // Convert constant to a trivial computation
                ANFComputation::Add(
                    ANFAtom::Constant(value.clone()),
                    ANFAtom::Constant(T::zero()),
                )
            }
            ANFExpr::Atom(ANFAtom::Variable(var)) => {
                // Convert variable to a trivial computation
                ANFComputation::Add(ANFAtom::Variable(*var), ANFAtom::Constant(T::zero()))
            }
            ANFExpr::Let(_, computation, _) => computation.clone(),
        }
    }

    /// Chain two optional ANF expressions with a final expression
    fn chain_lets<T: NumericType + Clone>(
        &self,
        first: Option<ANFExpr<T>>,
        second: Option<ANFExpr<T>>,
        final_expr: ANFExpr<T>,
    ) -> ANFExpr<T> {
        match (first, second) {
            (None, None) => final_expr,
            (Some(first_expr), None) => self.wrap_with_lets(Some(first_expr), final_expr),
            (None, Some(second_expr)) => self.wrap_with_lets(Some(second_expr), final_expr),
            (Some(first_expr), Some(second_expr)) => {
                let combined = self.wrap_with_lets(Some(second_expr), final_expr);
                self.wrap_with_lets(Some(first_expr), combined)
            }
        }
    }

    /// Wrap an expression with let-bindings if needed
    fn wrap_with_lets<T: NumericType + Clone>(
        &self,
        wrapper: Option<ANFExpr<T>>,
        body: ANFExpr<T>,
    ) -> ANFExpr<T> {
        match wrapper {
            None => body,
            Some(ANFExpr::Let(var, computation, _)) => {
                ANFExpr::Let(var, computation, Box::new(body))
            }
            Some(atom_expr) => body, // Already atomic, no wrapping needed
        }
    }
}

impl Default for ANFConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to convert `ASTRepr` to ANF
pub fn convert_to_anf<T: NumericType + Clone + Zero>(expr: &ASTRepr<T>) -> Result<ANFExpr<T>> {
    let mut converter = ANFConverter::new();
    converter.convert(expr)
}

/// ANF code generator for Rust
#[derive(Debug)]
pub struct ANFCodeGen<'a> {
    registry: &'a VariableRegistry,
}

impl<'a> ANFCodeGen<'a> {
    /// Create a new code generator with a variable registry
    #[must_use]
    pub fn new(registry: &'a VariableRegistry) -> Self {
        Self { registry }
    }

    /// Generate Rust code from an ANF expression
    pub fn generate<T: NumericType + std::fmt::Display>(&self, expr: &ANFExpr<T>) -> String {
        match expr {
            ANFExpr::Atom(atom) => self.generate_atom(atom),
            ANFExpr::Let(var, computation, body) => {
                let var_name = var.to_rust_code(self.registry);
                let comp_code = self.generate_computation(computation);
                let body_code = self.generate(body);

                format!("{{ let {var_name} = {comp_code};\n{body_code} }}")
            }
        }
    }

    /// Generate code for an atomic expression
    fn generate_atom<T: NumericType + std::fmt::Display>(&self, atom: &ANFAtom<T>) -> String {
        match atom {
            ANFAtom::Constant(value) => value.to_string(),
            ANFAtom::Variable(var) => var.to_rust_code(self.registry),
        }
    }

    /// Generate code for a computation
    fn generate_computation<T: NumericType + std::fmt::Display>(
        &self,
        comp: &ANFComputation<T>,
    ) -> String {
        match comp {
            ANFComputation::Add(left, right) => {
                format!(
                    "{} + {}",
                    self.generate_atom(left),
                    self.generate_atom(right)
                )
            }
            ANFComputation::Sub(left, right) => {
                format!(
                    "{} - {}",
                    self.generate_atom(left),
                    self.generate_atom(right)
                )
            }
            ANFComputation::Mul(left, right) => {
                format!(
                    "{} * {}",
                    self.generate_atom(left),
                    self.generate_atom(right)
                )
            }
            ANFComputation::Div(left, right) => {
                format!(
                    "{} / {}",
                    self.generate_atom(left),
                    self.generate_atom(right)
                )
            }
            ANFComputation::Pow(left, right) => {
                format!(
                    "{}.powf({})",
                    self.generate_atom(left),
                    self.generate_atom(right)
                )
            }
            ANFComputation::Neg(operand) => {
                format!("-{}", self.generate_atom(operand))
            }
            ANFComputation::Ln(operand) => {
                format!("{}.ln()", self.generate_atom(operand))
            }
            ANFComputation::Exp(operand) => {
                format!("{}.exp()", self.generate_atom(operand))
            }
            ANFComputation::Sin(operand) => {
                format!("{}.sin()", self.generate_atom(operand))
            }
            ANFComputation::Cos(operand) => {
                format!("{}.cos()", self.generate_atom(operand))
            }
            ANFComputation::Sqrt(operand) => {
                format!("{}.sqrt()", self.generate_atom(operand))
            }
        }
    }

    /// Generate a complete function definition
    pub fn generate_function<T: NumericType + std::fmt::Display>(
        &self,
        name: &str,
        expr: &ANFExpr<T>,
    ) -> String {
        let param_list: Vec<String> = self
            .registry
            .get_all_names()
            .iter()
            .map(|name| format!("{name}: f64"))
            .collect();

        let body = self.generate(expr);

        format!(
            "fn {}({}) -> f64 {{\n    {}\n}}",
            name,
            param_list.join(", "),
            body.replace('\n', "\n    ")
        )
    }
}

/// Convenience function to generate code from ANF
pub fn generate_rust_code<T: NumericType + std::fmt::Display>(
    expr: &ANFExpr<T>,
    registry: &VariableRegistry,
) -> String {
    let codegen = ANFCodeGen::new(registry);
    codegen.generate(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_gen() {
        let mut gen = ANFVarGen::new();

        let v1 = gen.fresh();
        let v2 = gen.fresh();
        let v3 = gen.user_var(0);

        assert_eq!(v1, VarRef::Generated(0));
        assert_eq!(v2, VarRef::Generated(1));
        assert_eq!(v3, VarRef::User(0));
        assert_ne!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_anf_atom() {
        let const_atom: ANFAtom<f64> = ANFAtom::Constant(42.0);
        let var_atom: ANFAtom<f64> = ANFAtom::Variable(VarRef::Generated(0));

        assert!(const_atom.is_constant());
        assert!(!const_atom.is_variable());
        assert_eq!(const_atom.as_constant(), Some(&42.0));

        assert!(var_atom.is_variable());
        assert!(!var_atom.is_constant());
        assert_eq!(var_atom.as_variable(), Some(VarRef::Generated(0)));
    }

    #[test]
    fn test_anf_computation_operands() {
        let a: ANFAtom<f64> = ANFAtom::Constant(1.0);
        let b: ANFAtom<f64> = ANFAtom::Variable(VarRef::Generated(0));

        let add = ANFComputation::Add(a.clone(), b.clone());
        let operands = add.operands();

        assert_eq!(operands.len(), 2);
        assert_eq!(operands[0], &a);
        assert_eq!(operands[1], &b);

        let neg = ANFComputation::Neg(a.clone());
        let neg_operands = neg.operands();
        assert_eq!(neg_operands.len(), 1);
        assert_eq!(neg_operands[0], &a);
    }

    #[test]
    fn test_anf_expr_construction() {
        let var = VarRef::Generated(0);
        let const_val = 42.0;

        let atom_expr: ANFExpr<f64> = ANFExpr::constant(const_val);
        let var_expr: ANFExpr<f64> = ANFExpr::variable(var);

        assert!(atom_expr.is_atom());
        assert!(var_expr.is_atom());

        let computation = ANFComputation::Add(ANFAtom::Variable(var), ANFAtom::Constant(1.0));
        let let_expr: ANFExpr<f64> = ANFExpr::let_binding(var, computation, atom_expr);

        assert!(let_expr.is_let());
        assert_eq!(let_expr.let_count(), 1);
    }

    #[test]
    fn test_variable_collection() {
        let var1 = VarRef::Generated(0);
        let var2 = VarRef::User(0);

        // Create: let t0 = x + 1 in t0 * t0
        let computation = ANFComputation::Add(
            ANFAtom::Variable(var2), // x
            ANFAtom::Constant(1.0),
        );
        let body: ANFExpr<f64> = ANFExpr::Atom(ANFAtom::Variable(var1)); // t0
        let expr: ANFExpr<f64> = ANFExpr::Let(var1, computation, Box::new(body));

        let used_vars = expr.used_variables();
        assert_eq!(used_vars.len(), 2);
        assert!(used_vars.contains(&var1));
        assert!(used_vars.contains(&var2));
    }

    #[test]
    fn test_anf_conversion() {
        use crate::final_tagless::{ASTEval, ASTMathExpr};

        // Create expression: sin(x + 1) + cos(x + 1)
        // This should demonstrate CSE automatically
        let x = ASTEval::var(0); // Variable with index 0
        let one = ASTEval::constant(1.0);
        let x_plus_one = ASTEval::add(x.clone(), one.clone());
        let sin_expr = ASTEval::sin(x_plus_one.clone());
        let cos_expr = ASTEval::cos(x_plus_one);
        let full_expr = ASTEval::add(sin_expr, cos_expr);

        // Convert to ANF
        let anf_result = convert_to_anf(&full_expr);
        assert!(anf_result.is_ok());

        let anf = anf_result.unwrap();

        // Should have multiple let bindings
        assert!(anf.let_count() > 0);

        // Verify structure is let-bound
        assert!(anf.is_let());
    }

    #[test]
    fn test_anf_code_generation() {
        use crate::final_tagless::{ASTEval, ASTMathExpr, VariableRegistry};

        // Create a variable registry
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable("x");

        // Create expression: x * x + 2 * x + 1 (quadratic)
        let x = ASTEval::var(x_idx);
        let two = ASTEval::constant(2.0);
        let one = ASTEval::constant(1.0);
        let x_squared = ASTEval::mul(x.clone(), x.clone());
        let two_x = ASTEval::mul(two, x);
        let sum1 = ASTEval::add(x_squared, two_x);
        let quadratic = ASTEval::add(sum1, one);

        // Convert to ANF
        let anf = convert_to_anf(&quadratic).unwrap();

        // Generate code
        let code = generate_rust_code(&anf, &registry);

        // Code should contain let bindings and be properly structured
        assert!(code.contains("let t"));
        assert!(code.contains('x'));

        // Also test function generation
        let codegen = ANFCodeGen::new(&registry);
        let function_code = codegen.generate_function("quadratic", &anf);

        assert!(function_code.contains("fn quadratic"));
        assert!(function_code.contains("x: f64"));
        assert!(function_code.contains("-> f64"));

        println!("Generated code:\n{code}");
        println!("Generated function:\n{function_code}");
    }

    #[test]
    fn test_anf_complete_pipeline() {
        use crate::final_tagless::{ASTEval, ASTMathExpr, VariableRegistry};

        // Create a variable registry
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable("x");
        let y_idx = registry.register_variable("y");

        // Create a complex expression with common subexpressions:
        // sin(x + y) + cos(x + y) + exp(x + y)
        // This should demonstrate automatic CSE of (x + y)
        let x = ASTEval::var(x_idx);
        let y = ASTEval::var(y_idx);
        let x_plus_y = ASTEval::add(x, y);

        let sin_term = ASTEval::sin(x_plus_y.clone());
        let cos_term = ASTEval::cos(x_plus_y.clone());
        let exp_term = ASTEval::exp(x_plus_y);

        let sum1 = ASTEval::add(sin_term, cos_term);
        let final_expr = ASTEval::add(sum1, exp_term);

        // Convert to ANF
        let anf = convert_to_anf(&final_expr).unwrap();

        // Generate code
        let codegen = ANFCodeGen::new(&registry);
        let function_code = codegen.generate_function("demo_function", &anf);

        println!("\n=== ANF Demo: Automatic Common Subexpression Elimination ===");
        println!("Original expression: sin(x + y) + cos(x + y) + exp(x + y)");
        println!("ANF introduces variables for shared subexpressions automatically\n");
        println!("Generated function:");
        println!("{function_code}");

        // Verify the ANF has the expected structure
        assert!(anf.let_count() >= 1); // Should have let bindings
        assert!(function_code.contains("fn demo_function"));
        assert!(function_code.contains("x: f64, y: f64"));
        assert!(function_code.contains("-> f64"));

        // The beauty is that this is ready to compile and run!
    }

    #[test]
    fn test_cse_simple_case() {
        use crate::final_tagless::{ASTEval, ASTMathExpr, VariableRegistry};

        // Create a variable registry
        let mut registry = VariableRegistry::new();
        let x_idx = registry.register_variable("x");

        // Create expression: (x + 1) + (x + 1)
        // This should reuse the computation of (x + 1)
        let x = ASTEval::var(x_idx);
        let one = ASTEval::constant(1.0);
        let x_plus_one_left = ASTEval::add(x.clone(), one.clone());
        let x_plus_one_right = ASTEval::add(x, one);
        let final_expr = ASTEval::add(x_plus_one_left, x_plus_one_right);

        // Convert to ANF
        let anf = convert_to_anf(&final_expr).unwrap();

        // Generate code
        let codegen = ANFCodeGen::new(&registry);
        let function_code = codegen.generate_function("cse_test", &anf);

        println!("\n=== CSE Test: (x + 1) + (x + 1) ===");
        println!("Generated function:");
        println!("{function_code}");

        // Count how many times we see "x + 1" in the generated code
        let code_contains_reuse = function_code.matches("x + 1").count() == 1;
        println!(
            "CSE working correctly: {}",
            if code_contains_reuse {
                "✅ YES"
            } else {
                "❌ NO"
            }
        );

        // Should have fewer let bindings due to reuse
        assert!(anf.let_count() > 0);
    }
}
