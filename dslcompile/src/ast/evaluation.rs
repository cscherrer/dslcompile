//! AST Evaluation Utilities
//!
//! This module provides efficient evaluation methods for AST expressions,
//! including optimized variable handling and specialized evaluation functions.

use crate::ast::{
    Scalar,
    ast_repr::{ASTRepr, Collection, Lambda},
};
use num_traits::{Float, FromPrimitive, One, Zero};

/// Work items for heap-allocated stack-based evaluation
#[derive(Debug, Clone)]
enum EvalWorkItem<T: Scalar + Clone> {
    /// Evaluate an expression and push result to value stack
    Eval(ASTRepr<T>),
    /// Apply binary operation to top two values on stack
    ApplyBinary(BinaryOp),
    /// Apply unary operation to top value on stack
    ApplyUnary(UnaryOp),
    /// Apply n-ary operation to top n values on stack
    ApplyNary(NaryOp, usize),
    /// Evaluate collection sum
    EvalCollectionSum(Collection<T>),
}

#[derive(Debug, Clone)]
enum BinaryOp {
    Sub,
    Div,
    Pow,
}

#[derive(Debug, Clone)]
enum NaryOp {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
enum UnaryOp {
    Neg,
    Ln,
    Exp,
    Sin,
    Cos,
    Sqrt,
}

/// Optimized evaluation methods for AST expressions
impl<T> ASTRepr<T>
where
    T: Scalar + Float + Copy + FromPrimitive + Zero,
{
    /// Evaluate the expression with given variable values using heap-allocated stack
    ///
    /// This method uses an explicit heap-allocated Vec as a stack to avoid
    /// call stack overflow on deep expressions. No recursion = no stack overflow!
    #[must_use]
    pub fn eval_with_vars(&self, variables: &[T]) -> T {
        let mut work_stack: Vec<EvalWorkItem<T>> = Vec::new();
        let mut value_stack: Vec<T> = Vec::new();

        // Start with the root expression
        work_stack.push(EvalWorkItem::Eval(self.clone()));

        // Process work items until stack is empty - no recursion!
        while let Some(work_item) = work_stack.pop() {
            match work_item {
                EvalWorkItem::Eval(expr) => {
                    match expr {
                        ASTRepr::Constant(value) => {
                            value_stack.push(value);
                        }
                        ASTRepr::Variable(index) => {
                            if index < variables.len() {
                                value_stack.push(variables[index]);
                            } else {
                                panic!(
                                    "Variable index {index} is out of bounds for evaluation! \
                                       Tried to access variable at index {index}, but only {} variables provided. \
                                       Use a valid variable index or provide more variables.",
                                    variables.len()
                                )
                            }
                        }
                        ASTRepr::BoundVar(index) => {
                            if index < variables.len() {
                                value_stack.push(variables[index]);
                            } else {
                                panic!(
                                    "BoundVar index {index} is out of bounds for evaluation! \
                                       Tried to access variable at index {index}, but only {} variables provided.",
                                    variables.len()
                                )
                            }
                        }
                        // Multiset operations: evaluate all terms, then apply operation
                        ASTRepr::Add(terms) => {
                            if terms.is_empty() {
                                value_stack.push(T::zero());
                            } else if terms.len() == 1 {
                                work_stack.push(EvalWorkItem::Eval(
                                    terms.elements().next().unwrap().clone(),
                                ));
                            } else {
                                work_stack.push(EvalWorkItem::ApplyNary(NaryOp::Add, terms.len()));
                                // Push terms in reverse order for correct stack evaluation
                                let terms_vec: Vec<_> = terms.elements().collect();
                                for term in terms_vec.iter().rev() {
                                    work_stack.push(EvalWorkItem::Eval((*term).clone()));
                                }
                            }
                        }
                        ASTRepr::Sub(left, right) => {
                            work_stack.push(EvalWorkItem::ApplyBinary(BinaryOp::Sub));
                            work_stack.push(EvalWorkItem::Eval(*right));
                            work_stack.push(EvalWorkItem::Eval(*left));
                        }
                        ASTRepr::Mul(factors) => {
                            if factors.is_empty() {
                                value_stack.push(T::one());
                            } else if factors.len() == 1 {
                                work_stack.push(EvalWorkItem::Eval(
                                    factors.elements().next().unwrap().clone(),
                                ));
                            } else {
                                work_stack
                                    .push(EvalWorkItem::ApplyNary(NaryOp::Mul, factors.len()));
                                // Push factors in reverse order for correct stack evaluation
                                let factors_vec: Vec<_> = factors.elements().collect();
                                for factor in factors_vec.iter().rev() {
                                    work_stack.push(EvalWorkItem::Eval((*factor).clone()));
                                }
                            }
                        }
                        ASTRepr::Div(left, right) => {
                            work_stack.push(EvalWorkItem::ApplyBinary(BinaryOp::Div));
                            work_stack.push(EvalWorkItem::Eval(*right));
                            work_stack.push(EvalWorkItem::Eval(*left));
                        }
                        ASTRepr::Pow(left, right) => {
                            work_stack.push(EvalWorkItem::ApplyBinary(BinaryOp::Pow));
                            work_stack.push(EvalWorkItem::Eval(*right));
                            work_stack.push(EvalWorkItem::Eval(*left));
                        }
                        // Unary operations: push operation, then child
                        ASTRepr::Neg(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Neg));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Ln(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Ln));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Exp(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Exp));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Sin(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Sin));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Cos(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Cos));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Sqrt(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Sqrt));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        // Complex operations
                        ASTRepr::Sum(collection) => {
                            work_stack.push(EvalWorkItem::EvalCollectionSum(*collection));
                        }
                        ASTRepr::Let(_, expr, body) => {
                            // TODO: Proper Let evaluation would substitute the bound variable
                            // For now, just evaluate the body with current variables
                            // We evaluate expr first but don't use it yet
                            work_stack.push(EvalWorkItem::Eval(*body));
                            work_stack.push(EvalWorkItem::Eval(*expr));
                        }
                        ASTRepr::Lambda(lambda) => {
                            // Lambda evaluation without arguments
                            if lambda.var_indices.is_empty() {
                                work_stack.push(EvalWorkItem::Eval(*lambda.body));
                            } else {
                                panic!("Cannot evaluate lambda without function application")
                            }
                        }
                    }
                }
                EvalWorkItem::ApplyBinary(op) => {
                    // Pop two values and apply binary operation
                    let right = value_stack
                        .pop()
                        .expect("Missing right operand for binary operation");
                    let left = value_stack
                        .pop()
                        .expect("Missing left operand for binary operation");
                    let result = match op {
                        BinaryOp::Sub => left - right,
                        BinaryOp::Div => left / right,
                        BinaryOp::Pow => left.powf(right),
                    };
                    value_stack.push(result);
                }
                EvalWorkItem::ApplyNary(op, count) => {
                    // Pop n values and apply n-ary operation
                    let mut operands = Vec::with_capacity(count);
                    for _ in 0..count {
                        let value = value_stack
                            .pop()
                            .expect("Missing operand for n-ary operation");
                        operands.push(value);
                    }
                    // Reverse to get original order (since we popped in reverse)
                    operands.reverse();

                    let result = match op {
                        NaryOp::Add => operands.into_iter().fold(T::zero(), |acc, x| acc + x),
                        NaryOp::Mul => operands.into_iter().fold(T::one(), |acc, x| acc * x),
                    };
                    value_stack.push(result);
                }
                EvalWorkItem::ApplyUnary(op) => {
                    // Pop one value and apply unary operation
                    let value = value_stack
                        .pop()
                        .expect("Missing operand for unary operation");
                    let result = match op {
                        UnaryOp::Neg => -value,
                        UnaryOp::Ln => value.ln(),
                        UnaryOp::Exp => value.exp(),
                        UnaryOp::Sin => value.sin(),
                        UnaryOp::Cos => value.cos(),
                        UnaryOp::Sqrt => value.sqrt(),
                    };
                    value_stack.push(result);
                }
                EvalWorkItem::EvalCollectionSum(collection) => {
                    let sum_result = self.eval_collection_sum_stack_based(&collection, variables);
                    value_stack.push(sum_result);
                }
            }
        }

        // Should have exactly one result
        value_stack
            .pop()
            .expect("Evaluation completed but no result on stack")
    }

    /// Stack-based collection sum evaluation (also non-recursive)
    fn eval_collection_sum_stack_based(&self, collection: &Collection<T>, variables: &[T]) -> T {
        match collection {
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => {
                // Use stack-based evaluation to avoid recursion
                expr.eval_with_vars(variables)
            }
            Collection::Range { start, end } => {
                // Evaluate range bounds using stack-based evaluation
                let start_val = start.eval_with_vars(variables);
                let end_val = end.eval_with_vars(variables);

                // Convert to integers for iteration
                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                // Sum over the mathematical range with identity function
                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            }
            Collection::Variable(_data_var) => {
                // TODO: Data array evaluation requires runtime data binding
                // For now, return zero as placeholder
                T::zero()
            }
            Collection::DataArray(data) => {
                // Sum directly over embedded data array
                data.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            Collection::Filter {
                collection: _,
                predicate: _,
            } => {
                // TODO: Implement filtered collection evaluation
                T::zero()
            }
            Collection::Map { lambda, collection } => {
                self.eval_mapped_collection_stack_based(lambda, collection, variables)
            }
        }
    }

    /// Stack-based mapped collection evaluation
    fn eval_mapped_collection_stack_based(
        &self,
        lambda: &Lambda<T>,
        collection: &Collection<T>,
        variables: &[T],
    ) -> T {
        match collection {
            Collection::Empty => T::zero(),
            Collection::Singleton(expr) => {
                let element_val = expr.eval_with_vars(variables);
                self.eval_lambda_stack_based(lambda, element_val, variables)
            }
            Collection::Range { start, end } => {
                // Evaluate range bounds using stack-based evaluation
                let start_val = start.eval_with_vars(variables);
                let end_val = end.eval_with_vars(variables);

                // Convert to integers for iteration
                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                // Sum lambda(i) for i in range
                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    let lambda_result = self.eval_lambda_stack_based(lambda, i_val, variables);
                    sum = sum + lambda_result;
                }
                sum
            }
            Collection::Variable(_data_var) => {
                // TODO: Data array evaluation with lambda mapping
                T::zero()
            }
            Collection::DataArray(data) => {
                // Apply lambda to each element in the data array
                data.iter()
                    .map(|&x| self.eval_lambda_stack_based(lambda, x, variables))
                    .fold(T::zero(), |acc, x| acc + x)
            }
            Collection::Filter {
                collection: _,
                predicate: _,
            } => {
                // TODO: Implement filtered mapping
                T::zero()
            }
            Collection::Map {
                lambda: inner_lambda,
                collection: inner_collection,
            } => {
                // Apply the lambda over the inner mapped collection
                let inner_result = self.eval_mapped_collection_stack_based(
                    inner_lambda,
                    inner_collection,
                    variables,
                );
                self.eval_lambda_stack_based(lambda, inner_result, variables)
            }
        }
    }

    /// Stack-based lambda evaluation
    fn eval_lambda_stack_based(&self, lambda: &Lambda<T>, value: T, variables: &[T]) -> T {
        // For single-value application, bind the first variable if available
        if lambda.var_indices.is_empty() {
            // Constant lambda - just evaluate the body using stack-based evaluation
            lambda.body.eval_with_vars(variables)
        } else {
            // For lambda evaluation, we need to handle BoundVar specially
            // BoundVar(0) in the lambda body should be substituted with the input value
            // We create a special evaluation context where BoundVar(0) = value
            self.eval_lambda_body_with_bound_value(&lambda.body, value, variables)
        }
    }

    /// Evaluate lambda body with a bound value for BoundVar(0)
    fn eval_lambda_body_with_bound_value(
        &self,
        body: &ASTRepr<T>,
        bound_value: T,
        variables: &[T],
    ) -> T {
        let mut work_stack: Vec<EvalWorkItem<T>> = Vec::new();
        let mut value_stack: Vec<T> = Vec::new();

        // Start with the lambda body
        work_stack.push(EvalWorkItem::Eval(body.clone()));

        // Process work items until stack is empty
        while let Some(work_item) = work_stack.pop() {
            match work_item {
                EvalWorkItem::Eval(expr) => {
                    match expr {
                        ASTRepr::Constant(value) => {
                            value_stack.push(value);
                        }
                        ASTRepr::Variable(index) => {
                            if index < variables.len() {
                                value_stack.push(variables[index]);
                            } else {
                                panic!(
                                    "Variable index {index} is out of bounds for lambda evaluation! \
                                       Tried to access variable at index {index}, but only {} variables provided.",
                                    variables.len()
                                )
                            }
                        }
                        ASTRepr::BoundVar(index) => {
                            // BoundVar(0) gets the lambda argument value
                            if index == 0 {
                                value_stack.push(bound_value);
                            } else {
                                panic!(
                                    "BoundVar index {index} is not supported in lambda evaluation! \
                                       Only BoundVar(0) is supported for single-argument lambdas."
                                )
                            }
                        }
                        // Multiset operations: evaluate all terms, then apply operation
                        ASTRepr::Add(terms) => {
                            if terms.is_empty() {
                                value_stack.push(T::zero());
                            } else if terms.len() == 1 {
                                work_stack.push(EvalWorkItem::Eval(
                                    terms.elements().next().unwrap().clone(),
                                ));
                            } else {
                                work_stack.push(EvalWorkItem::ApplyNary(NaryOp::Add, terms.len()));
                                // Push terms in reverse order for correct stack evaluation
                                let terms_vec: Vec<_> = terms.elements().collect();
                                for term in terms_vec.iter().rev() {
                                    work_stack.push(EvalWorkItem::Eval((*term).clone()));
                                }
                            }
                        }
                        ASTRepr::Sub(left, right) => {
                            work_stack.push(EvalWorkItem::ApplyBinary(BinaryOp::Sub));
                            work_stack.push(EvalWorkItem::Eval(*right));
                            work_stack.push(EvalWorkItem::Eval(*left));
                        }
                        ASTRepr::Mul(factors) => {
                            if factors.is_empty() {
                                value_stack.push(T::one());
                            } else if factors.len() == 1 {
                                work_stack.push(EvalWorkItem::Eval(
                                    factors.elements().next().unwrap().clone(),
                                ));
                            } else {
                                work_stack
                                    .push(EvalWorkItem::ApplyNary(NaryOp::Mul, factors.len()));
                                // Push factors in reverse order for correct stack evaluation
                                let factors_vec: Vec<_> = factors.elements().collect();
                                for factor in factors_vec.iter().rev() {
                                    work_stack.push(EvalWorkItem::Eval((*factor).clone()));
                                }
                            }
                        }
                        ASTRepr::Div(left, right) => {
                            work_stack.push(EvalWorkItem::ApplyBinary(BinaryOp::Div));
                            work_stack.push(EvalWorkItem::Eval(*right));
                            work_stack.push(EvalWorkItem::Eval(*left));
                        }
                        ASTRepr::Pow(left, right) => {
                            work_stack.push(EvalWorkItem::ApplyBinary(BinaryOp::Pow));
                            work_stack.push(EvalWorkItem::Eval(*right));
                            work_stack.push(EvalWorkItem::Eval(*left));
                        }
                        // Unary operations: push operation, then child
                        ASTRepr::Neg(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Neg));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Ln(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Ln));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Exp(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Exp));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Sin(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Sin));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Cos(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Cos));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        ASTRepr::Sqrt(inner) => {
                            work_stack.push(EvalWorkItem::ApplyUnary(UnaryOp::Sqrt));
                            work_stack.push(EvalWorkItem::Eval(*inner));
                        }
                        // Complex operations
                        ASTRepr::Sum(collection) => {
                            work_stack.push(EvalWorkItem::EvalCollectionSum(*collection));
                        }
                        ASTRepr::Let(_, expr, body) => {
                            // TODO: Proper Let evaluation would substitute the bound variable
                            // For now, just evaluate the body with current variables
                            work_stack.push(EvalWorkItem::Eval(*body));
                            work_stack.push(EvalWorkItem::Eval(*expr));
                        }
                        ASTRepr::Lambda(lambda) => {
                            // Lambda evaluation without arguments
                            if lambda.var_indices.is_empty() {
                                work_stack.push(EvalWorkItem::Eval(*lambda.body));
                            } else {
                                panic!("Cannot evaluate lambda without function application")
                            }
                        }
                    }
                }
                EvalWorkItem::ApplyBinary(op) => {
                    // Pop two values and apply binary operation
                    let right = value_stack
                        .pop()
                        .expect("Missing right operand for binary operation");
                    let left = value_stack
                        .pop()
                        .expect("Missing left operand for binary operation");
                    let result = match op {
                        BinaryOp::Sub => left - right,
                        BinaryOp::Div => left / right,
                        BinaryOp::Pow => left.powf(right),
                    };
                    value_stack.push(result);
                }
                EvalWorkItem::ApplyNary(op, count) => {
                    // Pop n values and apply n-ary operation
                    let mut operands = Vec::with_capacity(count);
                    for _ in 0..count {
                        let value = value_stack
                            .pop()
                            .expect("Missing operand for n-ary operation");
                        operands.push(value);
                    }
                    // Reverse to get original order (since we popped in reverse)
                    operands.reverse();

                    let result = match op {
                        NaryOp::Add => operands.into_iter().fold(T::zero(), |acc, x| acc + x),
                        NaryOp::Mul => operands.into_iter().fold(T::one(), |acc, x| acc * x),
                    };
                    value_stack.push(result);
                }
                EvalWorkItem::ApplyUnary(op) => {
                    // Pop one value and apply unary operation
                    let value = value_stack
                        .pop()
                        .expect("Missing operand for unary operation");
                    let result = match op {
                        UnaryOp::Neg => -value,
                        UnaryOp::Ln => value.ln(),
                        UnaryOp::Exp => value.exp(),
                        UnaryOp::Sin => value.sin(),
                        UnaryOp::Cos => value.cos(),
                        UnaryOp::Sqrt => value.sqrt(),
                    };
                    value_stack.push(result);
                }
                EvalWorkItem::EvalCollectionSum(collection) => {
                    let sum_result = self.eval_collection_sum_stack_based(&collection, variables);
                    value_stack.push(sum_result);
                }
            }
        }

        // Should have exactly one result
        value_stack
            .pop()
            .expect("Lambda body evaluation completed but no result on stack")
    }

    /// DEPRECATED: Old recursive implementation kept for compatibility
    /// Use `eval_with_vars()` instead - it now uses heap-allocated stack
    #[deprecated(
        note = "This method is now implemented using heap-allocated stack. Use eval_with_vars() directly."
    )]
    fn eval_with_vars_recursive(&self, variables: &[T]) -> T {
        self.eval_with_vars(variables)
    }

    /// DEPRECATED: Use `eval_collection_sum_stack_based` instead
    /// Evaluate a sum over a collection (now delegates to stack-based implementation)
    #[deprecated(note = "Use eval_collection_sum_stack_based for stack-safe evaluation")]
    fn eval_collection_sum(&self, collection: &Collection<T>, variables: &[T]) -> T {
        self.eval_collection_sum_stack_based(collection, variables)
    }

    /// DEPRECATED: Use `eval_mapped_collection_stack_based` instead
    /// Evaluate a mapped collection (now delegates to stack-based implementation)
    #[deprecated(note = "Use eval_mapped_collection_stack_based for stack-safe evaluation")]
    fn eval_mapped_collection(
        &self,
        lambda: &Lambda<T>,
        collection: &Collection<T>,
        variables: &[T],
    ) -> T {
        self.eval_mapped_collection_stack_based(lambda, collection, variables)
    }

    /// DEPRECATED: Use `eval_lambda_stack_based` instead
    /// Evaluate a lambda function (now delegates to stack-based implementation)
    #[deprecated(note = "Use eval_lambda_stack_based for stack-safe evaluation")]
    fn eval_lambda(&self, lambda: &Lambda<T>, value: T, variables: &[T]) -> T {
        self.eval_lambda_stack_based(lambda, value, variables)
    }

    /// Evaluate a two-variable expression with specific values
    #[must_use]
    pub fn eval_two_vars(&self, x: T, y: T) -> T {
        self.eval_with_vars(&[x, y])
    }

    /// Evaluate with a single variable value
    #[must_use]
    pub fn eval_one_var(&self, value: T) -> T {
        self.eval_with_vars(&[value])
    }

    /// Evaluate with no variables (constants only)
    #[must_use]
    pub fn eval_no_vars(&self) -> T {
        self.eval_with_vars(&[])
    }

    /// Evaluate expression with data arrays (for `DataArray` collections)
    #[must_use]
    pub(crate) fn eval_with_data(&self, params: &[T], data_arrays: &[Vec<T>]) -> T {
        match self {
            ASTRepr::Sum(collection) => {
                self.eval_collection_sum_with_data(collection, params, data_arrays)
            }
            _ => {
                // For non-sum expressions, use regular evaluation with params
                self.eval_with_vars(params)
            }
        }
    }

    /// Evaluate collection sum with data arrays
    fn eval_collection_sum_with_data(
        &self,
        collection: &Collection<T>,
        params: &[T],
        data_arrays: &[Vec<T>],
    ) -> T {
        match collection {
            Collection::Variable(data_var) => {
                // Sum over data array with identity function
                if *data_var < data_arrays.len() {
                    data_arrays[*data_var]
                        .iter()
                        .fold(T::zero(), |acc, &x| acc + x)
                } else {
                    T::zero()
                }
            }
            Collection::Map { lambda, collection } => {
                self.eval_mapped_collection_with_data(lambda, collection, params, data_arrays)
            }
            Collection::Range { start, end } => {
                // Mathematical ranges don't need data arrays
                let start_val = start.eval_with_vars(params);
                let end_val = end.eval_with_vars(params);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    sum = sum + i_val;
                }
                sum
            }
            Collection::DataArray(data) => {
                // Sum directly over embedded data array
                data.iter().fold(T::zero(), |acc, &x| acc + x)
            }
            _ => {
                // For other collection types, use regular evaluation
                self.eval_collection_sum(collection, params)
            }
        }
    }

    /// Evaluate mapped collection with data arrays
    fn eval_mapped_collection_with_data(
        &self,
        lambda: &Lambda<T>,
        collection: &Collection<T>,
        params: &[T],
        data_arrays: &[Vec<T>],
    ) -> T {
        match collection {
            Collection::Variable(data_var) => {
                // Map lambda over data array
                if *data_var < data_arrays.len() {
                    data_arrays[*data_var]
                        .iter()
                        .map(|&x| self.eval_lambda(lambda, x, params))
                        .fold(T::zero(), |acc, x| acc + x)
                } else {
                    T::zero()
                }
            }
            Collection::Range { start, end } => {
                // Map lambda over mathematical range
                let start_val = start.eval_with_vars(params);
                let end_val = end.eval_with_vars(params);

                let start_int = start_val.to_i64().unwrap_or(0);
                let end_int = end_val.to_i64().unwrap_or(0);

                let mut sum = T::zero();
                for i in start_int..=end_int {
                    let i_val = T::from(i).unwrap_or(T::zero());
                    let lambda_result = self.eval_lambda(lambda, i_val, params);
                    sum = sum + lambda_result;
                }
                sum
            }
            Collection::DataArray(data) => {
                // Apply lambda to each element in the embedded data array
                data.iter()
                    .map(|&x| self.eval_lambda(lambda, x, params))
                    .fold(T::zero(), |acc, x| acc + x)
            }
            _ => {
                // For other collection types, use regular evaluation
                self.eval_mapped_collection(lambda, collection, params)
            }
        }
    }
}

/// Specialized evaluation methods for f64 expressions
impl ASTRepr<f64> {
    /// Fast evaluation without heap allocation for two variables - now uses heap-allocated stack
    #[must_use]
    pub fn eval_two_vars_fast(expr: &ASTRepr<f64>, x: f64, y: f64) -> f64 {
        // For now, just delegate to the main eval_with_vars method which is already stack-safe
        // This eliminates the recursive calls while maintaining correctness
        expr.eval_with_vars(&[x, y])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficient_variable_indexing() {
        // Test efficient index-based variables
        let expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0), // x
            ASTRepr::Variable(1), // y
        ]);
        let result = expr.eval_with_vars(&[2.0, 3.0]);
        assert_eq!(result, 5.0);

        // Test multiplication with index-based variables
        let expr = ASTRepr::mul_from_array([
            ASTRepr::Variable(0), // x
            ASTRepr::Variable(1), // y
        ]);
        let result = expr.eval_with_vars(&[4.0, 5.0]);
        assert_eq!(result, 20.0);
    }

    #[test]
    #[should_panic(expected = "Variable index 10 is out of bounds")]
    fn test_out_of_bounds_variable_index() {
        // Test behavior when variable index is out of bounds - should panic
        let expr = ASTRepr::Variable(10); // Index 10, but only 2 variables provided
        let _result = expr.eval_with_vars(&[1.0, 2.0]); // Should panic!
    }

    #[test]
    fn test_two_variable_evaluation() {
        // Test two-variable evaluation: x + y
        let expr = ASTRepr::add_from_array([
            ASTRepr::Variable(0), // x
            ASTRepr::Variable(1), // y
        ]);
        let result = expr.eval_two_vars(3.0, 4.0);
        assert_eq!(result, 7.0);

        // Test more complex expression: x * y + 1
        let expr = ASTRepr::add_from_array([
            ASTRepr::mul_from_array([
                ASTRepr::Variable(0), // x
                ASTRepr::Variable(1), // y
            ]),
            ASTRepr::Constant(1.0),
        ]);
        let result = expr.eval_two_vars(2.0, 3.0);
        assert_eq!(result, 7.0); // 2 * 3 + 1 = 7
    }

    #[test]
    fn test_transcendental_evaluation() {
        // Test sine evaluation
        let expr = ASTRepr::Sin(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[0.0]);
        assert!((result - 0.0).abs() < 1e-10); // sin(0) = 0

        // Test exponential evaluation
        let expr = ASTRepr::Exp(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[0.0]);
        assert!((result - 1.0).abs() < 1e-10); // exp(0) = 1

        // Test natural logarithm evaluation
        let expr = ASTRepr::Ln(Box::new(ASTRepr::Variable(0)));
        let result = expr.eval_with_vars(&[1.0]);
        assert!((result - 0.0).abs() < 1e-10); // ln(1) = 0
    }

    #[test]
    fn test_power_evaluation() {
        // Test power evaluation: x^2
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        );
        let result = expr.eval_with_vars(&[3.0]);
        assert_eq!(result, 9.0); // 3^2 = 9

        // Test fractional power: x^0.5 (square root)
        let expr = ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(0.5)),
        );
        let result = expr.eval_with_vars(&[4.0]);
        assert!((result - 2.0).abs() < 1e-10); // 4^0.5 = 2
    }

    #[test]
    #[should_panic(expected = "Variable index 2 is out of bounds for evaluation")]
    fn test_two_vars_fast_out_of_bounds() {
        // Test that eval_two_vars_fast panics for Variable(2) and higher
        let expr = ASTRepr::Variable(2); // Index 2, but only supports 0 and 1
        let _result = ASTRepr::eval_two_vars_fast(&expr, 1.0, 2.0); // Should panic!
    }

    #[test]
    fn test_two_vars_fast_out_of_bounds_comprehensive() {
        let expr = ASTRepr::<f64>::Variable(2); // Index 2 is out of bounds for two variables
        let result = std::panic::catch_unwind(|| {
            ASTRepr::eval_two_vars_fast(&expr, 1.0, 2.0);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_one_var() {
        // Test simple variable access
        let x = ASTRepr::<f64>::Variable(0);
        assert_eq!(x.eval_one_var(5.0), 5.0);

        // Test constant
        let const_expr = ASTRepr::<f64>::Constant(42.0);
        assert_eq!(const_expr.eval_one_var(5.0), 42.0);

        // Test arithmetic with one variable
        let expr =
            ASTRepr::add_from_array([ASTRepr::<f64>::Variable(0), ASTRepr::<f64>::Constant(10.0)]);
        assert_eq!(expr.eval_one_var(5.0), 15.0);

        // Test transcendental functions
        let sin_expr = ASTRepr::Sin(Box::new(ASTRepr::<f64>::Variable(0)));
        assert!((sin_expr.eval_one_var(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_no_vars() {
        // Test constant expression
        let const_expr = ASTRepr::<f64>::Constant(3.14);
        assert_eq!(const_expr.eval_no_vars(), 3.14);

        // Test arithmetic with constants
        use crate::ast::multiset::MultiSet;
        let expr = ASTRepr::Add(MultiSet::from_iter([
            ASTRepr::<f64>::Constant(2.0),
            ASTRepr::<f64>::Constant(3.0),
        ]));
        assert_eq!(expr.eval_no_vars(), 5.0);

        // Test transcendental functions with constants
        let sin_expr = ASTRepr::Sin(Box::new(ASTRepr::<f64>::Constant(0.0)));
        assert!((sin_expr.eval_no_vars() - 0.0).abs() < 1e-10);

        // Test complex constant expression
        let complex_expr = ASTRepr::Mul(MultiSet::from_iter([
            ASTRepr::Add(MultiSet::from_iter([
                ASTRepr::<f64>::Constant(2.0),
                ASTRepr::<f64>::Constant(3.0),
            ])),
            ASTRepr::<f64>::Constant(4.0),
        ]));
        assert_eq!(complex_expr.eval_no_vars(), 20.0); // (2 + 3) * 4 = 20
    }

    #[test]
    fn test_division_evaluation() {
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);
        let div_expr = ASTRepr::Div(Box::new(x), Box::new(y));

        assert_eq!(div_expr.eval_with_vars(&[10.0, 2.0]), 5.0);
        assert_eq!(div_expr.eval_with_vars(&[1.0, 4.0]), 0.25);
    }

    #[test]
    fn test_negation_evaluation() {
        let x = ASTRepr::<f64>::Variable(0);
        let neg_expr = ASTRepr::Neg(Box::new(x));

        assert_eq!(neg_expr.eval_with_vars(&[5.0]), -5.0);
        assert_eq!(neg_expr.eval_with_vars(&[-3.0]), 3.0);
        assert_eq!(neg_expr.eval_with_vars(&[0.0]), 0.0);
    }

    #[test]
    fn test_sqrt_evaluation() {
        let x = ASTRepr::<f64>::Variable(0);
        let sqrt_expr = ASTRepr::Sqrt(Box::new(x));

        assert_eq!(sqrt_expr.eval_with_vars(&[9.0]), 3.0);
        assert_eq!(sqrt_expr.eval_with_vars(&[16.0]), 4.0);
        assert!((sqrt_expr.eval_with_vars(&[2.0]) - 1.4142135623730951).abs() < 1e-10);
    }

    #[test]
    fn test_complex_nested_evaluation() {
        use crate::ast::multiset::MultiSet;
        // Test (x + y) * (x - y) = x² - y²
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);

        let x_plus_y = ASTRepr::Add(MultiSet::from_iter([x.clone(), y.clone()]));
        let x_minus_y = ASTRepr::Sub(Box::new(x.clone()), Box::new(y.clone()));
        let expr = x_plus_y * x_minus_y;

        let result = expr.eval_with_vars(&[5.0, 3.0]);
        let expected = 5.0 * 5.0 - 3.0 * 3.0; // 25 - 9 = 16
        assert_eq!(result, expected);
    }

    #[test]
    fn test_trigonometric_identities() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test sin²(x) + cos²(x) = 1
        let sin_x = ASTRepr::Sin(Box::new(x.clone()));
        let cos_x = ASTRepr::Cos(Box::new(x.clone()));
        let sin_squared = ASTRepr::Pow(Box::new(sin_x), Box::new(ASTRepr::<f64>::Constant(2.0)));
        let cos_squared = ASTRepr::Pow(Box::new(cos_x), Box::new(ASTRepr::<f64>::Constant(2.0)));
        let identity = sin_squared + cos_squared;

        let result = identity.eval_with_vars(&[1.0]);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_logarithm_inverse() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test exp(ln(x)) = x for positive x
        let ln_x = ASTRepr::Ln(Box::new(x.clone()));
        let exp_ln_x = ASTRepr::Exp(Box::new(ln_x));

        let test_values = [1.0, 2.0, 5.0, 10.0, 100.0];
        for &val in &test_values {
            let result = exp_ln_x.eval_with_vars(&[val]);
            assert!(
                (result - val).abs() < 1e-10,
                "exp(ln({val})) = {result} != {val}"
            );
        }

        // Test ln(exp(x)) = x
        let exp_x = ASTRepr::Exp(Box::new(x.clone()));
        let ln_exp_x = ASTRepr::Ln(Box::new(exp_x));

        let test_values = [0.0, 1.0, -1.0, 2.5, -3.0];
        for &val in &test_values {
            let result = ln_exp_x.eval_with_vars(&[val]);
            assert!(
                (result - val).abs() < 1e-10,
                "ln(exp({val})) = {result} != {val}"
            );
        }
    }

    #[test]
    fn test_power_special_cases() {
        let x = ASTRepr::<f64>::Variable(0);

        // Test x^0 = 1
        let x_pow_0 = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::<f64>::Constant(0.0)));
        assert_eq!(x_pow_0.eval_with_vars(&[5.0]), 1.0);
        assert_eq!(x_pow_0.eval_with_vars(&[0.0]), 1.0);

        // Test x^1 = x
        let x_pow_1 = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::<f64>::Constant(1.0)));
        assert_eq!(x_pow_1.eval_with_vars(&[7.0]), 7.0);

        // Test 0^x = 0 for positive x
        let zero_pow_x = ASTRepr::Pow(Box::new(ASTRepr::<f64>::Constant(0.0)), Box::new(x.clone()));
        assert_eq!(zero_pow_x.eval_with_vars(&[2.0]), 0.0);

        // Test 1^x = 1
        let one_pow_x = ASTRepr::Pow(Box::new(ASTRepr::<f64>::Constant(1.0)), Box::new(x.clone()));
        assert_eq!(one_pow_x.eval_with_vars(&[100.0]), 1.0);
    }

    #[test]
    fn test_bound_var_evaluation() {
        // Test BoundVar evaluation
        let bound_var = ASTRepr::<f64>::BoundVar(0);
        assert_eq!(bound_var.eval_with_vars(&[42.0]), 42.0);

        let bound_var_1 = ASTRepr::<f64>::BoundVar(1);
        assert_eq!(bound_var_1.eval_with_vars(&[10.0, 20.0]), 20.0);
    }

    #[test]
    #[should_panic(expected = "BoundVar index 2 is out of bounds")]
    fn test_bound_var_out_of_bounds() {
        let bound_var = ASTRepr::<f64>::BoundVar(2);
        bound_var.eval_with_vars(&[1.0, 2.0]); // Only 2 variables, index 2 is out of bounds
    }

    #[test]
    fn test_let_expression_evaluation() {
        // Test Let expression (simplified evaluation)
        let x = ASTRepr::<f64>::Variable(0);
        let const_5 = ASTRepr::<f64>::Constant(5.0);
        let let_expr = ASTRepr::Let(0, Box::new(const_5), Box::new(x.clone()));

        // Current implementation just evaluates the body with existing variables
        let result = let_expr.eval_with_vars(&[10.0]);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_lambda_evaluation_edge_cases() {
        use crate::ast::ast_repr::Lambda;

        // Test lambda with no variables (constant lambda)
        let const_body = ASTRepr::<f64>::Constant(42.0);
        let const_lambda = Lambda {
            var_indices: vec![],
            body: Box::new(const_body),
        };
        let lambda_expr = ASTRepr::Lambda(Box::new(const_lambda));

        assert_eq!(lambda_expr.eval_with_vars(&[]), 42.0);
    }

    #[test]
    #[should_panic(expected = "Cannot evaluate lambda without function application")]
    fn test_lambda_with_variables_panics() {
        use crate::ast::ast_repr::Lambda;

        // Test lambda with variables (should panic)
        let var_body = ASTRepr::<f64>::Variable(0);
        let var_lambda = Lambda {
            var_indices: vec![0],
            body: Box::new(var_body),
        };
        let lambda_expr = ASTRepr::Lambda(Box::new(var_lambda));

        lambda_expr.eval_with_vars(&[5.0]);
    }

    #[test]
    fn test_collection_evaluation_empty() {
        use crate::ast::ast_repr::Collection;

        // Test empty collection sum
        let empty_collection = Collection::<f64>::Empty;
        let sum_expr = ASTRepr::Sum(Box::new(empty_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 0.0);
    }

    #[test]
    fn test_collection_evaluation_singleton() {
        use crate::ast::ast_repr::Collection;

        // Test singleton collection sum
        let singleton_expr = ASTRepr::<f64>::Constant(5.0);
        let singleton_collection = Collection::Singleton(Box::new(singleton_expr));
        let sum_expr = ASTRepr::Sum(Box::new(singleton_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 5.0);
    }

    #[test]
    fn test_collection_evaluation_range() {
        use crate::ast::ast_repr::Collection;

        // Test range collection sum: Σ(i=1 to 3) i = 1 + 2 + 3 = 6
        let start = ASTRepr::<f64>::Constant(1.0);
        let end = ASTRepr::<f64>::Constant(3.0);
        let range_collection = Collection::Range {
            start: Box::new(start),
            end: Box::new(end),
        };
        let sum_expr = ASTRepr::Sum(Box::new(range_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 6.0);
    }

    #[test]
    fn test_collection_evaluation_variable() {
        use crate::ast::ast_repr::Collection;

        // Test variable collection (placeholder implementation returns 0)
        let var_collection = Collection::<f64>::Variable(0);
        let sum_expr = ASTRepr::Sum(Box::new(var_collection));

        assert_eq!(sum_expr.eval_with_vars(&[]), 0.0);
    }

    #[test]
    fn test_eval_with_data_basic() {
        use crate::ast::multiset::MultiSet;
        // Test basic evaluation with data arrays
        let x = ASTRepr::<f64>::Variable(0);
        let result = x.eval_with_data(&[5.0], &[]);
        assert_eq!(result, 5.0);

        // Test with multiple parameters
        let expr = ASTRepr::Add(MultiSet::from_iter([
            ASTRepr::<f64>::Variable(0),
            ASTRepr::<f64>::Variable(1),
        ]));
        let result = expr.eval_with_data(&[3.0, 7.0], &[]);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_edge_case_evaluations() {
        // Test very large numbers
        let large_expr = ASTRepr::<f64>::Constant(1e100);
        assert_eq!(large_expr.eval_no_vars(), 1e100);

        // Test very small numbers
        let small_expr = ASTRepr::<f64>::Constant(1e-100);
        assert_eq!(small_expr.eval_no_vars(), 1e-100);

        // Test infinity handling
        let inf_expr = ASTRepr::<f64>::Constant(f64::INFINITY);
        assert!(inf_expr.eval_no_vars().is_infinite());

        // Test NaN handling
        let nan_expr = ASTRepr::<f64>::Constant(f64::NAN);
        assert!(nan_expr.eval_no_vars().is_nan());
    }

    #[test]
    fn test_eval_two_vars_comprehensive() {
        use crate::ast::multiset::MultiSet;
        // Test all basic operations with two variables
        let x = ASTRepr::<f64>::Variable(0);
        let y = ASTRepr::<f64>::Variable(1);

        // Addition
        let add_expr = ASTRepr::Add(MultiSet::from_iter([x.clone(), y.clone()]));
        assert_eq!(add_expr.eval_two_vars(3.0, 4.0), 7.0);

        // Subtraction
        let sub_expr = ASTRepr::Sub(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(sub_expr.eval_two_vars(10.0, 3.0), 7.0);

        // Multiplication
        let mul_expr = ASTRepr::Mul(MultiSet::from_iter([x.clone(), y.clone()]));
        assert_eq!(mul_expr.eval_two_vars(6.0, 7.0), 42.0);

        // Division
        let div_expr = ASTRepr::Div(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(div_expr.eval_two_vars(15.0, 3.0), 5.0);

        // Power
        let pow_expr = ASTRepr::Pow(Box::new(x.clone()), Box::new(y.clone()));
        assert_eq!(pow_expr.eval_two_vars(2.0, 3.0), 8.0);
    }
}
