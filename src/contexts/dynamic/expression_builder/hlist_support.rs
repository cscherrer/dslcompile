pub trait HListEval<T: Scalar> {
    /// Evaluate AST with zero-cost HList storage
    fn eval_expr(&self, ast: &ASTRepr<T>) -> T;

    /// Get variable value by index with zero runtime dispatch
    fn get_var(&self, index: usize) -> T;

    /// Apply a lambda function to arguments from this HList
    fn apply_lambda(&self, lambda: &crate::ast::ast_repr::Lambda<T>, args: &[T]) -> T;

    /// Convert HList to variable vector for lambda evaluation
    fn to_variable_vec(&self) -> Vec<T>;

    /// Convert HList to variable vector for lambda evaluation
    fn to_variable_vec(&self) -> Vec<T>;
}

/// Helper function to evaluate AST with variable context (Vec-based)
fn eval_with_variable_context<T>(ast: &ASTRepr<T>, variables: &[T]) -> T
where
    T: Scalar + Copy + num_traits::Float + num_traits::FromPrimitive,
{
    match ast {
        ASTRepr::Constant(value) => *value,
        ASTRepr::Variable(index) => {
            if *index < variables.len() {
                variables[*index]
            } else {
                panic!("Variable index {} out of bounds", index)
            }
        }
        ASTRepr::BoundVar(index) => {
            if *index < variables.len() {
                variables[*index]
            } else {
                panic!("BoundVar index {} out of bounds", index)
            }
        }
        ASTRepr::Add(left, right) => {
            eval_with_variable_context(left, variables) + eval_with_variable_context(right, variables)
        }
        ASTRepr::Sub(left, right) => {
            eval_with_variable_context(left, variables) - eval_with_variable_context(right, variables)
        }
        ASTRepr::Mul(left, right) => {
            eval_with_variable_context(left, variables) * eval_with_variable_context(right, variables)
        }
        ASTRepr::Div(left, right) => {
            eval_with_variable_context(left, variables) / eval_with_variable_context(right, variables)
        }
        ASTRepr::Pow(base, exp) => {
            eval_with_variable_context(base, variables).powf(eval_with_variable_context(exp, variables))
        }
        ASTRepr::Neg(inner) => -eval_with_variable_context(inner, variables),
        ASTRepr::Ln(inner) => eval_with_variable_context(inner, variables).ln(),
        ASTRepr::Exp(inner) => eval_with_variable_context(inner, variables).exp(),
        ASTRepr::Sin(inner) => eval_with_variable_context(inner, variables).sin(),
        ASTRepr::Cos(inner) => eval_with_variable_context(inner, variables).cos(),
        ASTRepr::Sqrt(inner) => eval_with_variable_context(inner, variables).sqrt(),
        ASTRepr::Let(var_index, expr, body) => {
            let bound_value = eval_with_variable_context(expr, variables);
            let mut extended_vars = variables.to_vec();
            if *var_index >= extended_vars.len() {
                extended_vars.resize(*var_index + 1, T::zero());
            }
            extended_vars[*var_index] = bound_value;
            eval_with_variable_context(body, &extended_vars)
        }
        _ => panic!("Unsupported AST node in eval_with_variable_context"),
    }
} 