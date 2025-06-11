/// Generate lambda body code with variable substitution
fn generate_lambda_body_with_var<T: Scalar + Float + Copy + std::fmt::Display + 'static>(
    &self,
    body: &ASTRepr<T>,
    var_index: usize,
    var_name: &str,
    registry: &VariableRegistry,
) -> Result<String> {
    // For now, do simple variable name substitution
    match body {
        ASTRepr::Variable(index) if *index == var_index => {
            Ok(var_name.to_string())
        }
        ASTRepr::Variable(index) => {
            // Other variables - look up in registry
            Ok(registry.debug_name(*index))
        }
        ASTRepr::Constant(value) => Ok(format!("{value}")),
        _ => Ok(String::new()),
    }
} 