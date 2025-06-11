pub fn var<U: Scalar>(&mut self) -> TypedBuilderExpr<U> {
    // Register the variable in the registry (gets automatic index)
    let var_id = {
        let mut registry = self.registry.borrow_mut();
        registry.register_variable()
    };

    // 🔧 CRITICAL FIX: Keep next_var_id synchronized with registry
    // The lambda creation uses next_var_id, so it must be in sync
    self.next_var_id = self.next_var_id.max(var_id + 1);

    TypedBuilderExpr::new(ASTRepr::Variable(var_id), self.registry.clone())
}

let iter_var_index = ctx.next_var_id;
ctx.next_var_id += 1;

eprintln!("🔧 DEBUG Vec<f64> lambda creation:");
eprintln!("  ctx.next_var_id was: {}, now: {}", iter_var_index, ctx.next_var_id);
eprintln!("  Registry size: {}", ctx.registry.borrow().len());

// Create iterator variable for the lambda
let iter_var =
    TypedBuilderExpr::new(ASTRepr::Variable(iter_var_index), ctx.registry.clone());

// Apply the user's function to get the lambda body
let body_expr = f(iter_var);

eprintln!("  Lambda body AST: {:?}", body_expr.ast);

// Create lambda from the body expression
let lambda = Lambda::Lambda {
    var_index: iter_var_index,
    body: Box::new(body_expr.ast),
}; 