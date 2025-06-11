fn build_iid_gaussian_expression() -> (DynamicContext<f64>, TypedBuilderExpr<f64>, String) {
    let mut ctx = DynamicContext::new();

    // ✅ Create scalar parameters that will be bound at evaluation time
    let mu = ctx.var(); // Parameter 0: mean (symbolic)
    let sigma = ctx.var(); // Parameter 1: std deviation (symbolic)

    println!("   Created symbolic parameters: mu=var_0, sigma=var_1");

    // ✅ Create symbolic constants (these can be folded)
    let neg_half = ctx.constant(-0.5);
    let log_2pi = ctx.constant((2.0 * std::f64::consts::PI).ln());

    // ✅ SYMBOLIC data summation - placeholder vector creates DataArray collection
    // The actual data gets bound at evaluation time through the unified API
    let placeholder_data = vec![0.0]; // Creates symbolic DataArray, not actual computation
    let gaussian_sum = ctx.sum(placeholder_data, |x| {
        println!("🔧 Inside lambda closure, x variable index: {:?}", x.ast);
        // ... existing code ...
    });

    // ... rest of the function ...
} 