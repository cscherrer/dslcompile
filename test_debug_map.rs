//! Simple test to debug the map operation issue

#[cfg(test)]
mod tests {
    use dslcompile::prelude::*;
    use frunk::hlist;

    #[test] 
    fn debug_map_operation() {
        println!("🔍 Debug: Testing map operation with embedded data");
        
        let mut ctx = DynamicContext::new();
        let data = vec![1.0, 2.0, 3.0];
        
        // Create embedded data expression
        let data_expr = ctx.data_array(data.clone());
        println!("Data expr AST: {:?}", ctx.to_ast(&data_expr));
        
        // Create the map expression
        let mapped_expr = data_expr.map(|x| &x * &x);
        println!("Mapped expr AST: {:?}", ctx.to_ast(&mapped_expr));
        
        // Try to evaluate with no parameters (should work for embedded data)
        println!("Attempting to evaluate with hlist![]...");
        let result = ctx.eval(&mapped_expr, hlist![]);
        println!("Result: {:?}", result);
    }

    #[test]
    fn debug_parameterized_data() {
        println!("🔍 Debug: Testing parameterized data evaluation");
        
        let mut ctx = DynamicContext::new();
        let data_param = ctx.var::<Vec<f64>>();
        
        println!("Data param AST: {:?}", ctx.to_ast(&data_param));
        
        let mapped_expr = data_param.map(|x| &x * &x);
        println!("Mapped param expr AST: {:?}", ctx.to_ast(&mapped_expr));
        
        let test_data = vec![1.0, 2.0, 3.0];
        let result = ctx.eval(&mapped_expr, hlist![test_data]);
        println!("Parameterized result: {:?}", result);
    }
}