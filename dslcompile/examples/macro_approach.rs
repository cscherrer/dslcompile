//! Macro-Based Zero-Overhead Heterogeneous Expressions
//!
//! This demonstrates how macros could solve the flexible arity problem
//! without any runtime storage overhead.

/// Macro to build zero-overhead expressions with flexible arity
macro_rules! hetero_expr {
    // Simple binary operations
    (|$x:ident: f64, $y:ident: f64| $x_ref:ident + $y_ref:ident) => {
        |$x: f64, $y: f64| -> f64 { $x_ref + $y_ref }
    };
    
    // Ternary operations
    (|$x:ident: f64, $y:ident: f64, $z:ident: f64| $x_ref:ident + $y_ref:ident + $z_ref:ident) => {
        |$x: f64, $y: f64, $z: f64| -> f64 { $x_ref + $y_ref + $z_ref }
    };
    
    // Mixed types - array indexing
    (|$arr:ident: &[f64], $idx:ident: usize, $bias:ident: f64| $arr_ref:ident[$idx_ref:ident] + $bias_ref:ident) => {
        |$arr: &[f64], $idx: usize, $bias: f64| -> f64 { $arr_ref[$idx_ref] + $bias_ref }
    };
}

/// More advanced macro for complex expressions
macro_rules! build_expr {
    // Entry point - capture parameters and expression
    (($($param:ident: $type:ty),*) => $body:tt) => {
        build_expr!(@expand ($($param: $type),*) $body)
    };
    
    // Expand addition
    (@expand ($($param:ident: $type:ty),*) { $left:ident + $right:ident }) => {
        |$($param: $type),*| -> f64 { $left + $right }
    };
    
    // Expand multiplication  
    (@expand ($($param:ident: $type:ty),*) { $left:ident * $right:ident }) => {
        |$($param: $type),*| -> f64 { $left * $right }
    };
    
    // Expand array access
    (@expand ($($param:ident: $type:ty),*) { $arr:ident[$idx:ident] }) => {
        |$($param: $type),*| -> f64 { $arr[$idx] as f64 }
    };
}

fn main() {
    println!("🚀 Macro-Based Zero-Overhead Expressions\n");
    
    // Example 1: Simple addition
    let add_expr = hetero_expr!(|x: f64, y: f64| x + y);
    let result1 = add_expr(3.0, 4.0);
    println!("✅ Simple addition: 3 + 4 = {}", result1);
    
    // Example 2: Three parameters
    let triple_add = hetero_expr!(|x: f64, y: f64, z: f64| x + y + z);
    let result2 = triple_add(1.0, 2.0, 3.0);
    println!("✅ Triple addition: 1 + 2 + 3 = {}", result2);
    
    // Example 3: Mixed types - array indexing
    let weights = [0.1, 0.2, 0.3, 0.4];
    let neural_expr = hetero_expr!(|arr: &[f64], idx: usize, bias: f64| arr[idx] + bias);
    let result3 = neural_expr(&weights, 1, 0.5);
    println!("✅ Neural network: weights[1] + bias = 0.2 + 0.5 = {}", result3);
    
    // Example 4: Advanced macro
    let complex_expr = build_expr!((x: f64, y: f64) => { x * y });
    let result4 = complex_expr(6.0, 7.0);
    println!("✅ Complex expression: 6 * 7 = {}", result4);
    
    println!("\n🎯 All expressions compile to direct function calls!");
    println!("🎯 Zero runtime overhead - no storage allocation!");
    println!("🎯 Flexible arity without heterogeneous containers!");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_macro_expressions() {
        // Test that macros generate correct functions
        let add = hetero_expr!(|x: f64, y: f64| x + y);
        assert_eq!(add(3.0, 4.0), 7.0);
        
        let triple = hetero_expr!(|x: f64, y: f64, z: f64| x + y + z);
        assert_eq!(triple(1.0, 2.0, 3.0), 6.0);
        
        let weights = [10.0, 20.0, 30.0];
        let neural = hetero_expr!(|arr: &[f64], idx: usize, bias: f64| arr[idx] + bias);
        assert_eq!(neural(&weights, 1, 5.0), 25.0);
    }
    
    #[test]
    fn test_advanced_macro() {
        let mult = build_expr!((a: f64, b: f64) => { a * b });
        assert_eq!(mult(4.0, 5.0), 20.0);
    }
} 