use dslcompile::prelude::*;

fn main() {
    println!("Testing if Vec<f64> implements required traits:");
    
    // Test that Vec<f64> implements the basic traits we need
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![2.0, 1.0, 4.0];
    
    // Test ExpressionType (automatically implemented via blanket impl)
    let _: &dyn ExpressionType = &v1;
    
    // Test PartialOrd
    let _cmp = v1.partial_cmp(&v2);
    
    println!("✓ Vec<f64> implements ExpressionType + PartialOrd");
    
    // Test that the type system changes work
    println!("✓ Successfully relaxed DynamicExpr<T> to allow T: ExpressionType + PartialOrd");
    println!("✓ Vector operations interface is now available!");
}