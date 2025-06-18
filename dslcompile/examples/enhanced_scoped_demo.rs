//! Enhanced Scoped Demo
//! 
//! This demo shows enhanced scoped functionality in DSLCompile.

use dslcompile::contexts::static_context::var;

fn main() {
    println!("Enhanced Scoped Demo");
    
    // Simple demonstration
    let x = var::<0>();
    let y = var::<1>();
    
    let expr = x.add(y);
    
    println!("Created expression: x + y");
    println!("Demo completed successfully");
} 