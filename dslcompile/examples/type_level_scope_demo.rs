use dslcompile::contexts::dynamic::expression_builder::{DynamicContext, TypedBuilderExpr};
use frunk::hlist;

fn main() {
    println!("=== Type-Level Scope Safety Demo ===");
    
    // Same scope - operations allowed
    println!("\n1. Same scope operations (✓ compiles):");
    let mut ctx = DynamicContext::<f64, 0>::new();
    let x = ctx.var();
    let y = ctx.var();
    let expr = &x + &y;  // ✓ Compiles - same scope
    let result = ctx.eval(&expr, hlist![3.0, 4.0]);
    println!("   x + y = {} (with x=3.0, y=4.0)", result);
    
    // Different scopes - prevented at compile time
    println!("\n2. Different scope operations (❌ compile error):");
    println!("   This code would NOT compile:");
    println!("   ```");
    println!("   let mut ctx1 = DynamicContext::<f64, 0>::new();");
    println!("   let mut ctx2 = DynamicContext::<f64, 1>::new_explicit();");
    println!("   let x1 = ctx1.var();");
    println!("   let x2 = ctx2.var();");
    println!("   let bad = &x1 + &x2;  // ❌ Compile error!");
    println!("   ```");
    println!("   Error: mismatched types - expected SCOPE `0`, found SCOPE `1`");
    
    // Explicit scope advancement for composition
    println!("\n3. Explicit scope advancement (✓ safe composition):");
    let mut ctx1 = DynamicContext::<f64, 0>::new();
    let x1 = ctx1.var();
    let expr1 = &x1 * 2.0;
    
    let ctx_next = ctx1.next();  // DynamicContext<f64, 1>
    let mut ctx2 = DynamicContext::<f64, 1>::new_explicit();
    let x2: TypedBuilderExpr<f64, 1> = ctx2.var();
    let three = ctx2.constant(3.0);
    let expr2 = x2.clone() * three;
    
    // Now we can merge contexts of the same scope
    let combined_ctx = ctx_next.merge(ctx2);
    println!("   Successfully created combined context with explicit scope management");
    
    println!("\n✅ Type-level scopes prevent variable collisions at COMPILE TIME!");
    println!("   No more runtime 'Variable index out of bounds' errors!");
    println!("   No more non-deterministic scope merging based on memory addresses!");
} 