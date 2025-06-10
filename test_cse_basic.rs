use dslcompile::ast::DynamicContext;
use dslcompile::symbolic::native_egglog::optimize_with_native_egglog;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing CSE functionality");
    
    // Create a simple expression: x * x
    let mut ctx = DynamicContext::new();
    let x = ctx.var(); // Variable 0
    let expr = &x * &x;
    
    println!("📥 Input: x₀ * x₀");
    
    // Convert to AST
    let ast_expr = dslcompile::ast::advanced::ast_from_expr(&expr);
    println!("   AST: {:?}", ast_expr);
    
    // Try egglog optimization
    match optimize_with_native_egglog(&ast_expr) {
        Ok(optimized) => {
            println!("📤 Optimized: {:?}", optimized);
            
            // Check if CSE fired
            let optimized_str = format!("{:?}", optimized);
            if optimized_str.contains("Let") || optimized_str.contains("BoundVar") {
                println!("   ✅ CSE rules fired! Found Let/BoundVar in output.");
            } else {
                println!("   ❌ CSE rules did NOT fire. Expression unchanged.");
                println!("   This means our CSE rules are not being applied.");
            }
        }
        Err(e) => {
            println!("   ⚠️ Optimization failed: {}", e);
        }
    }
    
    Ok(())
} 