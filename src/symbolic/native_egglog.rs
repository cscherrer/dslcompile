/// TEST: Debug CSE rules to see why they're not firing
pub fn debug_cse_rules() -> Result<()> {
    println!("🔍 CSE Rule Debug Test");
    println!("=====================");
    
    use crate::ast::DynamicContext;
    use crate::ast::advanced::ast_from_expr;
    
    // Create the EXACT pattern our CSE rules target: x * x
    let mut ctx = DynamicContext::new();
    let x = ctx.var(); // Variable 0
    
    // This should match our CSE Rule 1: (Mul ?expr ?expr)
    let expr = &x * &x;
    
    println!("📥 Input Expression: x₀ * x₀");
    println!("   Expected match: CSE Rule 1 - (Mul ?expr ?expr)");
    println!("   Expected output: (Let 1000 x₀ (Mul (BoundVar 1000) (BoundVar 1000)))");
    
    // Convert to AST and optimize
    let ast_expr = ast_from_expr(&expr);
    println!("   AST form: {:?}", ast_expr);
    
    // Run egglog optimization
    match optimize_with_native_egglog(&ast_expr) {
        Ok(optimized) => {
            println!("📤 Optimized Expression: {:?}", optimized);
            
            // Check if CSE fired
            if format!("{:?}", optimized).contains("Let") {
                println!("   ✅ CSE rules fired! Let expression found.");
            } else {
                println!("   ❌ CSE rules did NOT fire. Expression unchanged.");
            }
        }
        Err(e) => {
            println!("   ⚠️ Optimization failed: {}", e);
        }
    }
    
    // Test a more complex case: the Gaussian pattern
    println!("\n🔍 Testing Gaussian Pattern");
    println!("===========================");
    
    let mu = ctx.var(); // Variable 1  
    let sigma = ctx.var(); // Variable 2
    let x2 = ctx.var(); // Variable 3 (different x)
    
    // The problematic pattern: ((x - mu) / sigma)²
    let standardized = (&x2 - &mu) / &sigma;
    let squared = &standardized * &standardized;
    
    println!("📥 Input: ((x - μ) / σ)²");
    println!("   This should trigger CSE Rule 5 or Rule 1");
    
    let ast_expr2 = ast_from_expr(&squared);
    println!("   AST form: {:?}", ast_expr2);
    
    match optimize_with_native_egglog(&ast_expr2) {
        Ok(optimized2) => {
            println!("📤 Optimized: {:?}", optimized2);
            
            if format!("{:?}", optimized2).contains("Let") {
                println!("   ✅ CSE rules fired! Let expression found.");
            } else {
                println!("   ❌ CSE rules did NOT fire. Expression unchanged.");
            }
        }
        Err(e) => {
            println!("   ⚠️ Optimization failed: {}", e);
        }
    }
    
    Ok(())
} 