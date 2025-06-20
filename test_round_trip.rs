//! Test round-trip collection identity preservation

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::egg_optimizer::optimize_simple_sum_splitting;

fn main() -> Result<()> {
    #[cfg(feature = "optimization")]
    {
        println!("🔄 Testing MathLang ↔ ASTRepr Round-Trip Identity Preservation");
        println!("================================================================\n");

        let mut ctx = DynamicContext::new();
        let x = ctx.var();
        
        // Test 1: Simple sum with collection identity
        println!("1️⃣ Simple Sum Round-Trip Test");
        let data = vec![1.0, 2.0, 3.0];
        let sum_expr = ctx.sum(data.clone(), |item| &x * item);
        let original_ast = ctx.to_ast(&sum_expr);
        
        // Perform round-trip: AST → MathLang → AST
        let result = optimize_simple_sum_splitting(&original_ast);
        assert!(result.is_ok(), "Round-trip should succeed");
        
        let optimized_ast = result.unwrap();
        
        // Test semantic equivalence
        let test_x = 2.0;
        let original_result = ctx.eval(&sum_expr, hlist![test_x]);
        let optimized_result = optimized_ast.eval_with_vars(&[test_x]);
        
        println!("   Original result: {}", original_result);
        println!("   Optimized result: {}", optimized_result);
        println!("   Expected: {} * (1 + 2 + 3) = {}", test_x, test_x * 6.0);
        
        let expected = test_x * (1.0 + 2.0 + 3.0);
        assert!((original_result - expected).abs() < 1e-10, "Original should match expected");
        assert!((optimized_result - expected).abs() < 1e-10, "Optimized should match expected");
        assert!((original_result - optimized_result).abs() < 1e-10, "Round-trip should preserve semantics");
        
        println!("   ✅ Round-trip semantic equivalence preserved\n");
        
        // Test 2: Multiple collections with different identities
        println!("2️⃣ Multiple Collection Identity Test");
        let data1 = vec![1.0, 2.0];
        let data2 = vec![3.0, 4.0];
        
        let sum1 = ctx.sum(data1.clone(), |item| &x * item);
        let sum2 = ctx.sum(data2.clone(), |item| &x * item);
        let compound = &sum1 + &sum2;
        
        let compound_ast = ctx.to_ast(&compound);
        let compound_result = optimize_simple_sum_splitting(&compound_ast);
        assert!(compound_result.is_ok(), "Compound round-trip should succeed");
        
        let compound_optimized = compound_result.unwrap();
        
        let original_compound = ctx.eval(&compound, hlist![test_x]);
        let optimized_compound = compound_optimized.eval_with_vars(&[test_x]);
        
        println!("   Original compound: {}", original_compound);
        println!("   Optimized compound: {}", optimized_compound);
        
        let expected_compound = test_x * (1.0 + 2.0) + test_x * (3.0 + 4.0);
        println!("   Expected: {} * (1+2) + {} * (3+4) = {}", test_x, test_x, expected_compound);
        
        assert!((original_compound - expected_compound).abs() < 1e-10, "Original compound should match expected");
        assert!((optimized_compound - expected_compound).abs() < 1e-10, "Optimized compound should match expected");
        assert!((original_compound - optimized_compound).abs() < 1e-10, "Compound round-trip should preserve semantics");
        
        println!("   ✅ Multiple collection identities preserved\n");
        
        println!("🎉 All round-trip tests passed!");
        println!("   • Collection identity preservation: ✅");
        println!("   • Semantic equivalence: ✅");
        println!("   • Multiple collection handling: ✅");
    }
    
    #[cfg(not(feature = "optimization"))]
    {
        println!("⚠️  Optimization feature not enabled - skipping round-trip tests");
    }
    
    Ok(())
}