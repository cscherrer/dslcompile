//! Debug Sum Splitting - Understanding What's Actually Happening

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<()> {
    println!("🔍 DEBUGGING SUM SPLITTING OPTIMIZATION");
    println!("========================================\n");

    // Create a simple test case: Σ(a*x + b*x)
    let mut ctx = DynamicContext::new();
    let a = ctx.var::<f64>(); // Variable(0)
    let b = ctx.var::<f64>(); // Variable(1)
    let data = vec![1.0, 2.0, 3.0];

    let test_expr = ctx.sum(&data, |x| &a * &x + &b * &x);

    println!("📊 TEST EXPRESSION: Σ(a*x + b*x) over [1, 2, 3]");
    println!("Expected: (a+b) * Σ(x) = (a+b) * 6");
    println!("With a=2, b=3: (2+3) * 6 = 30\n");

    // Show original evaluation
    let original_result = ctx.eval(&test_expr, hlist![2.0, 3.0]);
    println!("✅ Original evaluation: {}", original_result);

    // Convert to AST and show structure
    let original_ast = ctx.to_ast(&test_expr);
    println!("\n🏗️  ORIGINAL AST STRUCTURE:");
    println!("{:#?}", original_ast);

    #[cfg(feature = "optimization")]
    {
        // Set up optimizer
        use dslcompile::symbolic::rule_loader::{RuleCategory, RuleConfig, RuleLoader};
        let config = RuleConfig {
            categories: vec![RuleCategory::CoreDatatypes, RuleCategory::Summation],
            validate_syntax: true,
            include_comments: true,
            ..Default::default()
        };

        let rule_loader = RuleLoader::new(config);
        let mut optimizer = NativeEgglogOptimizer::with_rule_loader(rule_loader)?;

        println!("\n🧮 APPLYING OPTIMIZATION...");
        match optimizer.optimize(&original_ast) {
            Ok(optimized_ast) => {
                println!("✅ Optimization completed");

                println!("\n🏗️  OPTIMIZED AST STRUCTURE:");
                println!("{:#?}", optimized_ast);

                // Test evaluation
                let optimized_result = optimized_ast.eval_with_vars(&[2.0, 3.0]);
                println!("\n📊 EVALUATION COMPARISON:");
                println!("Original:  {}", original_result);
                println!("Optimized: {}", optimized_result);
                println!(
                    "Match: {}",
                    (original_result - optimized_result).abs() < 1e-10
                );

                // Show string representations to see any differences
                let orig_str = format!("{:?}", original_ast);
                let opt_str = format!("{:?}", optimized_ast);

                println!("\n🔍 STRING COMPARISON:");
                println!("Original length:  {} chars", orig_str.len());
                println!("Optimized length: {} chars", opt_str.len());

                if orig_str == opt_str {
                    println!(
                        "❓ Strings are identical - extraction may not be returning optimized form"
                    );
                } else {
                    println!("✅ Strings differ - optimization visible!");
                    println!("\nDifferences:");
                    if opt_str.contains("Add") && opt_str.contains("Sum") {
                        println!("  - Contains sum splitting pattern (Add of Sums)");
                    }
                    if opt_str.len() < orig_str.len() {
                        println!("  - Optimized expression is shorter");
                    }
                }

                // Try to understand the internal AST structure
                println!("\n🔬 DETAILED AST ANALYSIS:");
                analyze_ast_structure("Original", &original_ast);
                analyze_ast_structure("Optimized", &optimized_ast);
            }
            Err(e) => {
                println!("❌ Optimization failed: {}", e);
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n⚠️  Optimization features not enabled");
        println!("Run with: cargo run --features optimization --example debug_sum_splitting");
    }

    Ok(())
}

#[cfg(feature = "optimization")]
fn analyze_ast_structure(label: &str, ast: &dslcompile::ast::ASTRepr<f64>) {
    use dslcompile::ast::{ASTRepr, ast_repr::Collection};

    println!("  {} structure:", label);
    match ast {
        ASTRepr::Sum(collection) => {
            println!("    - Top level: Sum");
            match &**collection {
                Collection::Map { lambda, collection } => {
                    println!(
                        "    - Contains: Map over {} items",
                        match &**collection {
                            Collection::DataArray(data) => data.len(),
                            _ => 0,
                        }
                    );
                    println!("    - Lambda body: {:?}", lambda.body);

                    // Check if lambda body shows optimization patterns
                    match lambda.body.as_ref() {
                        ASTRepr::Add(terms) => {
                            println!("    - Body is Add with {} terms", terms.len());
                            for (i, term) in terms.iter().enumerate() {
                                match term {
                                    ASTRepr::Sum(_) => {
                                        println!("      Term {}: Sum (factored out!)", i)
                                    }
                                    ASTRepr::Mul(factors) => println!(
                                        "      Term {}: Mul with {} factors",
                                        i,
                                        factors.len()
                                    ),
                                    _ => println!("      Term {}: {:?}", i, term),
                                }
                            }
                        }
                        ASTRepr::Mul(factors) => {
                            println!("    - Body is Mul with {} factors", factors.len());
                            for (i, factor) in factors.iter().enumerate() {
                                match factor {
                                    ASTRepr::Sum(_) => {
                                        println!("      Factor {}: Sum (coefficient factored!)", i)
                                    }
                                    ASTRepr::Variable(idx) => {
                                        println!("      Factor {}: Variable({})", i, idx)
                                    }
                                    _ => println!("      Factor {}: {:?}", i, factor),
                                }
                            }
                        }
                        _ => println!("    - Body: {:?}", lambda.body),
                    }
                }
                _ => println!("    - Unexpected collection structure: {:?}", collection),
            }
        }
        ASTRepr::Add(terms) => {
            println!("    - Top level: Add with {} terms", terms.len());
            for (i, term) in terms.iter().enumerate() {
                match term {
                    ASTRepr::Sum(_) => println!("      Term {}: Sum (split successfully!)", i),
                    _ => println!("      Term {}: {:?}", i, term),
                }
            }
        }
        ASTRepr::Mul(factors) => {
            println!("    - Top level: Mul with {} factors", factors.len());
            for (i, factor) in factors.iter().enumerate() {
                match factor {
                    ASTRepr::Sum(_) => println!("      Factor {}: Sum (coefficient factored!)", i),
                    ASTRepr::Add(_) => println!("      Factor {}: Add (coefficient is sum!)", i),
                    _ => println!("      Factor {}: {:?}", i, factor),
                }
            }
        }
        _ => println!("    - Structure: {:?}", ast),
    }
}

#[cfg(not(feature = "optimization"))]
fn analyze_ast_structure(_label: &str, _ast: &dslcompile::ast::ASTRepr<f64>) {
    // No-op when optimization is disabled
}
