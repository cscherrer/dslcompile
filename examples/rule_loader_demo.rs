//! Rule Loader Demo
//! Demonstrates the dynamic rule loading system for egglog optimization

use dslcompile::final_tagless::{ASTEval, ASTMathExpr};
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;
use dslcompile::symbolic::rule_loader::{RuleCategory, RuleConfig, RuleLoader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Rule Loader Demo");
    println!("==================");

    // Create a mathematical expression: ln(exp(x)) + 0
    let expr = ASTEval::add(
        ASTEval::ln(ASTEval::exp(ASTEval::var(0))),
        ASTEval::constant(0.0),
    );

    println!("\n📝 Original expression: ln(exp(x)) + 0");
    println!("Expected optimization: x (using ln(exp(x)) = x and a + 0 = a)");

    // Test 1: Default configuration with domain-aware optimizer
    println!("\n🧪 Test 1: Domain-Aware Optimizer (Default)");
    let mut optimizer = NativeEgglogOptimizer::new()?;
    let optimized = optimizer.optimize(&expr)?;
    println!("Result: {optimized:?}");

    // Test 2: Rule loader with basic arithmetic rules
    println!("\n🧪 Test 2: Rule Loader System");
    let basic_config = RuleConfig {
        categories: vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::BasicArithmetic,
            RuleCategory::Transcendental,
        ],
        ..Default::default()
    };

    let rule_loader = RuleLoader::new(basic_config);
    println!(
        "Loaded rule categories: {:?}",
        rule_loader.list_available_rules()?
    );

    // Test 3: Domain-aware rule configuration
    println!("\n🧪 Test 3: Domain-Aware Rule Configuration");
    let domain_aware_config = RuleConfig::domain_aware();
    let domain_loader = RuleLoader::new(domain_aware_config);
    println!(
        "Domain-aware categories: {:?}",
        domain_loader.list_available_rules()?
    );

    // Test 4: Demonstrate rule loading
    println!("\n📋 Rule Loading Test:");
    match rule_loader.load_rules() {
        Ok(program) => {
            println!(
                "✅ Successfully loaded {} characters of egglog rules",
                program.len()
            );

            // Show a snippet
            let lines: Vec<&str> = program.lines().take(5).collect();
            println!("Preview:");
            for line in lines {
                if !line.trim().is_empty() {
                    println!("  {line}");
                }
            }
        }
        Err(e) => {
            println!("⚠️  Could not load rules: {e}");
            println!("💡 This is expected if rule files haven't been created yet.");
        }
    }

    println!("\n✅ Rule loader demo completed successfully!");
    println!("💡 The domain-aware optimizer provides mathematical safety");
    println!("   while the rule loader system offers flexible configuration.");

    Ok(())
}
