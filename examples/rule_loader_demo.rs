use mathcompile::final_tagless::{ASTRepr, ExpressionBuilder};
use mathcompile::symbolic::rule_loader::{RuleCategory, RuleConfig, RuleLoader};

#[cfg(feature = "optimization")]
use mathcompile::symbolic::egglog_integration::EgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Rule Loader System Demo");
    println!("==========================");

    // Demonstrate rule loading
    println!("\n📁 Available Rule Categories:");
    println!("-----------------------------");

    for category in RuleCategory::all() {
        println!("• {}: {}", category.filename(), category.description());
    }

    // Create a rule loader with default configuration
    let rule_loader = RuleLoader::default();

    println!("\n📋 Rule File Status:");
    println!("--------------------");

    match rule_loader.list_available_rules() {
        Ok(rules_info) => {
            for (category, exists, description) in rules_info {
                let status = if exists { "✅ Found" } else { "❌ Missing" };
                println!("{} {}: {}", status, category.filename(), description);
            }
        }
        Err(e) => {
            println!("Error checking rule files: {e}");
        }
    }

    // Try to load rules
    println!("\n🔄 Loading Rules:");
    println!("-----------------");

    match rule_loader.load_rules() {
        Ok(program) => {
            println!(
                "✅ Successfully loaded {} characters of egglog rules",
                program.len()
            );

            // Show a snippet of the loaded program
            let lines: Vec<&str> = program.lines().take(10).collect();
            println!("\n📄 Rule Program Preview:");
            println!("------------------------");
            for line in lines {
                if !line.trim().is_empty() {
                    println!("{line}");
                }
            }
            if program.lines().count() > 10 {
                println!("... ({} more lines)", program.lines().count() - 10);
            }
        }
        Err(e) => {
            println!("❌ Failed to load rules: {e}");
            println!("\n💡 This is expected if rule files haven't been created yet.");
            println!("   The rule files should be in the 'rules/' directory:");
            for category in RuleCategory::default_set() {
                println!("   - rules/{}", category.filename());
            }
            return Ok(());
        }
    }

    // Test with egglog optimizer if optimization feature is enabled
    #[cfg(feature = "optimization")]
    {
        println!("\n🧮 Testing Egglog Integration with Rule Loader:");
        println!("-----------------------------------------------");

        // Test different optimizer configurations
        let optimizer_configs: Vec<(&str, Box<dyn Fn() -> Result<EgglogOptimizer, _>>)> = vec![
            ("Default", Box::new(EgglogOptimizer::new)),
            ("Domain-Aware", Box::new(EgglogOptimizer::domain_aware)),
            ("Canonical Only", Box::new(EgglogOptimizer::canonical_only)),
        ];

        for (name, create_optimizer) in optimizer_configs {
            println!("\n🔧 Testing {name} Configuration:");

            match create_optimizer() {
                Ok(mut optimizer) => {
                    println!("✅ Successfully created {name} EgglogOptimizer");

                    // Show rule information
                    match optimizer.rule_info() {
                        Ok(rule_info) => {
                            println!("📊 Loaded rule categories:");
                            for (category, exists, description) in rule_info {
                                let status = if exists { "✅" } else { "❌" };
                                println!("   {} {}: {}", status, category.filename(), description);
                            }
                        }
                        Err(e) => {
                            println!("⚠️  Could not get rule info: {e}");
                        }
                    }

                    // Test optimization with a simple expression
                    let mut builder = ExpressionBuilder::new();
                    let x = builder.var("x");
                    let expr = ASTRepr::Add(Box::new(x), Box::new(ASTRepr::Constant(0.0)));

                    println!("🔍 Testing optimization:");
                    println!("   Original: x + 0");

                    match optimizer.optimize(&expr) {
                        Ok(optimized) => {
                            println!("   Optimized: {optimized:?}");
                            println!("   ✅ Optimization successful!");
                        }
                        Err(e) => {
                            println!("   ❌ Optimization failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Failed to create {name} EgglogOptimizer: {e}");
                }
            }
        }

        // Test custom rule configuration
        println!("\n⚙️  Custom Rule Configuration:");
        println!("------------------------------");

        let custom_config = RuleConfig {
            categories: vec![
                RuleCategory::CoreDatatypes,
                RuleCategory::BasicArithmetic,
                RuleCategory::Trigonometric,
            ],
            validate_syntax: true,
            include_comments: true,
            ..Default::default()
        };

        match EgglogOptimizer::with_rule_config(custom_config) {
            Ok(optimizer) => {
                println!("✅ Custom configuration optimizer created successfully");
                match optimizer.rule_info() {
                    Ok(rule_info) => {
                        println!("📊 Custom rule categories loaded: {}", rule_info.len());
                    }
                    Err(e) => {
                        println!("⚠️  Could not get custom rule info: {e}");
                    }
                }
            }
            Err(e) => {
                println!("❌ Custom configuration failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n💡 Egglog integration testing skipped (optimization feature not enabled)");
        println!(
            "   To test with egglog, run with: cargo run --example rule_loader_demo --features optimization"
        );
    }

    println!("\n🎯 Summary:");
    println!("-----------");
    println!("• Rule files are organized by mathematical domain");
    println!("• Rules can be loaded selectively based on needs");
    println!("• Syntax validation ensures rule correctness");
    println!("• Integration with egglog optimizer is seamless");
    println!("• Custom configurations allow fine-tuned control");
    println!("• Multiple optimizer configurations available:");
    println!("  - Default: Core + Basic + Transcendental rules");
    println!("  - Domain-Aware: Safe rules with domain constraints");
    println!("  - Canonical Only: Simplified rule set for basic optimization");

    Ok(())
}
