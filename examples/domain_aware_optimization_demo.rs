use mathcompile::final_tagless::{ASTRepr, ExpressionBuilder};
use mathcompile::interval_domain::{IntervalDomain, IntervalDomainAnalyzer};
use mathcompile::symbolic::rule_loader::{RuleConfig, RuleLoader};

#[cfg(feature = "optimization")]
use mathcompile::symbolic::egglog_integration::EgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Domain-Aware Mathematical Optimization Demo");
    println!("===============================================");

    println!("\n📐 The Problem: Mathematical vs Computational Truth");
    println!("---------------------------------------------------");

    println!("Mathematical truth: 0^0 is indeterminate");
    println!("IEEE 754 standard: 0^0 = 1 (computational convention)");
    println!("Our goal: Use domain analysis to apply rules safely");

    // Demonstrate domain analysis
    println!("\n🔍 Domain Analysis Examples:");
    println!("-----------------------------");

    let mut analyzer = IntervalDomainAnalyzer::new(0.0);

    // Case 1: Variable known to be positive
    analyzer.set_variable_domain(0, IntervalDomain::positive(0.0));
    let positive_domain = analyzer.get_variable_domain(0);
    println!("Variable x with domain {positive_domain}:");
    println!("  ✅ Safe to apply: x^0 = 1");
    println!("  ✅ Safe to apply: x/x = 1");
    println!("  ✅ Safe to apply: ln(x) is defined");

    // Case 2: Variable known to be non-negative
    analyzer.set_variable_domain(1, IntervalDomain::non_negative(0.0));
    let non_negative_domain = analyzer.get_variable_domain(1);
    println!("\nVariable y with domain {non_negative_domain}:");
    println!("  ⚠️  Unsafe: y^0 = 1 (fails when y=0)");
    println!("  ⚠️  Unsafe: y/y = 1 (fails when y=0)");
    println!("  ✅ Safe to apply: √(y²) = y");

    // Case 3: Variable with unknown domain
    let unknown_domain = analyzer.get_variable_domain(2);
    println!("\nVariable z with domain {unknown_domain}:");
    println!("  ❌ Unsafe: z^0 = 1 (unknown if z=0)");
    println!("  ❌ Unsafe: z/z = 1 (unknown if z=0)");
    println!("  ❌ Unsafe: √(z²) = z (fails when z<0)");

    println!("\n🧮 Domain-Aware Rule Generation:");
    println!("--------------------------------");

    // Create domain-aware rule configuration
    let domain_config = RuleConfig::domain_aware()
        .with_variable_domain("x", IntervalDomain::positive(0.0))
        .with_variable_domain("y", IntervalDomain::non_negative(0.0))
        .with_variable_domain("z", IntervalDomain::closed_interval(-1.0, 1.0));

    let domain_loader = RuleLoader::new(domain_config);

    match domain_loader.load_rules() {
        Ok(rules) => {
            println!("✅ Generated domain-aware rules:");

            // Show a preview of the generated rules
            let lines: Vec<&str> = rules.lines().collect();
            let mut in_domain_section = false;
            let mut shown_lines = 0;

            for line in lines {
                if line.contains("DYNAMICALLY GENERATED DOMAIN-AWARE RULES") {
                    in_domain_section = true;
                    continue;
                }

                if in_domain_section && !line.trim().is_empty() && shown_lines < 10 {
                    println!("   {line}");
                    shown_lines += 1;
                }
            }

            if shown_lines == 0 {
                println!("   (No domain-specific rules generated - variables not found in rules)");
            }
        }
        Err(e) => {
            println!("❌ Failed to generate rules: {e}");
        }
    }

    println!("\n🔬 Practical Examples:");
    println!("----------------------");

    // Example 1: Safe optimization with positive domain
    let mut builder = ExpressionBuilder::new();
    let x = builder.var("x");

    // x^0 where x > 0 - safe to optimize to 1
    let x_pow_0 = ASTRepr::Pow(Box::new(x.clone()), Box::new(ASTRepr::Constant(0.0)));
    println!("Expression: x^0 where x ∈ (0, +∞)");
    println!("  Original: {x_pow_0:?}");
    println!("  Safe optimization: 1.0");
    println!("  Reason: x > 0 guarantees x ≠ 0");

    // Example 2: Unsafe optimization without domain info
    let y = builder.var("y");
    let y_pow_0 = ASTRepr::Pow(Box::new(y.clone()), Box::new(ASTRepr::Constant(0.0)));
    println!("\nExpression: y^0 where y ∈ ℝ (unknown domain)");
    println!("  Original: {y_pow_0:?}");
    println!("  Conservative: No optimization");
    println!("  Reason: y could be 0, making 0^0 indeterminate");

    // Example 3: IEEE 754 specific case
    let zero_pow_zero = ASTRepr::Pow(
        Box::new(ASTRepr::Constant(0.0)),
        Box::new(ASTRepr::Constant(0.0)),
    );
    println!("\nExpression: 0^0 (literal constants)");
    println!("  Original: {zero_pow_zero:?}");
    println!("  IEEE 754 optimization: 1.0");
    println!("  Reason: IEEE 754 standard defines 0^0 = 1");

    println!("\n🎯 Advanced Domain Cases:");
    println!("-------------------------");

    // Case 1: Interval that excludes zero
    let interval_1_to_5 = IntervalDomain::closed_interval(1.0, 5.0);
    println!("Domain [1, 5]: {interval_1_to_5}");
    println!("  Contains zero? {}", interval_1_to_5.contains_zero(0.0));
    println!("  Is positive? {}", interval_1_to_5.is_positive(0.0));
    println!("  Safe for x^0 = 1? ✅ Yes");

    // Case 2: Interval that includes zero
    let interval_neg1_to_1 = IntervalDomain::closed_interval(-1.0, 1.0);
    println!("\nDomain [-1, 1]: {interval_neg1_to_1}");
    println!("  Contains zero? {}", interval_neg1_to_1.contains_zero(0.0));
    println!("  Is positive? {}", interval_neg1_to_1.is_positive(0.0));
    println!("  Safe for x^0 = 1? ❌ No");

    // Case 3: Open interval excluding zero
    let open_interval = IntervalDomain::open_interval(0.0, 1.0);
    println!("\nDomain (0, 1): {open_interval}");
    println!("  Contains zero? {}", open_interval.contains_zero(0.0));
    println!("  Is positive? {}", open_interval.is_positive(0.0));
    println!("  Safe for x^0 = 1? ✅ Yes");

    #[cfg(feature = "optimization")]
    {
        println!("\n🚀 Integration with Egglog Optimizer:");
        println!("-------------------------------------");

        match EgglogOptimizer::new() {
            Ok(mut optimizer) => {
                println!("✅ Created EgglogOptimizer");

                // Test with a simple expression that has domain implications
                let expr = ASTRepr::Add(
                    Box::new(ASTRepr::Pow(Box::new(x), Box::new(ASTRepr::Constant(0.0)))),
                    Box::new(ASTRepr::Constant(1.0)),
                );

                println!("Testing expression: x^0 + 1");

                match optimizer.optimize(&expr) {
                    Ok(optimized) => {
                        println!("Original:  {expr:?}");
                        println!("Optimized: {optimized:?}");

                        // Note: Current optimizer doesn't have domain awareness yet
                        println!("Note: Domain-aware optimization requires integration");
                        println!("      of IntervalDomainAnalyzer with EgglogOptimizer");
                    }
                    Err(e) => {
                        println!("❌ Optimization failed: {e}");
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to create optimizer: {e}");
            }
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n💡 Egglog integration skipped (optimization feature not enabled)");
        println!(
            "   Run with: cargo run --example domain_aware_optimization_demo --features optimization"
        );
    }

    println!("\n📊 Summary of Domain-Aware Approach:");
    println!("------------------------------------");
    println!("✅ Advantages:");
    println!("   • Mathematical correctness: No undefined behavior");
    println!("   • IEEE 754 compliance: Handles computational conventions");
    println!("   • Performance: Aggressive optimization when safe");
    println!("   • Flexibility: Rules adapt to known constraints");

    println!("\n🔧 Implementation Strategy:");
    println!("   1. Analyze expression domains using IntervalDomainAnalyzer");
    println!("   2. Generate domain-specific egglog rules");
    println!("   3. Apply IEEE 754 rules for literal constants");
    println!("   4. Use conservative rules when domain is unknown");

    println!("\n🎯 Next Steps:");
    println!("   • Integrate IntervalDomainAnalyzer with EgglogOptimizer");
    println!("   • Add absolute value to AST for √(x²) = |x|");
    println!("   • Implement conditional rewrite rules in egglog");
    println!("   • Add domain constraint propagation");

    Ok(())
}
