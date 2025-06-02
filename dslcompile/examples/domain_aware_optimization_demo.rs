use dslcompile::final_tagless::{ASTEval, ASTMathExpr, ASTRepr, ExpressionBuilder};
use dslcompile::interval_domain::{IntervalDomain, IntervalDomainAnalyzer};
use dslcompile::symbolic::rule_loader::{RuleConfig, RuleLoader};

#[cfg(feature = "optimization")]
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

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
    let builder = ExpressionBuilder::new();
    let x_var = builder.var(); // Returns TypedBuilderExpr<f64>

    // x^0 where x > 0 - safe to optimize to 1
    // Use ASTEval to construct expressions instead of raw ASTRepr
    let x = ASTEval::var(0);
    let x_pow_0 = ASTEval::pow(x.clone(), ASTEval::constant(0.0));
    println!("Expression: var_0^0 where var_0 ∈ (0, +∞)");
    println!("  Original: var_0^0");
    println!("  Safe optimization: 1.0");
    println!("  Reason: var_0 > 0 guarantees var_0 ≠ 0");

    // Example 2: Unsafe optimization without domain info
    let y_var = builder.var(); // Returns TypedBuilderExpr<f64>
    let y = ASTEval::var(1);
    let y_pow_0 = ASTEval::pow(y.clone(), ASTEval::constant(0.0));
    println!("\nExpression: var_1^0 where var_1 ∈ ℝ (unknown domain)");
    println!("  Original: var_1^0");
    println!("  Conservative: No optimization");
    println!("  Reason: var_1 could be 0, making 0^0 indeterminate");

    // Example 3: IEEE 754 specific case
    let zero_pow_zero = ASTEval::pow(ASTEval::constant(0.0), ASTEval::constant(0.0));
    println!("\nExpression: 0^0 (literal constants)");
    println!("  Original: 0.0^0.0");
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
        println!("\n🚀 Integration with Native egglog Optimizer:");
        println!("-------------------------------------");

        let mut optimizer = NativeEgglogOptimizer::new()?;

        println!("\n📊 Testing Domain-Safe Optimizations:");
        println!("-------------------------------------");

        // Test 1: Always safe - ln(exp(x)) = x
        println!("\n1. Always Safe: ln(exp(x)) = x");
        let safe_expr = ASTRepr::Ln(Box::new(ASTRepr::Exp(Box::new(ASTRepr::Variable(0)))));

        println!("   Original: ln(exp(x))");
        match optimizer.optimize(&safe_expr) {
            Ok(optimized) => {
                println!("   ✅ Optimization successful");
                println!("   Result: {optimized:?}");
            }
            Err(e) => println!("   ❌ Optimization failed: {e}"),
        }

        // Test 2: Domain-dependent - exp(ln(x)) = x (only if x > 0)
        println!("\n2. Domain-Dependent: exp(ln(x)) = x");
        let domain_dependent = ASTRepr::Exp(Box::new(ASTRepr::Ln(Box::new(ASTRepr::Variable(0)))));

        println!("   Original: exp(ln(x))");
        println!("   Note: Only safe if x > 0");
        match optimizer.optimize(&domain_dependent) {
            Ok(optimized) => {
                println!("   ✅ Optimization completed");
                println!("   Result: {optimized:?}");
            }
            Err(e) => println!("   ❌ Optimization failed: {e}"),
        }

        // Test 3: Logarithm product rule - ln(a * b) = ln(a) + ln(b)
        println!("\n3. Logarithm Product Rule: ln(2 * 3) = ln(2) + ln(3)");
        let ln_product = ASTRepr::Ln(Box::new(ASTRepr::Mul(
            Box::new(ASTRepr::Constant(2.0)),
            Box::new(ASTRepr::Constant(3.0)),
        )));

        println!("   Original: ln(2 * 3)");
        println!("   Safe because both 2 and 3 are positive constants");
        match optimizer.optimize(&ln_product) {
            Ok(optimized) => {
                println!("   ✅ Optimization successful");
                println!("   Result: {optimized:?}");
            }
            Err(e) => println!("   ❌ Optimization failed: {e}"),
        }

        // Test 4: Square root simplification - sqrt(x^2) = |x| ≈ x
        println!("\n4. Square Root Simplification: sqrt(x^2) = |x|");
        let sqrt_square = ASTRepr::Sqrt(Box::new(ASTRepr::Pow(
            Box::new(ASTRepr::Variable(0)),
            Box::new(ASTRepr::Constant(2.0)),
        )));

        println!("   Original: sqrt(x^2)");
        println!("   Only simplifies to x if x ≥ 0");
        match optimizer.optimize(&sqrt_square) {
            Ok(optimized) => {
                println!("   ✅ Optimization completed");
                println!("   Result: {optimized:?}");
            }
            Err(e) => println!("   ❌ Optimization failed: {e}"),
        }

        println!("\n🔍 Testing Interval Analysis:");
        println!("-----------------------------");

        // Test interval analysis on different expression types
        let test_expressions = vec![
            ("Constant 5.0", ASTRepr::Constant(5.0)),
            ("Variable x", ASTRepr::Variable(0)),
            (
                "2 + 3*x",
                ASTRepr::Add(
                    Box::new(ASTRepr::Constant(2.0)),
                    Box::new(ASTRepr::Mul(
                        Box::new(ASTRepr::Constant(3.0)),
                        Box::new(ASTRepr::Variable(0)),
                    )),
                ),
            ),
            ("exp(x)", ASTRepr::Exp(Box::new(ASTRepr::Variable(0)))),
        ];

        for (name, expr) in test_expressions {
            println!("\n   Expression: {name}");
            match optimizer.analyze_interval(&expr) {
                Ok(interval_info) => {
                    println!("   ✅ Interval analysis: {interval_info}");
                }
                Err(e) => {
                    println!("   ❌ Analysis failed: {e}");
                }
            }
        }

        println!("\n🛡️  Testing Domain Safety Checks:");
        println!("----------------------------------");

        // Test domain safety for various operations
        let safety_tests = vec![
            ("ln(5.0)", ASTRepr::Constant(5.0), "ln"),
            (
                "sqrt(x^2)",
                ASTRepr::Pow(
                    Box::new(ASTRepr::Variable(0)),
                    Box::new(ASTRepr::Constant(2.0)),
                ),
                "sqrt",
            ),
            (
                "1/(x+1)",
                ASTRepr::Add(
                    Box::new(ASTRepr::Variable(0)),
                    Box::new(ASTRepr::Constant(1.0)),
                ),
                "div",
            ),
        ];

        for (name, expr, operation) in safety_tests {
            println!("\n   Checking: {name} for {operation} operation");
            match optimizer.is_domain_safe(&expr, operation) {
                Ok(is_safe) => {
                    let status = if is_safe {
                        "✅ SAFE"
                    } else {
                        "⚠️  UNSAFE"
                    };
                    println!("   Result: {status}");
                }
                Err(e) => {
                    println!("   ❌ Safety check failed: {e}");
                }
            }
        }

        println!("\n🎯 Key Benefits of Domain-Aware Optimization:");
        println!("----------------------------------------------");
        println!("• Prevents mathematical errors (NaN, undefined results)");
        println!("• Enables more aggressive optimizations when safe");
        println!("• Provides interval analysis for numerical stability");
        println!("• Uses egglog's native abstract interpretation");
        println!("• Follows the proven Herbie approach");

        println!("\n🔮 Future Enhancements:");
        println!("----------------------");
        println!("• Complete interval extraction from egglog");
        println!("• Multiple lattice analyses (intervals + not-equals)");
        println!("• User-provided domain constraints");
        println!("• Cost-based extraction with domain information");
        println!("• Integration with constraint solvers");
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n💡 Native egglog integration skipped (optimization feature not enabled)");
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
    println!("   • Integrate IntervalDomainAnalyzer with Native egglog Optimizer");
    println!("   • Add absolute value to AST for √(x²) = |x|");
    println!("   • Implement conditional rewrite rules in egglog");
    println!("   • Add domain constraint propagation");

    Ok(())
}
