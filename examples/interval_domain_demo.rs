use mathcompile::final_tagless::{ASTEval, ASTMathExpr, ExpressionBuilder};
use mathcompile::interval_domain::{IntervalDomain, IntervalDomainAnalyzer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Interval Domain Analysis Demo");
    println!("================================");

    println!("\n📐 Mathematical Rigor:");
    println!("----------------------");

    // Demonstrate precise interval representation
    let positive = IntervalDomain::positive(0.0);
    let non_negative = IntervalDomain::non_negative(0.0);
    let interval_1_5 = IntervalDomain::closed_interval(1.0, 5.0);
    let open_interval = IntervalDomain::open_interval(0.0, 1.0);

    println!("Positive numbers: {positive}");
    println!("Non-negative:     {non_negative}");
    println!("Closed [1,5]:     {interval_1_5}");
    println!("Open (0,1):       {open_interval}");

    println!("\n🔍 Endpoint Precision:");
    println!("----------------------");

    // Show the difference between open and closed endpoints
    println!("Does (0,1) contain 0? {}", open_interval.contains(0.0));
    println!("Does (0,1) contain 1? {}", open_interval.contains(1.0));
    println!("Does [0,1] contain 0? {}", non_negative.contains(0.0));

    println!("\n🧮 Domain Operations:");
    println!("---------------------");

    // Demonstrate join and meet operations
    let pos = IntervalDomain::positive(0.0);
    let neg = IntervalDomain::negative(0.0);
    let joined = pos.join(&neg);

    println!("Positive ∪ Negative = {joined}");
    println!("Contains 0? {}", joined.contains(0.0));
    println!("Contains 1? {}", joined.contains(1.0));
    println!("Contains -1? {}", joined.contains(-1.0));

    // Meet operation
    let interval_0_10 = IntervalDomain::closed_interval(0.0, 10.0);
    let interval_5_15 = IntervalDomain::closed_interval(5.0, 15.0);
    let intersection = interval_0_10.meet(&interval_5_15);

    println!("[0,10] ∩ [5,15] = {intersection}");

    println!("\n🔬 Domain Analysis:");
    println!("-------------------");

    // Create expression builder and analyzer
    let builder = ExpressionBuilder::new();
    let mut analyzer = IntervalDomainAnalyzer::new(0.0);

    // Set up variables with domains - using index-based variables
    let x_var = builder.var(); // Returns TypedBuilderExpr<f64>
    analyzer.set_variable_domain(0, IntervalDomain::positive(0.0));

    // Convert to ASTRepr using ASTEval for analysis
    let x_ast = ASTEval::var(0); // Variable at index 0
    let ln_x = ASTEval::ln(x_ast.clone());
    let ln_domain = analyzer.analyze_domain(&ln_x);
    println!("Domain of ln(var_0) where var_0 > 0: {ln_domain}");

    // Analyze exp(anything) - always positive
    let exp_x = ASTEval::exp(x_ast);
    let exp_domain = analyzer.analyze_domain(&exp_x);
    println!("Domain of exp(var_0): {exp_domain}");

    println!("\n✨ Key Advantages:");
    println!("------------------");
    println!("✓ No redundancy: Positive = Interval{{lower: Open(0), upper: Unbounded}}");
    println!("✓ Full expressiveness: Can represent any mathematical interval");
    println!("✓ Precise semantics: Open vs closed endpoints are explicit");
    println!("✓ Infinite intervals: Unbounded endpoints handle ±∞ naturally");
    println!("✓ Uniform operations: Join/meet work consistently on all interval types");
    println!("✓ Index-based variables: High performance with clear variable management");

    Ok(())
}
