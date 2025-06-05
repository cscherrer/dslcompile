use dslcompile::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 Testing Fixed Symbolic Sum");
    println!("==============================");

    let math = DynamicContext::new();
    let k = math.var(); // Parameter variable

    // Test data
    let data = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];

    // ✅ Create SYMBOLIC SUM AST node - no more stack overflow!
    let sum_expr = math.sum(data, |(x, _y)| {
        k.clone() * x // k * x for each data point
    })?;

    println!("✅ Successfully created symbolic sum expression!");
    println!("   Pattern: k * x");
    println!("   Data points: (1,0), (2,0), (3,0)");

    // Evaluate with k=2.0
    let result = math.eval(&sum_expr, &[2.0]);
    println!("   Result with k=2.0: {}", result);
    println!("   Expected: 2*1 + 2*2 + 2*3 = 12");

    if (result - 12.0).abs() < 1e-10 {
        println!("✅ SUCCESS: Symbolic sum working correctly!");
    } else {
        println!("❌ FAILED: Got {}, expected 12", result);
    }

    Ok(())
} 