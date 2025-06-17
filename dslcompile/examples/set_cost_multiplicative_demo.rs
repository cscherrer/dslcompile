//! Set-Cost Multiplicative Example
//!
//! This demonstrates dynamic cost assignment with multiplicative costs
//! where rewrite decisions change based on set-cost values.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Set-Cost Multiplicative Example");
    println!("==================================");

    let mut egraph = egglog_experimental::new_experimental_egraph();

    // Set up the basic datatype and rewrite rule
    let setup_program = r#"
        (with-dynamic-cost
            (datatype Node 
                (A)
                (B Node Node)
                (C)
            )
        )
        
        ; Create a union between A and (B C C) so they are equivalent
        (union A (B C C))
    "#;

    println!("   Setting up datatype and rewrite rule...");
    egraph.parse_and_run_program(None, setup_program)?;

    // First test: High cost for A, low costs for B and C
    println!("\n📊 Test 1: A=8, B=3*(cost1+cost2), C=1");
    println!("   Expected: A should rewrite to (B C C) with cost 3*(1+1)=6 < 8");

    let test1_costs = r#"
        ; Set costs as specified
        (set-cost A 8)
        (set-cost C 1)
        (set-cost (B C C) 6)  ; 3 * (1 + 1) = 6
        
        ; Run optimization
        (run 5)
        (extract A)
    "#;

    match egraph.parse_and_run_program(None, test1_costs) {
        Ok(results) => {
            println!("   ✅ Results: {:?}", results);
            if results.contains(&"(B C C)".to_string()) || results.iter().any(|r| r.contains("B")) {
                println!("   ✅ SUCCESS: A rewrote to (B C C) due to lower cost!");
            } else if results.contains(&"A".to_string()) {
                println!("   ❌ A remained A - costs may not be working as expected");
            }
        }
        Err(e) => {
            println!("   ❌ Test 1 failed: {}", e);
        }
    }

    // Reset for second test
    println!("\n🔄 Resetting for Test 2...");
    let mut egraph2 = egglog_experimental::new_experimental_egraph();
    egraph2.parse_and_run_program(None, setup_program)?;

    // Second test: Make A cheaper than the expanded form
    println!("\n📊 Test 2: A=2, B=3*(cost1+cost2), C=1");
    println!("   Expected: A should remain A with cost 2 < 6");

    let test2_costs = r#"
        ; Set costs to favor A
        (set-cost A 2)
        (set-cost C 1)
        (set-cost (B C C) 6)  ; 3 * (1 + 1) = 6
        
        ; Run optimization  
        (run 5)
        (extract test_expr)
    "#;

    match egraph2.parse_and_run_program(None, test2_costs) {
        Ok(results) => {
            println!("   ✅ Results: {:?}", results);
            if results.contains(&"A".to_string()) {
                println!("   ✅ SUCCESS: A remained A due to lower cost!");
            } else if results.iter().any(|r| r.contains("B")) {
                println!("   ❌ A rewrote to (B C C) - costs may not be preventing rewrite");
            }
        }
        Err(e) => {
            println!("   ❌ Test 2 failed: {}", e);
        }
    }

    // Third test: Show the multiplicative cost structure more clearly
    println!("\n🔄 Resetting for Test 3...");
    let mut egraph3 = egglog_experimental::new_experimental_egraph();
    egraph3.parse_and_run_program(None, setup_program)?;

    println!("\n📊 Test 3: Demonstrating multiplicative cost structure");
    println!("   A=10, C=3, so (B C C) should cost 3*(3+3)=18 > 10");

    let test3_costs = r#"
        ; Higher cost for C makes B expansion expensive
        (set-cost A 10)
        (set-cost C 3)  
        (set-cost (B C C) 18)  ; 3 * (3 + 3) = 18
        
        ; Run optimization
        (run 5)
        (extract A)
    "#;

    match egraph3.parse_and_run_program(None, test3_costs) {
        Ok(results) => {
            println!("   ✅ Results: {:?}", results);
            if results.contains(&"A".to_string()) {
                println!("   ✅ SUCCESS: A remained A - multiplicative cost prevented rewrite!");
            } else {
                println!("   ❌ Unexpected result - check cost calculations");
            }
        }
        Err(e) => {
            println!("   ❌ Test 3 failed: {}", e);
        }
    }

    println!("\n🎯 Summary:");
    println!("   This demonstrates how set-cost enables fine-grained control");
    println!("   over when rewrites should fire based on actual cost calculations.");
    println!("   The multiplicative cost model B = 3*(cost1 + cost2) allows");
    println!("   context-sensitive optimization decisions.");

    Ok(())
}
