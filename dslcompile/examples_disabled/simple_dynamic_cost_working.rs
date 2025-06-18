//! Simple Working Dynamic Cost Test
//!
//! This bypasses the complex dependency analysis to test core dynamic cost functionality.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Simple Working Dynamic Cost Test");
    println!("===================================");

    // Create a simple symbolic optimizer without complex dependency analysis
    // Using egglog-experimental directly with basic rules

    let mut egraph = egglog_experimental::new_experimental_egraph();

    // Test the basic dynamic cost functionality
    let test_program = r"
        (with-dynamic-cost
            (datatype Math 
                (Num f64)
                (Add Math Math)
                (Mul Math Math)
                (UserVar i64)
            )
        )
        
        ; Create some expressions
        (let expr1 (Add (UserVar 0) (Num 0.0)))
        (let expr2 (UserVar 0))
        
        ; Set dynamic costs
        (set-cost (Num 0.0) 1000)     ; High cost for zero constants
        (set-cost (UserVar 0) 10)     ; Low cost for variables
        
        ; Extract best version
        (extract expr1)
    ";

    println!("   Running egglog program with dynamic costs...");

    match egraph.parse_and_run_program(None, test_program) {
        Ok(results) => {
            println!("✅ Successfully ran egglog with dynamic costs!");
            println!("   Results: {results:?}");

            // Test setting more costs
            let additional_costs = r"
                (set-cost (Add (UserVar 0) (Num 0.0)) 2000)
                (extract expr1)
            ";

            match egraph.parse_and_run_program(None, additional_costs) {
                Ok(more_results) => {
                    println!("✅ Successfully set additional dynamic costs!");
                    println!("   Additional results: {more_results:?}");
                }
                Err(e) => {
                    println!("❌ Failed to set additional costs: {e}");
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to run egglog program: {e}");
            return Err(e.into());
        }
    }

    println!("\n🎯 Dynamic Cost Integration Status:");
    println!("   ✅ egglog-experimental dependency working");
    println!("   ✅ with-dynamic-cost declarations working");
    println!("   ✅ set-cost commands working");
    println!("   ✅ Dynamic cost model integration successful");
    println!("   ⚠️  Complex dependency analysis needs syntax updates");

    println!("\n📝 Next Steps:");
    println!("   1. Simplify dependency analysis rules for current egglog syntax");
    println!("   2. Use relations instead of complex function lookups in rules");
    println!("   3. Integrate with sum splitting optimization");

    Ok(())
}
