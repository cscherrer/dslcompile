//! Research egglog-experimental API for direct Rust integration
//!
//! This explores what APIs are available for programmatic interaction
//! with egglog without using string-based parsing.

use std::collections::HashMap;

#[cfg(feature = "optimization")]
use egglog_experimental::{EGraph, new_experimental_egraph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Researching egglog-experimental API");
    println!("=====================================");
    
    #[cfg(feature = "optimization")]
    {
        // Create an EGraph instance
        let egraph = new_experimental_egraph();
        println!("✅ Created experimental EGraph instance");
        
        // Try to inspect what methods are available
        println!("\n📋 EGraph type info:");
        println!("   Type: {}", std::any::type_name::<EGraph>());
        
        // Test basic string-based interface (current approach)
        test_string_interface(egraph)?;
    }
    
    #[cfg(not(feature = "optimization"))]
    {
        println!("❌ Optimization feature not enabled");
    }
    
    Ok(())
}

#[cfg(feature = "optimization")]
fn test_string_interface(mut egraph: EGraph) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing String-Based Interface (Current)");
    println!("-------------------------------------------");
    
    let program = r"
        (datatype Math 
            (Num f64)
            (Add Math Math)
            (UserVar i64)
        )
        
        (let x (Add (UserVar 0) (Num 0.0)))
        (let y (UserVar 0))
        
        ; Basic rules
        (rule ((= lhs (Add a (Num 0.0))))
              ((union lhs a)))
        
        (run 3)
        (extract x)
    ";
    
    match egraph.parse_and_run_program(None, program) {
        Ok(results) => {
            println!("✅ String-based interface works");
            println!("   Results: {:?}", results);
        }
        Err(e) => {
            println!("❌ String-based interface failed: {}", e);
        }
    }
    
    // Try to explore available methods using reflection-like approaches
    explore_egraph_methods(&egraph);
    
    Ok(())
}

#[cfg(feature = "optimization")]
fn explore_egraph_methods(egraph: &EGraph) {
    println!("\n🔍 Exploring EGraph Methods");
    println!("---------------------------");
    
    // We can't directly inspect methods in Rust without macros,
    // but we can test common patterns
    
    println!("Available methods would typically include:");
    println!("   - Methods for adding terms/expressions");
    println!("   - Methods for adding rules");
    println!("   - Methods for running saturation");
    println!("   - Methods for extraction");
    println!("   - Methods for querying the e-graph");
    
    // Check if there are any public fields or methods we can access
    // This is limited without full API documentation
    
    println!("\n⚠️  Need to check source code or generated docs for full API");
}

// Research questions to investigate:
#[allow(dead_code)]
fn research_questions() {
    println!("\n📝 Research Questions for egglog Direct Integration:");
    println!("1. Can we construct egglog terms directly using Rust data structures?");
    println!("2. Are there builder patterns for creating expressions?");
    println!("3. Can we add rewrite rules programmatically?");
    println!("4. Is there direct access to the e-graph state?");
    println!("5. Are there APIs for custom extraction functions?");
    println!("6. Can we register custom sorts/datatypes from Rust?");
    println!("7. Is there support for incremental updates?");
    println!("8. Are there hooks for custom analysis?");
}