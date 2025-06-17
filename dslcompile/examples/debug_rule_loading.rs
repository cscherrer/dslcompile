//! Debug Rule Loading
//!
//! Simple test to debug what's happening with egglog rule loading

use dslcompile::symbolic::rule_loader::{RuleLoader, RuleConfig, RuleCategory};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 RULE LOADING DEBUG");
    println!("=====================\n");

    let config = RuleConfig {
        categories: vec![
            RuleCategory::CoreDatatypes,
            RuleCategory::DependencyAnalysis,
            RuleCategory::Summation,
        ],
        validate_syntax: true,
        include_comments: true,
        ..Default::default()
    };

    let rule_loader = RuleLoader::new(config);
    
    match rule_loader.load_rules() {
        Ok(program) => {
            println!("✅ Rules loaded successfully!");
            println!("Program length: {} characters", program.len());
            
            // Find the problematic line around 405
            let lines: Vec<&str> = program.lines().collect();
            println!("Total lines: {}", lines.len());
            
            if lines.len() > 400 {
                println!("\nLines around 405:");
                for (i, line) in lines.iter().enumerate().skip(400).take(15) {
                    println!("{:3}: {}", i + 1, line);
                }
            }
            
            // Also show the first few lines to see structure
            println!("\nFirst 10 lines:");
            for (i, line) in lines.iter().enumerate().take(10) {
                println!("{:3}: {}", i + 1, line);
            }
        }
        Err(e) => {
            println!("❌ Failed to load rules: {}", e);
        }
    }

    Ok(())
}