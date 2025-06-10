//! Minimal CSE Debug Test
//! Tests whether our CSE rules are firing on the simplest possible case

use dslcompile::ast::DynamicContext;
use dslcompile::symbolic::native_egglog::optimize_with_native_egglog;
use dslcompile::ast::advanced::ast_from_expr;
use std::process::Command;

fn main() {
    println!("🔍 Testing CSE Rules Directly in Egglog");
    println!("=======================================");
    
    // Test 1: Simple multiplication CSE
    let test1 = r#"
(datatype Math
  (Num f64)
  (UserVar i64)
  (BoundVar i64)
  (Let i64 Math Math)
  (Add Math Math)
  (Mul Math Math)
  (Neg Math))

; Test expression: x * x (should trigger CSE Rule 1)
(let expr1 (Mul (UserVar 0) (UserVar 0)))

; Load CSE rules
(include "src/egglog_rules/cse_rules.egg")

; Run optimization
(run 10)

; Extract result
(extract expr1)
"#;
    
    // Write test to temporary file
    std::fs::write("test_cse.egg", test1).expect("Failed to write test file");
    
    // Run egglog directly
    let output = Command::new("../egglog/target/release/egglog")
        .arg("test_cse.egg")
        .output();
    
    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);
            
            println!("📤 Egglog Output:");
            println!("{}", stdout);
            
            if !stderr.is_empty() {
                println!("⚠️ Egglog Errors:");
                println!("{}", stderr);
            }
            
            // Check if CSE fired
            if stdout.contains("Let") || stdout.contains("BoundVar") {
                println!("✅ CSE rules fired! Found Let/BoundVar in output.");
            } else {
                println!("❌ CSE rules did NOT fire. Expression unchanged.");
            }
        }
        Err(e) => {
            println!("⚠️ Failed to run egglog: {}", e);
            println!("Make sure egglog is built: cd ../egglog && cargo build --release");
        }
    }
    
    // Clean up
    let _ = std::fs::remove_file("test_cse.egg");
} 