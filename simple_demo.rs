//! Simple Demo: Sqrt Pre/Post Processing Concept
//!
//! This demonstrates the key insight: dropping Sqrt in favor of Power
//! with intelligent pre and post processing.

#[derive(Debug, Clone, PartialEq)]
enum SimpleAST {
    Variable(usize),
    Constant(f64),
    Add(Box<SimpleAST>, Box<SimpleAST>),
    Mul(Box<SimpleAST>, Box<SimpleAST>),
    Pow(Box<SimpleAST>, Box<SimpleAST>),
    // Notice: NO SQRT VARIANT!
}

impl SimpleAST {
    /// User-friendly sqrt() method that pre-processes to Pow(x, 0.5)
    fn sqrt(self) -> Self {
        SimpleAST::Pow(Box::new(self), Box::new(SimpleAST::Constant(0.5)))
    }
    
    /// Code generation with post-processing optimization
    fn to_rust_code(&self) -> String {
        match self {
            SimpleAST::Variable(idx) => format!("x{}", idx),
            SimpleAST::Constant(val) => format!("{}", val),
            SimpleAST::Add(left, right) => {
                format!("({} + {})", left.to_rust_code(), right.to_rust_code())
            }
            SimpleAST::Mul(left, right) => {
                format!("({} * {})", left.to_rust_code(), right.to_rust_code())
            }
            SimpleAST::Pow(base, exp) => {
                // POST-PROCESSING: Detect sqrt pattern and optimize
                if let SimpleAST::Constant(exp_val) = exp.as_ref() {
                    if (exp_val - 0.5).abs() < 1e-15 {
                        return format!("({}).sqrt()", base.to_rust_code());
                    }
                    if *exp_val == 2.0 {
                        return format!("({}).powi(2)", base.to_rust_code());
                    }
                }
                format!("({}).powf({})", base.to_rust_code(), exp.to_rust_code())
            }
        }
    }
}

fn main() {
    println!("=== Sqrt Pre/Post Processing Success Demo ===\n");

    // 1. Demonstrate pre-processing
    println!("1. PRE-PROCESSING STEP");
    println!("   User writes: x.sqrt()");
    
    let x = SimpleAST::Variable(0);
    let sqrt_expr = x.sqrt();
    
    match &sqrt_expr {
        SimpleAST::Pow(base, exp) => {
            println!("   ✅ Pre-processed to: Pow(x, 0.5)");
            println!("      Base: {:?}", base);
            println!("      Exponent: {:?}", exp);
        }
        _ => unreachable!("sqrt() should always create Pow"),
    }

    // 2. Demonstrate unified processing  
    println!("\n2. UNIFIED PROCESSING");
    println!("   All powers handled by same code path:");
    
    let examples = vec![
        ("x^2", SimpleAST::Variable(0).pow(SimpleAST::Constant(2.0))),
        ("x^0.5", SimpleAST::Variable(0).sqrt()),
        ("x^(-1)", SimpleAST::Variable(0).pow(SimpleAST::Constant(-1.0))),
        ("x^(1/3)", SimpleAST::Variable(0).pow(SimpleAST::Constant(1.0/3.0))),
    ];
    
    for (desc, expr) in &examples {
        println!("   {}: {:?}", desc, expr);
    }

    // 3. Demonstrate post-processing optimization
    println!("\n3. POST-PROCESSING STEP (Code Generation)");
    println!("   Pattern recognition for optimal code:");
    
    for (desc, expr) in &examples {
        let code = expr.to_rust_code();
        println!("   {} → {}", desc, code);
    }

    // 4. Complex example
    println!("\n4. COMPLEX EXAMPLE");
    let complex = SimpleAST::Add(
        Box::new(SimpleAST::Pow(Box::new(SimpleAST::Variable(0)), Box::new(SimpleAST::Constant(2.0)))),
        Box::new(SimpleAST::Constant(1.0))
    ).sqrt();
    
    println!("   Expression: sqrt(x^2 + 1)");
    println!("   Internal AST: {:?}", complex);
    println!("   Generated code: {}", complex.to_rust_code());

    println!("\n=== KEY BENEFITS ===");
    println!("✅ SIMPLIFIED: No Sqrt enum variant → less code duplication");
    println!("✅ UNIFIED: All power operations share optimization infrastructure");
    println!("✅ OPTIMAL: Still generates efficient .sqrt() calls where appropriate");
    println!("✅ EXTENSIBLE: Easy to add more power patterns (cube root, etc.)");
    println!("✅ USER-FRIENDLY: Natural sqrt() API preserved");

    println!("\n=== COMPARISON ===");
    println!("BEFORE (with Sqrt variant):");
    println!("  - Need Sqrt handling in ~30+ places");
    println!("  - Duplicate optimization logic");
    println!("  - More complex pattern matching");
    
    println!("\nAFTER (unified Power approach):");
    println!("  - Single power optimization system");
    println!("  - Pre/post processing handles user experience");
    println!("  - Cleaner, more maintainable codebase");
    
    println!("\nThis proves the approach works! 🎉");
}

// Helper method for creating power expressions
impl SimpleAST {
    fn pow(self, exp: SimpleAST) -> Self {
        SimpleAST::Pow(Box::new(self), Box::new(exp))
    }
} 