use dslcompile::{ast::ASTRepr, symbolic::native_egglog::optimize_with_native_egglog};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 AST Term Analysis for IID Probabilistic Programming");
    println!("======================================================");

    let test_sizes = [3, 5, 10]; // Start with smaller sizes to see the pattern

    for &n in &test_sizes {
        println!("\n📊 Dataset size: {n} observations");

        // Build the same expression as in the demo
        let mu = ASTRepr::Variable(0);
        let sigma = ASTRepr::Variable(1);

        let mut likelihood_terms = Vec::new();
        for i in 0..n {
            let x_i = ASTRepr::Variable(2 + i); // Variables 2, 3, 4, ...
            let diff = ASTRepr::Sub(Box::new(x_i), Box::new(mu.clone()));
            let standardized = ASTRepr::Div(Box::new(diff), Box::new(sigma.clone()));
            let standardized_squared =
                ASTRepr::Mul(Box::new(standardized.clone()), Box::new(standardized));
            let log_density_term = ASTRepr::Mul(
                Box::new(ASTRepr::Constant(-0.5)),
                Box::new(standardized_squared),
            );
            let log_sigma = ASTRepr::Ln(Box::new(sigma.clone()));
            let log_2pi = ASTRepr::Constant((2.0 * std::f64::consts::PI).ln());
            let half_log_2pi = ASTRepr::Mul(Box::new(ASTRepr::Constant(0.5)), Box::new(log_2pi));
            let normalization = ASTRepr::Add(Box::new(log_sigma), Box::new(half_log_2pi));
            let single_term = ASTRepr::Sub(Box::new(log_density_term), Box::new(normalization));
            likelihood_terms.push(single_term);
        }

        let mut iid_likelihood = likelihood_terms[0].clone();
        for term in likelihood_terms.into_iter().skip(1) {
            iid_likelihood = ASTRepr::Add(Box::new(iid_likelihood), Box::new(term));
        }

        // Analyze original expression
        let original_nodes = count_nodes(&iid_likelihood);
        let original_depth = count_depth(&iid_likelihood);
        let original_variables = count_variables(&iid_likelihood);
        let original_constants = count_constants(&iid_likelihood);
        let original_operations = count_operations(&iid_likelihood);

        println!("   📋 Original Expression Analysis:");
        println!("      Total nodes: {}", original_nodes);
        println!("      Max depth: {}", original_depth);
        println!("      Variables: {}", original_variables);
        println!("      Constants: {}", original_constants);
        println!("      Operations: {}", original_operations);

        // Time and analyze optimization
        #[cfg(feature = "optimization")]
        {
            let start = Instant::now();
            match optimize_with_native_egglog(&iid_likelihood) {
                Ok(optimized) => {
                    let opt_time = start.elapsed();

                    let opt_nodes = count_nodes(&optimized);
                    let opt_depth = count_depth(&optimized);
                    let opt_variables = count_variables(&optimized);
                    let opt_constants = count_constants(&optimized);
                    let opt_operations = count_operations(&optimized);

                    println!("   🔧 Optimized Expression Analysis:");
                    println!(
                        "      Total nodes: {} ({})",
                        opt_nodes,
                        if opt_nodes < original_nodes {
                            "↓ reduced"
                        } else if opt_nodes > original_nodes {
                            "↑ increased"
                        } else {
                            "= same"
                        }
                    );
                    println!(
                        "      Max depth: {} ({})",
                        opt_depth,
                        if opt_depth < original_depth {
                            "↓ reduced"
                        } else if opt_depth > original_depth {
                            "↑ increased"
                        } else {
                            "= same"
                        }
                    );
                    println!(
                        "      Variables: {} ({})",
                        opt_variables,
                        if opt_variables < original_variables {
                            "↓ reduced"
                        } else if opt_variables > original_variables {
                            "↑ increased"
                        } else {
                            "= same"
                        }
                    );
                    println!(
                        "      Constants: {} ({})",
                        opt_constants,
                        if opt_constants < original_constants {
                            "↓ reduced"
                        } else if opt_constants > original_constants {
                            "↑ increased"
                        } else {
                            "= same"
                        }
                    );
                    println!(
                        "      Operations: {} ({})",
                        opt_operations,
                        if opt_operations < original_operations {
                            "↓ reduced"
                        } else if opt_operations > original_operations {
                            "↑ increased"
                        } else {
                            "= same"
                        }
                    );

                    println!("   ⏱️  Optimization time: {:?}", opt_time);

                    // Check if expressions are actually different
                    let changed = !expressions_equal(&iid_likelihood, &optimized);
                    println!("   🔄 Expression changed: {}", changed);

                    if opt_time.as_millis() > 100 {
                        println!("   ⚠️  Optimization taking longer than expected!");
                    }
                }
                Err(e) => {
                    let opt_time = start.elapsed();
                    println!("   ❌ Optimization failed after {:?}: {}", opt_time, e);
                }
            }
        }

        #[cfg(not(feature = "optimization"))]
        {
            println!("   ⚠️  Optimization feature not enabled");
        }
    }

    println!("\n🎯 Analysis Summary:");
    println!("   • Check if node count grows exponentially with dataset size");
    println!("   • Look for optimization time scaling issues");
    println!("   • Identify if optimization actually reduces complexity");
    println!("   • Determine if expressions are meaningfully changed");

    Ok(())
}

// Helper functions to count AST components
fn count_nodes(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + count_nodes(left) + count_nodes(right),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + count_nodes(inner),
        ASTRepr::Sum(_) => 1,
        ASTRepr::BoundVar(_) => 1,
        ASTRepr::Let(_, bound, body) => 1 + count_nodes(bound) + count_nodes(body),
    }
}

fn count_depth(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + count_depth(left).max(count_depth(right)),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + count_depth(inner),
        ASTRepr::Sum(_) => 1,
        ASTRepr::BoundVar(_) => 1,
        ASTRepr::Let(_, bound, body) => 1 + count_depth(bound).max(count_depth(body)),
    }
}

fn count_variables(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Variable(_) => 1,
        ASTRepr::Constant(_) => 0,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => count_variables(left) + count_variables(right),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => count_variables(inner),
        ASTRepr::Sum(_) => 0,
        ASTRepr::BoundVar(_) => 1,
        ASTRepr::Let(_, bound, body) => count_variables(bound) + count_variables(body),
    }
}

fn count_constants(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) => 1,
        ASTRepr::Variable(_) => 0,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => count_constants(left) + count_constants(right),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => count_constants(inner),
        ASTRepr::Sum(_) => 0,
        ASTRepr::BoundVar(_) => 0,
        ASTRepr::Let(_, bound, body) => count_constants(bound) + count_constants(body),
    }
}

fn count_operations(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
        ASTRepr::Add(left, right)
        | ASTRepr::Sub(left, right)
        | ASTRepr::Mul(left, right)
        | ASTRepr::Div(left, right)
        | ASTRepr::Pow(left, right) => 1 + count_operations(left) + count_operations(right),
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + count_operations(inner),
        ASTRepr::Sum(_) => 1,
        ASTRepr::BoundVar(_) => 0,
        ASTRepr::Let(_, bound, body) => 1 + count_operations(bound) + count_operations(body),
    }
}

fn expressions_equal(expr1: &ASTRepr<f64>, expr2: &ASTRepr<f64>) -> bool {
    // Simple structural equality check
    match (expr1, expr2) {
        (ASTRepr::Constant(a), ASTRepr::Constant(b)) => (a - b).abs() < 1e-10,
        (ASTRepr::Variable(a), ASTRepr::Variable(b)) => a == b,
        (ASTRepr::Add(a1, a2), ASTRepr::Add(b1, b2))
        | (ASTRepr::Sub(a1, a2), ASTRepr::Sub(b1, b2))
        | (ASTRepr::Mul(a1, a2), ASTRepr::Mul(b1, b2))
        | (ASTRepr::Div(a1, a2), ASTRepr::Div(b1, b2))
        | (ASTRepr::Pow(a1, a2), ASTRepr::Pow(b1, b2)) => {
            expressions_equal(a1, b1) && expressions_equal(a2, b2)
        }
        (ASTRepr::Neg(a), ASTRepr::Neg(b))
        | (ASTRepr::Ln(a), ASTRepr::Ln(b))
        | (ASTRepr::Exp(a), ASTRepr::Exp(b))
        | (ASTRepr::Sin(a), ASTRepr::Sin(b))
        | (ASTRepr::Cos(a), ASTRepr::Cos(b)) => expressions_equal(a, b),
        _ => false,
    }
}
