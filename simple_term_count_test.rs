use dslcompile::ast::ASTRepr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Simple Term Count Analysis");
    println!("=============================");

    let test_sizes = [3, 5, 10];

    for &n in &test_sizes {
        println!("\n📊 Dataset size: {n} observations");

        // Build the same expression as in the IID demo
        let mu = ASTRepr::Variable(0);
        let sigma = ASTRepr::Variable(1);

        let mut likelihood_terms = Vec::new();
        for i in 0..n {
            let x_i = ASTRepr::Variable(2 + i);
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

        // Count terms
        let total_nodes = count_nodes(&iid_likelihood);
        let variables = count_variables(&iid_likelihood);
        let constants = count_constants(&iid_likelihood);
        let operations = count_operations(&iid_likelihood);

        println!("   📋 Expression Analysis:");
        println!("      Total nodes: {}", total_nodes);
        println!("      Variables: {}", variables);
        println!("      Constants: {}", constants);
        println!("      Operations: {}", operations);
        println!("      Nodes per observation: {:.1}", total_nodes as f64 / n as f64);
    }

    println!("\n🎯 Analysis Summary:");
    println!("   • Check if complexity grows linearly with dataset size");
    println!("   • Each Gaussian term has many repeated subexpressions");
    println!("   • CSE should eliminate redundant (x-μ)/σ computations");

    Ok(())
}

fn count_nodes(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 1,
        ASTRepr::Add(left, right) | ASTRepr::Sub(left, right) | 
        ASTRepr::Mul(left, right) | ASTRepr::Div(left, right) | 
        ASTRepr::Pow(left, right) => 1 + count_nodes(left) + count_nodes(right),
        ASTRepr::Neg(inner) | ASTRepr::Ln(inner) | ASTRepr::Exp(inner) | 
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) => 1 + count_nodes(inner),
        _ => 1, // Simplified for other variants
    }
}

fn count_variables(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Variable(_) => 1,
        ASTRepr::Constant(_) => 0,
        ASTRepr::Add(left, right) | ASTRepr::Sub(left, right) | 
        ASTRepr::Mul(left, right) | ASTRepr::Div(left, right) | 
        ASTRepr::Pow(left, right) => count_variables(left) + count_variables(right),
        ASTRepr::Neg(inner) | ASTRepr::Ln(inner) | ASTRepr::Exp(inner) | 
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) => count_variables(inner),
        _ => 0,
    }
}

fn count_constants(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) => 1,
        ASTRepr::Variable(_) => 0,
        ASTRepr::Add(left, right) | ASTRepr::Sub(left, right) | 
        ASTRepr::Mul(left, right) | ASTRepr::Div(left, right) | 
        ASTRepr::Pow(left, right) => count_constants(left) + count_constants(right),
        ASTRepr::Neg(inner) | ASTRepr::Ln(inner) | ASTRepr::Exp(inner) | 
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) => count_constants(inner),
        _ => 0,
    }
}

fn count_operations(expr: &ASTRepr<f64>) -> usize {
    match expr {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) => 0,
        ASTRepr::Add(left, right) | ASTRepr::Sub(left, right) | 
        ASTRepr::Mul(left, right) | ASTRepr::Div(left, right) | 
        ASTRepr::Pow(left, right) => 1 + count_operations(left) + count_operations(right),
        ASTRepr::Neg(inner) | ASTRepr::Ln(inner) | ASTRepr::Exp(inner) | 
        ASTRepr::Sin(inner) | ASTRepr::Cos(inner) => 1 + count_operations(inner),
        _ => 1,
    }
} 