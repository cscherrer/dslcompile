use dslcompile::ast::{ast_repr::ASTRepr, normalization::normalize};

fn main() {
    // Test that transcendental functions are preserved during normalization
    let expr = ASTRepr::Sub(
        Box::new(ASTRepr::Sin(Box::new(ASTRepr::<f64>::Variable(0)))),
        Box::new(ASTRepr::Ln(Box::new(ASTRepr::<f64>::Variable(1)))),
    );

    println!("Original: {:?}", expr);

    let normalized = normalize(&expr);
    println!("Normalized: {:?}", normalized);

    // Should still contain Sin and Ln operations
    match normalized {
        ASTRepr::Add(operands) => {
            let operands_vec = operands.as_vec();
            println!("Number of operands: {}", operands_vec.len());
            for (i, operand) in operands_vec.iter().enumerate() {
                println!("Operand {}: {:?}", i, operand);
            }
        }
        _ => println!("Not an Add operation: {:?}", normalized),
    }
}
