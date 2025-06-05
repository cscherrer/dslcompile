        match expr {
            ASTRepr::Constant(_) | ASTRepr::Variable(_) => false,
            ASTRepr::Add(_, _) | ASTRepr::Sub(_, _) | ASTRepr::Mul(_, _) | ASTRepr::Div(_, _) | ASTRepr::Pow(_, _) => true,
            ASTRepr::Neg(_) | ASTRepr::Ln(_) | ASTRepr::Exp(_) | ASTRepr::Sin(_) | ASTRepr::Cos(_) | ASTRepr::Sqrt(_) => true,
            ASTRepr::Sum { .. } => {
                // TODO: Implement Sum variant for normalization
                true // Treat as complex expression requiring normalization
            }
        } 