use dslcompile::ast::ast_utils::remap_variables;
use dslcompile::final_tagless::DirectEval;
use dslcompile::prelude::*;
use std::collections::HashMap;

#[test]
fn test_shared_variable_composition_naive() {
    // Define f(x,y) = x² + xy + y² independently
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var(); // index 0 in f's context
    let y_f = math_f.var(); // index 1 in f's context
    let f_expr = &x_f * &x_f + &x_f * &y_f + &y_f * &y_f;

    // Define g(y,z) = 2y + 3z independently
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var(); // index 0 in g's context (should be y!)
    let z_g = math_g.var(); // index 1 in g's context (should be z!)
    let g_expr = 2.0 * &y_g + 3.0 * &z_g;

    // Naive combination - this is WRONG!
    // f uses variables [0,1] for [x,y]
    // g uses variables [0,1] for [y,z]
    // When combined, g's y (index 0) collides with f's x (index 0)
    let h_wrong = f_expr.as_ast().clone() + g_expr.as_ast().clone();

    // This evaluates as f(x,y) + g(x,y) instead of f(x,y) + g(y,z)
    let result_wrong = DirectEval::eval_with_vars(&h_wrong, &[1.0, 2.0]);
    // f(1,2) = 1 + 2 + 4 = 7
    // g(1,2) = 2*1 + 3*2 = 8  (but this should be g(2,z) for some z!)
    // Total: 15 (but this is wrong!)

    println!("Naive (wrong) result: h(1,2) = {result_wrong}");
    assert_eq!(result_wrong, 15.0); // This is the wrong answer!
}

#[test]
fn test_shared_variable_composition_manual() {
    // Define f(x,y) = x² + xy + y² independently
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var(); // index 0
    let y_f = math_f.var(); // index 1
    let f_expr = &x_f * &x_f + &x_f * &y_f + &y_f * &y_f;
    let f_ast = f_expr.as_ast();

    // Define g(y,z) = 2y + 3z independently
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var(); // index 0 in g's context
    let z_g = math_g.var(); // index 1 in g's context
    let g_expr = 2.0 * &y_g + 3.0 * &z_g;
    let g_ast = g_expr.as_ast();

    // Manual remapping for shared variables
    // Target variable layout: [x=0, y=1, z=2]
    // f(x,y): x=0, y=1 (no remapping needed)
    // g(y,z): y=1, z=2 (remap 0->1, 1->2)

    let mut g_var_map = HashMap::new();
    g_var_map.insert(0, 1); // g's first var (y) maps to index 1
    g_var_map.insert(1, 2); // g's second var (z) maps to index 2
    let g_remapped = remap_variables(g_ast, &g_var_map);

    // Now create h(x,y,z) = f(x,y) + g(y,z)
    let h_ast = dslcompile::ast::ASTRepr::Add(Box::new(f_ast.clone()), Box::new(g_remapped));

    // Test: h(1,2,3) = f(1,2) + g(2,3)
    // f(1,2) = 1² + 1*2 + 2² = 1 + 2 + 4 = 7
    // g(2,3) = 2*2 + 3*3 = 4 + 9 = 13
    // h(1,2,3) = 7 + 13 = 20
    let result = DirectEval::eval_with_vars(&h_ast, &[1.0, 2.0, 3.0]);
    assert_eq!(result, 20.0);

    println!("Manual remapping: h(1,2,3) = f(1,2) + g(2,3) = {result}");
}

#[test]
fn test_shared_variable_composition_systematic() {
    // A more systematic approach using variable name tracking

    #[derive(Debug, Clone)]
    struct NamedFunction {
        ast: dslcompile::ast::ASTRepr<f64>,
        var_names: Vec<String>, // Maps variable indices to semantic names
    }

    impl NamedFunction {
        fn new(ast: dslcompile::ast::ASTRepr<f64>, var_names: Vec<String>) -> Self {
            Self { ast, var_names }
        }

        fn remap_to_global_indices(
            &self,
            global_var_map: &HashMap<String, usize>,
        ) -> dslcompile::ast::ASTRepr<f64> {
            let mut local_to_global = HashMap::new();

            for (local_idx, var_name) in self.var_names.iter().enumerate() {
                if let Some(&global_idx) = global_var_map.get(var_name) {
                    local_to_global.insert(local_idx, global_idx);
                }
            }

            remap_variables(&self.ast, &local_to_global)
        }
    }

    // Define f(x,y) = x² + xy + y²
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var(); // local index 0
    let y_f = math_f.var(); // local index 1
    let f_expr = &x_f * &x_f + &x_f * &y_f + &y_f * &y_f;
    let f_named = NamedFunction::new(
        f_expr.as_ast().clone(),
        vec!["x".to_string(), "y".to_string()],
    );

    // Define g(y,z) = 2y + 3z
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var(); // local index 0
    let z_g = math_g.var(); // local index 1
    let g_expr = 2.0 * &y_g + 3.0 * &z_g;
    let g_named = NamedFunction::new(
        g_expr.as_ast().clone(),
        vec!["y".to_string(), "z".to_string()],
    );

    // Create global variable mapping
    let mut global_var_map = HashMap::new();
    global_var_map.insert("x".to_string(), 0);
    global_var_map.insert("y".to_string(), 1);
    global_var_map.insert("z".to_string(), 2);

    // Remap both functions to global indices
    let f_global = f_named.remap_to_global_indices(&global_var_map);
    let g_global = g_named.remap_to_global_indices(&global_var_map);

    // Compose h(x,y,z) = f(x,y) + g(y,z)
    let h_ast = dslcompile::ast::ASTRepr::Add(Box::new(f_global), Box::new(g_global));

    // Test: h(1,2,3) = f(1,2) + g(2,3) = 7 + 13 = 20
    let result = DirectEval::eval_with_vars(&h_ast, &[1.0, 2.0, 3.0]);
    assert_eq!(result, 20.0);

    println!("Systematic approach: h(1,2,3) = {result}");
}

#[test]
fn test_complex_shared_variable_case() {
    // More complex case: h(w,x,y,z) = f(x,y) + g(y,z) + k(w,x)

    // f(x,y) = sin(x) * cos(y)
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var();
    let y_f = math_f.var();
    let f_expr = x_f.clone().sin() * y_f.clone().cos();

    // g(y,z) = exp(y + z)
    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var();
    let z_g = math_g.var();
    let g_expr = (y_g.clone() + z_g.clone()).exp();

    // k(w,x) = w² - x²
    let math_k = ExpressionBuilder::new();
    let w_k = math_k.var();
    let x_k = math_k.var();
    let k_expr = &w_k * &w_k - &x_k * &x_k;

    // Create global variable mapping: [w=0, x=1, y=2, z=3]
    let mut global_var_map = HashMap::new();
    global_var_map.insert("w".to_string(), 0);
    global_var_map.insert("x".to_string(), 1);
    global_var_map.insert("y".to_string(), 2);
    global_var_map.insert("z".to_string(), 3);

    // Manual remapping for each function
    // f(x,y): local [0,1] -> global [1,2]
    let mut f_map = HashMap::new();
    f_map.insert(0, 1); // x
    f_map.insert(1, 2); // y
    let f_remapped = remap_variables(f_expr.as_ast(), &f_map);

    // g(y,z): local [0,1] -> global [2,3]
    let mut g_map = HashMap::new();
    g_map.insert(0, 2); // y
    g_map.insert(1, 3); // z
    let g_remapped = remap_variables(g_expr.as_ast(), &g_map);

    // k(w,x): local [0,1] -> global [0,1]
    let mut k_map = HashMap::new();
    k_map.insert(0, 0); // w
    k_map.insert(1, 1); // x
    let k_remapped = remap_variables(k_expr.as_ast(), &k_map);

    // Compose h(w,x,y,z) = f(x,y) + g(y,z) + k(w,x)
    let h_ast = dslcompile::ast::ASTRepr::Add(
        Box::new(dslcompile::ast::ASTRepr::Add(
            Box::new(f_remapped),
            Box::new(g_remapped),
        )),
        Box::new(k_remapped),
    );

    // Test with w=1, x=2, y=3, z=4
    let w_val = 1.0_f64;
    let x_val = 2.0_f64;
    let y_val = 3.0_f64;
    let z_val = 4.0_f64;

    let result = DirectEval::eval_with_vars(&h_ast, &[w_val, x_val, y_val, z_val]);

    // Expected:
    // f(2,3) = sin(2) * cos(3) ≈ 0.9093 * (-0.9899) ≈ -0.9004
    // g(3,4) = exp(3+4) = exp(7) ≈ 1096.63
    // k(1,2) = 1² - 2² = 1 - 4 = -3
    // h(1,2,3,4) ≈ -0.9004 + 1096.63 + (-3) ≈ 1092.73

    let expected_f = x_val.sin() * y_val.cos();
    let expected_g = (y_val + z_val).exp();
    let expected_k = w_val * w_val - x_val * x_val;
    let expected = expected_f + expected_g + expected_k;

    assert!((result - expected).abs() < 1e-10);

    println!("Complex composition: h(1,2,3,4) = {result}");
}

#[test]
fn test_automatic_shared_variable_detection() {
    // Demonstrate a more advanced approach that could automatically detect shared variables

    fn analyze_variable_usage(
        functions: &[(&dslcompile::ast::ASTRepr<f64>, &[&str])],
    ) -> (HashMap<String, usize>, Vec<HashMap<usize, usize>>) {
        let mut all_vars = std::collections::BTreeSet::new();

        // Collect all unique variable names
        for (_, var_names) in functions {
            for &name in *var_names {
                all_vars.insert(name.to_string());
            }
        }

        // Create global mapping
        let global_mapping: HashMap<String, usize> = all_vars
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();

        // Create local-to-global mappings for each function
        let mut local_mappings = Vec::new();
        for (_, var_names) in functions {
            let mut local_map = HashMap::new();
            for (local_idx, &var_name) in var_names.iter().enumerate() {
                if let Some(&global_idx) = global_mapping.get(var_name) {
                    local_map.insert(local_idx, global_idx);
                }
            }
            local_mappings.push(local_map);
        }

        (global_mapping, local_mappings)
    }

    // Define functions with their variable names
    let math_f = ExpressionBuilder::new();
    let x_f = math_f.var();
    let y_f = math_f.var();
    let f_expr = &x_f * &x_f + &y_f * &y_f; // f(x,y) = x² + y²
    let f_ast = f_expr.as_ast();

    let math_g = ExpressionBuilder::new();
    let y_g = math_g.var();
    let z_g = math_g.var();
    let g_expr = 2.0 * &y_g + &z_g; // g(y,z) = 2y + z
    let g_ast = g_expr.as_ast();

    // Analyze variable usage
    let functions: &[(&dslcompile::ast::ASTRepr<f64>, &[&str])] =
        &[(f_ast, &["x", "y"][..]), (g_ast, &["y", "z"][..])];

    let (global_mapping, local_mappings) = analyze_variable_usage(functions);

    println!("Global variable mapping: {global_mapping:?}");
    println!("Local mappings: {local_mappings:?}");

    // Apply remappings
    let f_remapped = remap_variables(f_ast, &local_mappings[0]);
    let g_remapped = remap_variables(g_ast, &local_mappings[1]);

    // Compose
    let h_ast = dslcompile::ast::ASTRepr::Add(Box::new(f_remapped), Box::new(g_remapped));

    // Test: h(x=1, y=2, z=3) = f(1,2) + g(2,3) = (1+4) + (4+3) = 5 + 7 = 12
    // Variable order should be [x=0, y=1, z=2] based on alphabetical sorting
    let result = DirectEval::eval_with_vars(&h_ast, &[1.0, 2.0, 3.0]);
    assert_eq!(result, 12.0);

    println!("Automatic detection: h(1,2,3) = {result}");
}
