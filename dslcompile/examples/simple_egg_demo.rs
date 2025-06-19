//! Simple Egg Demonstration: Non-Additive Cost Functions
//!
//! This is a working demonstration of egg's CostFunction trait showing
//! how non-additive cost functions work for mathematical expressions.

#[cfg(feature = "optimization")]
mod egg_demo {
    use egg::{*, rewrite as rw};
    use ordered_float::OrderedFloat;
    use std::collections::HashMap;

    // Define a simple mathematical language
    define_language! {
        enum SimpleMath {
            // Basic values
            Num(OrderedFloat<f64>),
            Var(Symbol),
            
            // Binary operations  
            "+" = Add([Id; 2]),
            "*" = Mul([Id; 2]),
            "^" = Pow([Id; 2]),
            
            // Unary operations
            "ln" = Ln([Id; 1]),
            "exp" = Exp([Id; 1]),
            
            // Summation operation (the key non-additive case)
            "sum" = Sum([Id; 1]),
        }
    }

    // Analysis to track variable dependencies
    #[derive(Default)]
    struct DepAnalysis;

    impl Analysis<SimpleMath> for DepAnalysis {
        type Data = std::collections::BTreeSet<Symbol>;
        
        fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
            let old_len = to.len();
            to.extend(from);
            DidMerge(to.len() > old_len, false)
        }
        
        fn make(_egraph: &EGraph<SimpleMath, Self>, enode: &SimpleMath) -> Self::Data {
            let mut deps = std::collections::BTreeSet::new();
            match enode {
                SimpleMath::Var(name) => {
                    deps.insert(*name);
                }
                _ => {} // Other nodes don't add new dependencies
            }
            deps
        }
    }

    /// Custom cost function demonstrating non-additive costs for summations
    pub struct SummationCostFunction {
        /// Estimated sizes for collection operations
        collection_estimates: HashMap<Id, usize>,
        /// Base operation costs
        operation_costs: HashMap<&'static str, f64>,
    }

    impl SummationCostFunction {
        pub fn new() -> Self {
            let mut operation_costs = HashMap::new();
            operation_costs.insert("add", 1.0);
            operation_costs.insert("mul", 2.0);
            operation_costs.insert("pow", 10.0);
            operation_costs.insert("ln", 15.0);
            operation_costs.insert("exp", 15.0);
            operation_costs.insert("sum", 1000.0); // Base summation cost
            
            Self {
                collection_estimates: HashMap::new(),
                operation_costs,
            }
        }
        
        /// Estimate collection size (simplified)
        fn estimate_collection_size(&mut self, _collection_id: Id) -> usize {
            // For this demo, assume a fixed collection size
            // In a real implementation, this would analyze the collection structure
            100
        }
        
        /// Analyze coupling complexity based on variable dependencies
        fn analyze_coupling(&self, dependencies: &std::collections::BTreeSet<Symbol>) -> f64 {
            match dependencies.len() {
                0 => 1.0,  // No coupling
                1 => 1.0,  // Simple dependency
                2 => 1.5,  // Moderate coupling
                3 => 2.0,  // High coupling
                _ => 3.0,  // Very high coupling
            }
        }
    }

    impl CostFunction<SimpleMath> for SummationCostFunction {
        type Cost = f64;
        
        fn cost<C>(&mut self, enode: &SimpleMath, mut costs: C) -> Self::Cost
        where
            C: FnMut(Id) -> Self::Cost,
        {
            match enode {
                // Constants and variables have minimal cost
                SimpleMath::Num(_) => 0.1,
                SimpleMath::Var(_) => 0.5,
                
                // Binary operations: sum child costs + operation cost
                SimpleMath::Add([a, b]) => {
                    costs(*a) + costs(*b) + self.operation_costs["add"]
                }
                SimpleMath::Mul([a, b]) => {
                    costs(*a) + costs(*b) + self.operation_costs["mul"]
                }
                SimpleMath::Pow([a, b]) => {
                    let base_cost = costs(*a);
                    let exp_cost = costs(*b);
                    base_cost + exp_cost + self.operation_costs["pow"]
                }
                
                // Unary operations
                SimpleMath::Ln([a]) => {
                    costs(*a) + self.operation_costs["ln"]
                }
                SimpleMath::Exp([a]) => {
                    costs(*a) + self.operation_costs["exp"]
                }
                
                // SUMMATION: The key non-additive cost case
                SimpleMath::Sum([collection]) => {
                    let inner_cost = costs(*collection);
                    let collection_size = self.estimate_collection_size(*collection);
                    
                    // For this demo, assume the collection has some variable dependencies
                    // In a real implementation, we'd analyze the actual dependencies
                    let mock_dependencies = std::collections::BTreeSet::from([Symbol::from("x"), Symbol::from("y")]);
                    let coupling_multiplier = self.analyze_coupling(&mock_dependencies);
                    
                    // NON-ADDITIVE COST CALCULATION:
                    // Total cost = base_summation_cost + (collection_size * inner_complexity * coupling)
                    let base_cost = self.operation_costs["sum"];
                    let iteration_cost = (collection_size as f64) * inner_cost * coupling_multiplier;
                    
                    let total_cost = base_cost + iteration_cost;
                    
                    println!(
                        "💰 NON-ADDITIVE Summation cost: base={:.1}, size={}, inner={:.1}, coupling={:.1}x → total={:.1}",
                        base_cost, collection_size, inner_cost, coupling_multiplier, total_cost
                    );
                    
                    total_cost
                }
            }
        }
    }

    /// Mathematical rewrite rules
    fn make_rules() -> Vec<Rewrite<SimpleMath, DepAnalysis>> {
        vec![
            // Basic arithmetic identities
            rw!("add-comm"; "(+ ?a ?b)" => "(+ ?b ?a)"),
            rw!("mul-comm"; "(* ?a ?b)" => "(* ?b ?a)"),
            rw!("add-zero"; "(+ ?a 0.0)" => "?a"),
            rw!("mul-one"; "(* ?a 1.0)" => "?a"),
            rw!("mul-zero"; "(* ?a 0.0)" => "0.0"),
            
            // Power rules
            rw!("pow-one"; "(^ ?x 1.0)" => "?x"),
            rw!("pow-zero"; "(^ ?x 0.0)" => "1.0"),
            
            // Logarithm and exponential
            rw!("ln-exp"; "(ln (exp ?x))" => "?x"),
            
            // Distributivity (can increase expression size)
            rw!("distribute"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        ]
    }

    pub fn demonstrate_non_additive_costs() {
        println!("🚀 Egg Non-Additive Cost Function Demonstration");
        println!("==============================================");
        
        // Create e-graph with dependency analysis
        let mut egraph: EGraph<SimpleMath, DepAnalysis> = Default::default();
        
        // Build a simple mathematical expression: (x + 1) * 2
        let x = egraph.add(SimpleMath::Var("x".into()));
        let one = egraph.add(SimpleMath::Num(OrderedFloat(1.0)));
        let two = egraph.add(SimpleMath::Num(OrderedFloat(2.0)));
        let add_expr = egraph.add(SimpleMath::Add([x, one]));
        let mul_expr = egraph.add(SimpleMath::Mul([add_expr, two]));
        
        println!("   Expression: (x + 1) * 2");
        
        // Build a summation expression: sum((x + 1) * 2)  
        let sum_expr = egraph.add(SimpleMath::Sum([mul_expr]));
        
        println!("   Summation: sum((x + 1) * 2)");
        
        // Run rewrite rules
        let rules = make_rules();
        let runner = Runner::default()
            .with_egraph(egraph)
            .run(&rules);
        
        println!("   Rewriting completed in {} iterations", runner.iterations.len());
        
        // Extract with standard AST size cost function
        println!("\n🔢 Standard Additive Cost Extraction:");
        let ast_extractor = Extractor::new(&runner.egraph, AstSize);
        let (ast_cost, ast_expr) = ast_extractor.find_best(sum_expr);
        println!("   Cost: {:.1} (additive)", ast_cost);
        println!("   Expression: {}", ast_expr);
        
        // Extract with custom non-additive cost function
        println!("\n💰 Non-Additive Summation Cost Extraction:");
        let summation_cost_fn = SummationCostFunction::new();
        let sum_extractor = Extractor::new(&runner.egraph, summation_cost_fn);
        let (sum_cost, sum_expr) = sum_extractor.find_best(sum_expr);
        println!("   Expression: {}", sum_expr);
        
        // Demonstrate the key insight: cost calculation is non-additive
        println!("\n🎯 Key Insight: Non-Additive Cost Modeling");
        println!("   Standard cost functions are additive: cost(parent) = sum(child_costs) + operation_cost");
        println!("   Our summation cost is multiplicative: cost(sum) = base + (size × inner_complexity × coupling)");
        println!("   This enables sophisticated cost modeling for collection operations!");
        
        // Compare costs for different scenarios
        println!("\n📊 Cost Comparison for Different Expression Structures:");
        demonstrate_cost_scenarios(&runner.egraph);
    }

    fn demonstrate_cost_scenarios(egraph: &EGraph<SimpleMath, DepAnalysis>) {
        let mut cost_fn = SummationCostFunction::new();
        
        // Scenario 1: Simple expression vs summation of simple expression
        let simple_cost = cost_fn.cost(&SimpleMath::Add([Id::from(0), Id::from(1)]), |_| 1.0);
        let sum_simple_cost = cost_fn.cost(&SimpleMath::Sum([Id::from(0)]), |_| simple_cost);
        
        println!("   Simple addition cost: {:.1}", simple_cost);
        println!("   Sum of simple addition: {:.1} ({}x multiplier)", 
                sum_simple_cost, sum_simple_cost / simple_cost);
        
        // Scenario 2: Complex expression vs summation of complex expression  
        let complex_cost = cost_fn.cost(&SimpleMath::Mul([Id::from(0), Id::from(1)]), |_| 15.0);
        let sum_complex_cost = cost_fn.cost(&SimpleMath::Sum([Id::from(0)]), |_| complex_cost);
        
        println!("   Complex expression cost: {:.1}", complex_cost);
        println!("   Sum of complex expression: {:.1} ({}x multiplier)",
                sum_complex_cost, sum_complex_cost / complex_cost);
                
        println!("\n✨ The non-additive cost function correctly models that:");
        println!("     - Summations are expensive proportional to collection size");
        println!("     - Cost scales with inner expression complexity");
        println!("     - Variable coupling affects computational difficulty");
    }
}

fn main() {
    #[cfg(feature = "optimization")]
    {
        egg_demo::demonstrate_non_additive_costs();
    }
    
    #[cfg(not(feature = "optimization"))]
    {
        println!("🥚 Egg Optimization Feature Not Enabled");
        println!("=======================================");
        println!("To see the non-additive cost function demonstration, run:");
        println!("   cargo run --example simple_egg_demo --features egg_optimization");
        println!();
        println!("This example shows how egg's CostFunction trait enables:");
        println!("   ✅ Non-additive cost modeling");
        println!("   ✅ Collection size-aware costs"); 
        println!("   ✅ Variable coupling analysis");
        println!("   ✅ Domain-specific optimization strategies");
        println!();
        println!("Key benefit: Sophisticated cost functions that go far beyond");
        println!("simple additive models, enabling better optimization decisions");
        println!("for mathematical expressions, especially summation operations.");
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "optimization")]
    #[test]
    fn test_egg_demo() {
        super::egg_demo::demonstrate_non_additive_costs();
    }
}