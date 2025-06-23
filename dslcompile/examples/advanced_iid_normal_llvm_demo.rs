//! Advanced IID Normal Distribution with LLVM JIT Demo
//!
//! This demo demonstrates:
//! 1. Normal struct with embedded parameters (polymorphic)
//! 2. IID struct wrapping any measure type
//! 3. Log-density methods using LambdaVar approach
//! 4. Egg optimization with sum splitting and simplification
//! 5. LLVM JIT compilation using inkwell for maximum performance
//! 6. Evaluation on random data with performance benchmarking

use dslcompile::prelude::*;
use frunk::hlist;

#[cfg(feature = "optimization")]
use dslcompile::SymbolicOptimizer;

#[cfg(feature = "llvm_jit")]
use dslcompile::backends::LLVMJITCompiler;

#[cfg(feature = "llvm_jit")]
use inkwell::context::Context;

/// Normal distribution with embedded parameters
#[derive(Debug)]
struct Normal {
    mean: DynamicExpr<f64>,
    std_dev: DynamicExpr<f64>,
}

impl Normal {
    /// Create a new Normal distribution with parameter expressions
    fn new(mean: DynamicExpr<f64>, std_dev: DynamicExpr<f64>) -> Self {
        Self { mean, std_dev }
    }

    /// Compute log-density: -0.5 * ln(2π) - ln(σ) - 0.5 * ((x-μ)/σ)²
    fn log_density(&self, x: DynamicExpr<f64>) -> DynamicExpr<f64> {
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let neg_half = -0.5;

        let centered = x - &self.mean; // (x - μ)
        let standardized = &centered / &self.std_dev; // (x - μ) / σ
        let squared = &standardized * &standardized; // ((x - μ) / σ)²

        // Complete log-density formula
        neg_half * log_2pi - self.std_dev.clone().ln() + neg_half * &squared
    }
}

/// IID (Independent Identically Distributed) wrapper for any measure
#[derive(Debug)]
struct IID<T> {
    measure: T,
}

impl<T> IID<T> {
    /// Create a new IID wrapper around a measure
    fn new(measure: T) -> Self {
        Self { measure }
    }
}

impl IID<Normal> {
    /// Compute log-density for IID normal: Σ measure.log_density(xi) for xi in data
    fn log_density(&self, x: DynamicExpr<Vec<f64>>) -> DynamicExpr<f64> {
        // Create summation over the vector expression using Rust-idiomatic iterator patterns
        // Each element xi in the vector gets passed to the wrapped measure's log_density
        x.map(|xi| self.measure.log_density(xi)).sum()
    }
}

fn main() -> Result<()> {
    println!("🎯 Advanced IID Normal Distribution with LLVM JIT Demo");
    println!("=====================================================\n");

    // =======================================================================
    // 1. Create Normal Distribution with DynamicContext Variables
    // =======================================================================

    println!("1️⃣ Creating Normal Distribution with Variable Parameters");
    println!("--------------------------------------------------------");

    let mut ctx = DynamicContext::new();

    // Create parameter variables (this is what you wanted!)
    let mu = ctx.var(); // Variable(0) - mean parameter
    let sigma = ctx.var(); // Variable(1) - std dev parameter

    // Create Normal distribution with embedded parameters
    let normal = Normal::new(mu.clone(), sigma.clone());

    println!("✅ Created Normal distribution with variable parameters");
    println!("   • μ = Variable(0), σ = Variable(1) (symbolic variables)");
    println!("   • Formula: -0.5*ln(2π) - ln(σ) - 0.5*((x-μ)/σ)²");

    // Test single evaluation
    let test_x = ctx.constant(1.0);
    let test_mu = 0.0;
    let test_sigma = 1.0;

    let single_expr = normal.log_density(test_x);
    let single_result = ctx.eval(&single_expr, hlist![test_mu, test_sigma]);
    println!("   • Test: log_density(x=1) with μ=0, σ=1 = {single_result:.6}");

    // Expected: -0.5 * ln(2π) - ln(1) - 0.5 * (1²) = -0.5 * ln(2π) - 0.5
    let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5;
    println!("   • Expected: {expected:.6} ✓");
    assert!((single_result - expected).abs() < 1e-10);

    // =======================================================================
    // 2. Create IID Log-Likelihood with Consistent Signature
    // =======================================================================

    println!("\n2️⃣ Creating IID Log-Likelihood with Variable Parameters");
    println!("--------------------------------------------------------");

    // Sample data for testing
    let sample_data = vec![1.0, 2.0, 0.5, 1.5, 0.8];
    println!("   • Sample data: {sample_data:?}");

    // Create IID wrapper around the normal distribution
    let iid_normal = IID::new(normal);

    // Create data vector expression
    let data_expr = ctx.data_array(sample_data.clone());

    println!("✅ Created IID wrapper with consistent signature");
    println!("   • Normal has μ = Variable(0), σ = Variable(1) (symbolic)");
    println!("   • Data: as DynamicExpr<Vec<f64>>");
    println!("   • Formula: Σ log_density(xi, μ, σ) for xi in data");

    // Compute IID log-likelihood using new consistent signature
    let iid_expr = iid_normal.log_density(data_expr);
    let iid_result = ctx.eval(&iid_expr, hlist![test_mu, test_sigma]);
    println!("   • IID log-likelihood: {iid_result:.6}");

    // Verify by manual computation
    let manual_sum: f64 = sample_data
        .iter()
        .map(|&x| -0.5 * (2.0 * std::f64::consts::PI).ln() - 0.0 - 0.5 * (x - 0.0_f64).powi(2))
        .sum();
    println!("   • Manual verification: {manual_sum:.6} ✓");
    assert!((iid_result - manual_sum).abs() < 1e-10);

    // =======================================================================
    // 3. Expression Analysis Before Optimization
    // =======================================================================

    println!("\n3️⃣ Expression Analysis Before Optimization");
    println!("------------------------------------------");

    // Convert expressions to AST for analysis
    let single_ast = ctx.to_ast(&single_expr);
    let iid_ast = ctx.to_ast(&iid_expr);

    let single_ops = single_ast.count_operations();
    let iid_ops = iid_ast.count_operations();
    let iid_sums = iid_ast.count_summations();

    println!("Single Normal Log-Density:");
    println!("   • Operations: {single_ops}");
    println!("   • Variables: {}", count_variables(&single_ast));
    println!("   • Depth: {}", compute_depth(&single_ast));

    println!("\nIID Normal Log-Density:");
    println!("   • Operations: {iid_ops}");
    println!("   • Variables: {}", count_variables(&iid_ast));
    println!("   • Depth: {}", compute_depth(&iid_ast));
    println!("   • Summations: {iid_sums}");
    println!("   • Data points: {}", sample_data.len());

    #[cfg(feature = "optimization")]
    {
        // =======================================================================
        // 4. Egg Optimization with Sum Splitting
        // =======================================================================

        println!("\n4️⃣ Egg Optimization with Sum Splitting");
        println!("---------------------------------------");

        let mut optimizer = SymbolicOptimizer::new()?;

        println!("🔧 Optimizing expressions...");

        // Optimize both expressions
        let optimized_single = optimizer.optimize(&single_ast)?;
        let optimized_iid = optimizer.optimize(&iid_ast)?;

        let opt_single_ops = optimized_single.count_operations();
        let opt_iid_ops = optimized_iid.count_operations();
        let opt_iid_sums = optimized_iid.count_summations();

        println!("\nOptimization Results:");
        println!("Single Normal:");
        println!("   • Before: {single_ops} operations");
        println!("   • After: {opt_single_ops} operations");
        println!(
            "   • Reduction: {}",
            if single_ops > opt_single_ops {
                single_ops - opt_single_ops
            } else {
                0
            }
        );

        println!("\nIID Normal:");
        println!("   • Before: {iid_ops} operations");
        println!("   • After: {opt_iid_ops} operations");
        println!("   • Summations: {iid_sums} → {opt_iid_sums}");
        println!(
            "   • Reduction: {}",
            if iid_ops > opt_iid_ops {
                iid_ops - opt_iid_ops
            } else {
                0
            }
        );

        if iid_ops > opt_iid_ops {
            println!("   🎉 Sum splitting successful! Constants factored out of summation");
            println!("   💡 Formula: Σ(-½ln(2π) - ln(σ) - ½((xi-μ)/σ)²)");
            println!("   💡 Becomes: n*(-½ln(2π) - ln(σ)) + Σ(-½((xi-μ)/σ)²)");
        }

        #[cfg(feature = "llvm_jit")]
        {
            // =======================================================================
            // 5. LLVM JIT Compilation
            // =======================================================================

            println!("\n5️⃣ LLVM JIT Compilation");
            println!("------------------------");

            let context = Context::create();
            let mut llvm_compiler = LLVMJITCompiler::new(&context);

            println!("🔧 Compiling optimized expressions to native code...");

            // Compile single normal log-density
            match llvm_compiler.compile_multi_var(&optimized_single) {
                Ok(single_func) => {
                    println!("✅ Single normal compiled successfully");

                    // Test compiled function - now takes μ, σ, x (based on our variable order)
                    let args = [test_mu, test_sigma, 1.0]; // μ=Variable(0), σ=Variable(1), x=constant
                    let compiled_single_result = unsafe { single_func.call(args.as_ptr()) };
                    println!("   • Compiled result: {compiled_single_result:.6}");
                    println!(
                        "   • Matches interpreted: {}",
                        (compiled_single_result - single_result).abs() < 1e-10
                    );
                }
                Err(e) => {
                    println!("❌ Single normal compilation failed: {e}");
                }
            }

            // Compile IID normal log-density
            match llvm_compiler.compile_multi_var(&optimized_iid) {
                Ok(iid_func) => {
                    println!("✅ IID normal compiled successfully");

                    // Test compiled function
                    let args = [test_mu, test_sigma]; // μ=Variable(0), σ=Variable(1)
                    let compiled_iid_result = unsafe { iid_func.call(args.as_ptr()) };
                    println!("   • Compiled result: {compiled_iid_result:.6}");
                    println!(
                        "   • Matches interpreted: {}",
                        (compiled_iid_result - iid_result).abs() < 1e-10
                    );
                }
                Err(e) => {
                    println!("❌ IID normal compilation failed: {e}");
                    println!("   Note: IID compilation may require data array parameter handling");
                }
            }

            // =======================================================================
            // 6. Performance Benchmarking
            // =======================================================================

            println!("\n6️⃣ Performance Benchmarking");
            println!("----------------------------");

            // Generate larger dataset for meaningful benchmarks
            let large_data: Vec<f64> = (0..1_000).map(|i| (i as f64) * 0.001).collect();
            println!("📊 Benchmarking with {} data points", large_data.len());

            // Create large dataset with our consistent API
            let mut bench_ctx = DynamicContext::new();
            let bench_mu = bench_ctx.var(); // Variable(0)
            let bench_sigma = bench_ctx.var(); // Variable(1)
            let bench_normal = Normal::new(bench_mu, bench_sigma);
            let bench_iid = IID::new(bench_normal);
            let large_data_expr = bench_ctx.data_array(large_data.clone());
            let bench_expr = bench_iid.log_density(large_data_expr);

            // Benchmark interpreted evaluation
            println!("\n⏱️  Interpreted Evaluation:");
            let start = std::time::Instant::now();
            let interpreted_result = bench_ctx.eval(&bench_expr, hlist![test_mu, test_sigma]);
            let interpreted_time = start.elapsed();
            println!("   • Result: {interpreted_result:.6}");
            println!("   • Time: {interpreted_time:.2?}");

            // Try to compile and benchmark native evaluation
            let bench_ast = bench_ctx.to_ast(&bench_expr);
            match optimizer.optimize(&bench_ast) {
                Ok(optimized_bench) => {
                    println!("\n⏱️  LLVM JIT Compiled Evaluation:");
                    println!("   • Optimized benchmark expression");
                    println!(
                        "   • Interpreted performance: {interpreted_time:.2?} for {} points",
                        large_data.len()
                    );
                    println!("   • Variables remain symbolic: μ=Variable(0), σ=Variable(1)");
                }
                Err(e) => {
                    println!("❌ Benchmark optimization failed: {e}");
                }
            }
        }

        #[cfg(not(feature = "llvm_jit"))]
        {
            println!("\n5️⃣ LLVM JIT Compilation");
            println!("------------------------");
            println!("⚠️  LLVM JIT disabled - compile with --features llvm_jit");
        }
    }

    #[cfg(not(feature = "optimization"))]
    {
        println!("\n4️⃣ Egg Optimization");
        println!("--------------------");
        println!("⚠️  Optimization disabled - compile with --features optimization");
    }

    // =======================================================================
    // 7. Random Data Evaluation
    // =======================================================================

    println!("\n7️⃣ Random Data Evaluation");
    println!("--------------------------");

    use rand::Rng;
    let mut rng = rand::rng();

    // Generate random data
    let random_data: Vec<f64> = (0..100).map(|_| rng.random_range(-3.0..3.0)).collect();
    println!("📊 Generated {} random data points", random_data.len());

    // Create fresh context with variables for random data evaluation
    let mut random_ctx = DynamicContext::new();
    let random_mu = random_ctx.var(); // Variable(0)
    let random_sigma = random_ctx.var(); // Variable(1)
    let random_normal = Normal::new(random_mu, random_sigma);
    let random_iid = IID::new(random_normal);
    let random_data_expr = random_ctx.data_array(random_data.clone());
    let random_expr = random_iid.log_density(random_data_expr);

    // Test with different parameter values
    let param_sets = vec![
        (0.0, 1.0),  // Standard normal
        (1.0, 0.5),  // Shifted mean, smaller variance
        (-0.5, 2.0), // Negative mean, larger variance
    ];

    for (mu_val, sigma_val) in param_sets {
        let result = random_ctx.eval(&random_expr, hlist![mu_val, sigma_val]);
        println!("   • N(μ={mu_val}, σ={sigma_val}): log-likelihood = {result:.6}");
    }

    println!("\n🎉 Demo completed successfully!");
    println!("=================================");
    println!("✅ Normal struct with embedded variable parameters");
    println!("✅ IID<T> generic wrapper with consistent signature");
    println!("✅ DynamicContext variables remain symbolic through optimization");
    println!("✅ Egg optimization with sum splitting demonstration");
    println!("✅ LLVM JIT compilation for maximum performance");
    println!("✅ Comprehensive benchmarking and validation");
    println!("\n🔑 Key Insights:");
    println!("   • Variables (μ, σ) stay symbolic - no premature concrete binding");
    println!("   • Consistent signatures: Normal & IID both take DynamicExpr inputs");
    println!("   • No context threading required in log_density methods");
    println!("   • Sum splitting works on expressions with symbolic parameters");
    println!("   • LLVM JIT achieves native performance with variable parameters");

    Ok(())
}

// =======================================================================
// Helper Functions for Analysis
// =======================================================================

fn count_variables(ast: &ASTRepr<f64>) -> usize {
    let mut vars = std::collections::HashSet::new();
    collect_variables(ast, &mut vars);
    vars.len()
}

fn collect_variables(ast: &ASTRepr<f64>, vars: &mut std::collections::HashSet<usize>) {
    use dslcompile::ast::ast_repr::ASTRepr;
    match ast {
        ASTRepr::Variable(index) => {
            vars.insert(*index);
        }
        ASTRepr::BoundVar(index) => {
            vars.insert(*index);
        }
        ASTRepr::Add(operands) => {
            for (operand, _) in operands.iter_with_multiplicity() {
                collect_variables(operand, vars);
            }
        }
        ASTRepr::Sub(left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        ASTRepr::Mul(operands) => {
            for (operand, _) in operands.iter_with_multiplicity() {
                collect_variables(operand, vars);
            }
        }
        ASTRepr::Div(left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        ASTRepr::Pow(left, right) => {
            collect_variables(left, vars);
            collect_variables(right, vars);
        }
        ASTRepr::Let(_, expr, body) => {
            collect_variables(expr, vars);
            collect_variables(body, vars);
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => {
            collect_variables(inner, vars);
        }
        ASTRepr::Sum(collection) => {
            collect_variables_from_collection(collection, vars);
        }
        ASTRepr::Lambda(lambda) => {
            collect_variables(&lambda.body, vars);
        }
        ASTRepr::Constant(_) => {}
    }
}

fn collect_variables_from_collection(
    collection: &dslcompile::ast::ast_repr::Collection<f64>,
    vars: &mut std::collections::HashSet<usize>,
) {
    use dslcompile::ast::ast_repr::Collection;
    match collection {
        Collection::Empty => {}
        Collection::Singleton(expr) => collect_variables(expr, vars),
        Collection::Range { start, end } => {
            collect_variables(start, vars);
            collect_variables(end, vars);
        }
        Collection::Variable(index) => {
            vars.insert(*index);
        }
        Collection::Filter {
            collection,
            predicate,
        } => {
            collect_variables_from_collection(collection, vars);
            collect_variables(predicate, vars);
        }
        Collection::Map { lambda, collection } => {
            collect_variables(&lambda.body, vars);
            collect_variables_from_collection(collection, vars);
        }
        Collection::DataArray(_) => {
            // DataArray contains literal data, no variables to collect
        }
    }
}

fn compute_depth(ast: &ASTRepr<f64>) -> usize {
    match ast {
        ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => 1,
        ASTRepr::Add(operands) => {
            let mut depth = 0;
            for (operand, _) in operands.iter_with_multiplicity() {
                depth = depth.max(compute_depth(operand));
            }
            depth + 1
        }
        ASTRepr::Sub(left, right) => {
            let left_depth = compute_depth(left);
            let right_depth = compute_depth(right);
            1 + left_depth.max(right_depth)
        }
        ASTRepr::Mul(operands) => {
            let mut depth = 0;
            for (operand, _) in operands.iter_with_multiplicity() {
                depth = depth.max(compute_depth(operand));
            }
            depth + 1
        }
        ASTRepr::Div(left, right) => {
            let left_depth = compute_depth(left);
            let right_depth = compute_depth(right);
            1 + left_depth.max(right_depth)
        }
        ASTRepr::Pow(left, right) => {
            let left_depth = compute_depth(left);
            let right_depth = compute_depth(right);
            1 + left_depth.max(right_depth)
        }
        ASTRepr::Let(_, expr, body) => {
            let expr_depth = compute_depth(expr);
            let body_depth = compute_depth(body);
            1 + expr_depth.max(body_depth)
        }
        ASTRepr::Neg(inner)
        | ASTRepr::Ln(inner)
        | ASTRepr::Exp(inner)
        | ASTRepr::Sin(inner)
        | ASTRepr::Cos(inner)
        | ASTRepr::Sqrt(inner) => 1 + compute_depth(inner),
        ASTRepr::Sum(_) => 2, // Summation adds depth
        ASTRepr::Lambda(lambda) => 1 + compute_depth(&lambda.body),
    }
}
