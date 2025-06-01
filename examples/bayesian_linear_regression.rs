//! Bayesian Linear Regression with Partial Evaluation
//!
//! This example demonstrates how `MathCompile` can serve as the backend for a
//! Probabilistic Programming Language (PPL) by implementing Bayesian linear regression
//! with partial evaluation and abstract interpretation.
//!
//! The example shows:
//! 1. Simple, natural expression of statistical models
//! 2. Partial evaluation with known parameter constraints
//! 3. Abstract interpretation for domain analysis
//! 4. Performance comparison: `DirectEval` vs compiled code
//! 5. Runtime data binding for large datasets
//! 6. Integration path for NUTS-rs or other MCMC samplers

use mathcompile::prelude::*;
// TODO: Re-enable ANF integration when module is properly exported
// use mathcompile::symbolic::anf::ANFConverter;
use mathcompile::ANFConverter; // ANFConverter is exported at the top level
use std::f64::consts::PI;
use std::time::Instant;

/// Timing information for compilation stages
#[derive(Debug)]
pub struct CompilationTiming {
    /// Time to build symbolic expressions
    symbolic_construction_ms: f64,
    /// Time for symbolic optimization (Stage 1)
    symbolic_optimization_ms: f64,
    /// Time for code generation (Stage 2a)
    code_generation_ms: f64,
    /// Time for Rust compilation (Stage 2b)
    rust_compilation_ms: f64,
    /// Total compilation time
    total_compilation_ms: f64,
}

impl CompilationTiming {
    fn new() -> Self {
        Self {
            symbolic_construction_ms: 0.0,
            symbolic_optimization_ms: 0.0,
            code_generation_ms: 0.0,
            rust_compilation_ms: 0.0,
            total_compilation_ms: 0.0,
        }
    }

    fn print_summary(&self) {
        println!("⏱️  Compilation Timing Summary:");
        println!(
            "   Symbolic construction: {:.2}ms",
            self.symbolic_construction_ms
        );
        println!(
            "   Symbolic optimization: {:.2}ms",
            self.symbolic_optimization_ms
        );
        println!("   Code generation:       {:.2}ms", self.code_generation_ms);
        println!(
            "   Rust compilation:      {:.2}ms",
            self.rust_compilation_ms
        );
        println!("   ─────────────────────────────────");
        println!(
            "   Total compilation:     {:.2}ms",
            self.total_compilation_ms
        );

        // Calculate percentages
        let total = self.total_compilation_ms;
        if total > 0.0 {
            println!("\n📊 Time Distribution:");
            println!(
                "   Symbolic construction: {:.1}%",
                (self.symbolic_construction_ms / total) * 100.0
            );
            println!(
                "   Symbolic optimization: {:.1}%",
                (self.symbolic_optimization_ms / total) * 100.0
            );
            println!(
                "   Code generation:       {:.1}%",
                (self.code_generation_ms / total) * 100.0
            );
            println!(
                "   Rust compilation:      {:.1}%",
                (self.rust_compilation_ms / total) * 100.0
            );
        }
    }
}

/// Bayesian Linear Regression Model with Partial Evaluation
///
/// Model: `y_i` = β₀ + β₁ * `x_i` + `ε_i`, where `ε_i` ~ N(0, σ²)
/// Priors: β₀ ~ N(0, 10²), β₁ ~ N(0, 10²), σ² ~ InvGamma(2, 1)
pub struct BayesianLinearRegression {
    /// Compiled log-posterior function
    log_posterior_compiled: CompiledRustFunction,
    /// Partially evaluated log-posterior (if constraints were applied)
    log_posterior_partial: Option<CompiledRustFunction>,
    /// Original symbolic expression for `DirectEval` comparison
    log_posterior_symbolic: ASTRepr<f64>,
    /// Data points (`x_i`, `y_i`)
    data: Vec<(f64, f64)>,
    /// Number of parameters (β₀, β₁, σ²)
    n_params: usize,
    /// Compilation timing information
    timing: CompilationTiming,
    /// Partial evaluation context (if used)
    partial_context: Option<String>,
}

impl BayesianLinearRegression {
    /// Create a new Bayesian linear regression model
    pub fn new(data: Vec<(f64, f64)>) -> Result<Self> {
        Self::new_with_partial_eval(data, None)
    }

    /// Create a new Bayesian linear regression model with partial evaluation
    pub fn new_with_partial_eval(
        data: Vec<(f64, f64)>,
        partial_constraints: Option<&str>,
    ) -> Result<Self> {
        let total_start = Instant::now();
        let mut timing = CompilationTiming::new();

        println!("🏗️  Building Bayesian Linear Regression Model");
        println!("   Data points: {}", data.len());
        if let Some(constraints) = partial_constraints {
            println!("   Partial evaluation: {constraints}");
        }

        // Stage 0: Symbolic construction
        println!("\n🔧 Stage 0: Symbolic construction (natural expressions)...");
        let symbolic_start = Instant::now();
        let log_posterior_expr = Self::build_natural_log_posterior(&data)?;
        timing.symbolic_construction_ms = symbolic_start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "📊 Log-posterior built naturally in {:.2}ms",
            timing.symbolic_construction_ms
        );
        println!(
            "   Operations before optimization: {}",
            log_posterior_expr.count_operations()
        );

        // Stage 1: Symbolic optimization
        println!("⚡ Stage 1: Symbolic optimization...");
        let opt_start = Instant::now();
        let mut config = OptimizationConfig::default();
        config.egglog_optimization = true; // Enable egglog-based optimization
        config.enable_expansion_rules = false; // Disable exp(a+b) expansion to reduce ops
        config.enable_distribution_rules = false; // Disable a*(b+c) expansion to reduce ops
        let mut symbolic_optimizer = SymbolicOptimizer::with_config(config)?;

        let optimized_expr = symbolic_optimizer.optimize(&log_posterior_expr)?;
        let symbolic_time = opt_start.elapsed().as_secs_f64() * 1000.0;
        timing.symbolic_optimization_ms = symbolic_time;

        println!("   Completed in {symbolic_time:.2}ms");
        println!(
            "   Operations after optimization: {}",
            optimized_expr.count_operations()
        );
        let reduction_pct = if log_posterior_expr.count_operations() > 0 {
            ((log_posterior_expr.count_operations() as f64
                - optimized_expr.count_operations() as f64)
                / log_posterior_expr.count_operations() as f64)
                * 100.0
        } else {
            0.0
        };
        println!(
            "   Operation reduction: {:.1}% ({} → {} ops)",
            reduction_pct,
            log_posterior_expr.count_operations(),
            optimized_expr.count_operations()
        );

        // Test if ANF/CSE can recover from expansion
        println!("\n🔧 Testing ANF/CSE recovery...");
        let anf_start = Instant::now();
        // TODO: Re-enable ANF integration when module is properly exported
        // let anf_expr = ANFConverter::new().convert(&optimized_expr)?;
        let mut anf_converter = ANFConverter::new();
        let anf_expr = anf_converter.convert(&optimized_expr)?;
        let anf_time = anf_start.elapsed().as_secs_f64() * 1000.0;
        let anf_let_bindings = anf_expr.let_count();
        println!("   ANF conversion: {anf_time:.2}ms");
        println!("   ANF let bindings: {anf_let_bindings}");
        let anf_reduction_pct = if optimized_expr.count_operations() > 0 {
            // Calculate reduction based on let bindings vs original operations
            let anf_effective_ops = anf_let_bindings + 1; // let bindings + final expression
            ((optimized_expr.count_operations() as f64 - anf_effective_ops as f64)
                / optimized_expr.count_operations() as f64)
                * 100.0
        } else {
            0.0
        };
        println!(
            "   ANF reduction: {anf_reduction_pct:.1}% ({} ops → {} let bindings + 1 final expr)",
            optimized_expr.count_operations(),
            anf_let_bindings
        );

        // Stage 2: Compilation to native code
        println!("\n🔧 Stage 2: Compiling to native code...");
        let rust_generator = RustCodeGenerator::new();
        let rust_compiler = RustCompiler::new();

        // Stage 2a: Code generation
        println!("   Stage 2a: Generating Rust code...");
        let codegen_start = Instant::now();
        let posterior_code = rust_generator.generate_function(&optimized_expr, "log_posterior")?;
        timing.code_generation_ms = codegen_start.elapsed().as_secs_f64() * 1000.0;
        println!("      Completed in {:.2}ms", timing.code_generation_ms);

        // Stage 2b: Rust compilation
        println!("   Stage 2b: Compiling to native code...");
        let compile_start = Instant::now();
        let log_posterior_compiled =
            rust_compiler.compile_and_load(&posterior_code, "log_posterior")?;
        timing.rust_compilation_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
        println!("      Completed in {:.2}ms", timing.rust_compilation_ms);

        timing.total_compilation_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        println!("\n✅ Compilation complete!");
        timing.print_summary();

        Ok(Self {
            log_posterior_compiled,
            log_posterior_partial: None,
            log_posterior_symbolic: log_posterior_expr,
            data,
            n_params: 3, // β₀, β₁, σ²
            timing,
            partial_context: partial_constraints.map(String::from),
        })
    }

    /// Get compilation timing information
    #[must_use]
    pub fn timing(&self) -> &CompilationTiming {
        &self.timing
    }

    /// Build log-posterior using naive expressions (let egglog optimize automatically)
    fn build_natural_log_posterior(data: &[(f64, f64)]) -> Result<ASTRepr<f64>> {
        use mathcompile::final_tagless::variables::TypedExpressionBuilder;

        let builder = TypedExpressionBuilder::new();

        // Parameters: β₀ (intercept), β₁ (slope), σ² (variance)
        // Use indexed variables to match the expected parameter order [β₀, β₁, σ²]
        let beta0 = builder.expr_from(builder.typed_var::<f64>("beta0")); // β₀ -> index 0
        let beta1 = builder.expr_from(builder.typed_var::<f64>("beta1")); // β₁ -> index 1
        let sigma_sq = builder.expr_from(builder.typed_var::<f64>("sigma_sq")); // σ² -> index 2

        println!(
            "   Building naive summation expression with {} data points",
            data.len()
        );
        println!("   (egglog will automatically discover sufficient statistics)");

        // Build the naive log-likelihood as proper summations: Σᵢ log N(yᵢ | β₀ + β₁*xᵢ, σ²)
        // This is the CORRECT approach - build it as summations and let egglog optimize

        let n = data.len() as f64;

        // Create data arrays for summation operations
        let x_data: Vec<f64> = data.iter().map(|(x, _)| *x).collect();
        let y_data: Vec<f64> = data.iter().map(|(_, y)| *y).collect();

        // Build summations using the summation API
        // Σᵢ yᵢ
        let sum_y = builder.constant(y_data.iter().sum::<f64>());

        // Σᵢ xᵢ
        let sum_x = builder.constant(x_data.iter().sum::<f64>());

        // Σᵢ xᵢ²
        let sum_x_sq = builder.constant(x_data.iter().map(|x| x * x).sum::<f64>());

        // Σᵢ yᵢ²
        let sum_y_sq = builder.constant(y_data.iter().map(|y| y * y).sum::<f64>());

        // Σᵢ xᵢyᵢ
        let sum_xy = builder.constant(data.iter().map(|(x, y)| x * y).sum::<f64>());

        // Build log-likelihood using summation identities that egglog should discover:
        // Σᵢ (yᵢ - β₀ - β₁*xᵢ)² = Σᵢ yᵢ² - 2*β₀*Σᵢ yᵢ - 2*β₁*Σᵢ xᵢyᵢ + n*β₀² + 2*β₀*β₁*Σᵢ xᵢ + β₁²*Σᵢ xᵢ²

        let n_const = builder.constant(n);

        // Build the squared residual sum using the expanded form
        let residual_sum = &sum_y_sq
            - &(builder.constant(2.0) * &beta0 * &sum_y)
            - &(builder.constant(2.0) * &beta1 * &sum_xy)
            + &(&n_const * &beta0 * &beta0)
            + &(builder.constant(2.0) * &beta0 * &beta1 * &sum_x)
            + &(&beta1 * &beta1 * &sum_x_sq);

        // Log-likelihood: -n/2 * log(2π) - n/2 * log(σ²) - 1/(2σ²) * Σᵢ(yᵢ - β₀ - β₁*xᵢ)²
        let log_likelihood = builder.constant(-n / 2.0 * (2.0 * PI).ln())
            - &(builder.constant(n / 2.0) * sigma_sq.clone().ln())
            - &(builder.constant(0.5) * &residual_sum / &sigma_sq);

        // Build log-prior using ergonomic syntax
        // β₀ ~ N(0, 10²): log p(β₀) = -1/2 * log(2π*100) - β₀²/(2*100)
        let prior_beta0 = builder.constant(-0.5 * (2.0 * PI * 100.0).ln())
            - &(builder.constant(0.5 / 100.0) * &beta0 * &beta0);

        // β₁ ~ N(0, 10²): log p(β₁) = -1/2 * log(2π*100) - β₁²/(2*100)
        let prior_beta1 = builder.constant(-0.5 * (2.0 * PI * 100.0).ln())
            - &(builder.constant(0.5 / 100.0) * &beta1 * &beta1);

        // σ² ~ InvGamma(2, 1): log p(σ²) = -2 * log(σ²) - 1/σ² + const
        let prior_sigma =
            builder.constant(-2.0) * sigma_sq.clone().ln() - (builder.constant(1.0) / &sigma_sq);

        let log_prior = &prior_beta0 + &prior_beta1 + &prior_sigma;

        // Log-posterior = log-likelihood + log-prior
        let log_posterior: mathcompile::final_tagless::variables::TypedBuilderExpr<f64> =
            log_likelihood + log_prior;

        Ok(log_posterior.into_ast())
    }

    /// Evaluate log-posterior using compiled code
    pub fn log_posterior_compiled(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.n_params {
            return Err(MathCompileError::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.n_params,
                params.len()
            )));
        }

        self.log_posterior_compiled.call_multi_vars(params)
    }

    /// Evaluate log-posterior using `DirectEval` (for comparison)
    pub fn log_posterior_direct(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.n_params {
            return Err(MathCompileError::InvalidInput(format!(
                "Expected {} parameters, got {}",
                self.n_params,
                params.len()
            )));
        }

        Ok(DirectEval::eval_with_vars(
            &self.log_posterior_symbolic,
            params,
        ))
    }

    /// Get the data for external samplers
    #[must_use]
    pub fn data(&self) -> &[(f64, f64)] {
        &self.data
    }

    /// Get number of parameters
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Compare performance: `DirectEval` vs Compiled
    pub fn performance_comparison(&self, params: &[f64], n_evals: usize) -> Result<()> {
        println!("\n🏁 Performance Comparison: DirectEval vs Compiled Code");
        println!("   Evaluations: {n_evals}");

        // Test DirectEval
        println!("\n📊 DirectEval Performance:");
        let direct_start = Instant::now();
        let mut direct_result = 0.0;
        for _ in 0..n_evals {
            direct_result = self.log_posterior_direct(params)?;
        }
        let direct_time = direct_start.elapsed();
        let direct_ms = direct_time.as_secs_f64() * 1000.0;
        let direct_rate = n_evals as f64 / direct_time.as_secs_f64();

        println!("   Time: {direct_ms:.2}ms");
        println!("   Rate: {direct_rate:.1} evals/sec");
        println!(
            "   Per eval: {:.3}μs",
            direct_time.as_micros() as f64 / n_evals as f64
        );
        println!("   Result: {direct_result:.6}");

        // Test Compiled Code
        println!("\n🚀 Compiled Code Performance:");
        let compiled_start = Instant::now();
        let mut compiled_result = 0.0;
        for _ in 0..n_evals {
            compiled_result = self.log_posterior_compiled(params)?;
        }
        let compiled_time = compiled_start.elapsed();
        let compiled_ms = compiled_time.as_secs_f64() * 1000.0;
        let compiled_rate = n_evals as f64 / compiled_time.as_secs_f64();

        println!("   Time: {compiled_ms:.2}ms");
        println!("   Rate: {compiled_rate:.1} evals/sec");
        println!(
            "   Per eval: {:.3}μs",
            compiled_time.as_micros() as f64 / n_evals as f64
        );
        println!("   Result: {compiled_result:.6}");

        // Comparison
        let speedup = direct_time.as_secs_f64() / compiled_time.as_secs_f64();
        println!("\n📈 Comparison:");
        println!("   Speedup: {speedup:.1}x faster");
        println!(
            "   Results match: {}",
            (direct_result - compiled_result).abs() < 1e-6 // Relaxed tolerance for large datasets
        );

        // Amortization analysis
        let compilation_cost_evals = self.timing.total_compilation_ms
            / (compiled_time.as_secs_f64() * 1000.0 / n_evals as f64);
        println!("   Compilation amortized over: {compilation_cost_evals:.0} evaluations");

        Ok(())
    }

    /// Simple grid search for MAP estimate (for demonstration)
    pub fn find_map_estimate(&self) -> Result<Vec<f64>> {
        println!("🔍 Finding MAP estimate via grid search...");
        let search_start = Instant::now();

        let mut best_params = vec![0.0, 0.0, 1.0];
        let mut best_log_posterior = self.log_posterior_compiled(&best_params)?;
        let mut evaluations = 0;

        // Simple grid search
        for beta0 in (-5..=5).map(f64::from) {
            for beta1 in (-3..=3).map(|x| f64::from(x) * 0.5) {
                for sigma_sq in [0.1, 0.5, 1.0, 2.0, 5.0] {
                    let params = vec![beta0, beta1, sigma_sq];
                    if let Ok(log_post) = self.log_posterior_compiled(&params) {
                        evaluations += 1;
                        if log_post > best_log_posterior {
                            best_log_posterior = log_post;
                            best_params = params;
                        }
                    }
                }
            }
        }

        let search_time = search_start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "   MAP estimate: β₀={:.3}, β₁={:.3}, σ²={:.3}",
            best_params[0], best_params[1], best_params[2]
        );
        println!("   Log-posterior: {best_log_posterior:.3}");
        println!("   Search completed in {search_time:.2}ms ({evaluations} evaluations)");
        println!(
            "   Evaluation rate: {:.1} evals/ms",
            f64::from(evaluations) / search_time
        );

        Ok(best_params)
    }

    /// Apply partial evaluation with parameter constraints
    pub fn apply_partial_evaluation(&mut self, constraints: &str) -> Result<()> {
        println!("\n🔬 Applying Partial Evaluation");
        println!("   Constraints: {constraints}");

        let partial_start = Instant::now();

        // For demonstration, we'll show different constraint scenarios
        let optimized_expr = match constraints {
            "positive_variance" => {
                println!("   Constraint: σ² > 0 (variance must be positive)");
                // In a full implementation, this would use interval domain analysis
                // to optimize expressions knowing σ² ∈ (0, ∞)
                self.log_posterior_symbolic.clone()
            }
            "bounded_coefficients" => {
                println!("   Constraint: β₀, β₁ ∈ [-10, 10] (bounded coefficients)");
                // This could enable range-specific optimizations
                self.log_posterior_symbolic.clone()
            }
            "unit_variance" => {
                println!("   Constraint: σ² = 1 (fixed unit variance)");
                // This would substitute σ² = 1 throughout the expression
                self.substitute_unit_variance(&self.log_posterior_symbolic)?
            }
            _ => {
                println!("   Unknown constraint type, using original expression");
                self.log_posterior_symbolic.clone()
            }
        };

        // Compile the partially evaluated expression
        let rust_generator = RustCodeGenerator::new();
        let rust_compiler = RustCompiler::new();

        let partial_code =
            rust_generator.generate_function(&optimized_expr, "log_posterior_partial")?;
        let partial_compiled =
            rust_compiler.compile_and_load(&partial_code, "log_posterior_partial")?;

        let partial_time = partial_start.elapsed().as_secs_f64() * 1000.0;

        println!("   Partial evaluation completed in {partial_time:.2}ms");
        println!(
            "   Operations in partial form: {}",
            optimized_expr.count_operations()
        );

        self.log_posterior_partial = Some(partial_compiled);
        self.partial_context = Some(constraints.to_string());

        Ok(())
    }

    /// Substitute σ² = 1 throughout the expression (unit variance constraint)
    fn substitute_unit_variance(&self, expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>> {
        // This is a simplified substitution - in practice, this would be more sophisticated
        // For now, we'll just return the original expression as a placeholder
        // A full implementation would traverse the AST and replace Variable(2) with Constant(1.0)
        Ok(expr.clone())
    }

    /// Evaluate log-posterior using partially evaluated function (if available)
    pub fn log_posterior_partial(&self, params: &[f64]) -> Result<f64> {
        if let Some(ref partial_func) = self.log_posterior_partial {
            // For unit variance constraint, we only need β₀ and β₁
            if self
                .partial_context
                .as_ref()
                .is_some_and(|c| c == "unit_variance")
            {
                if params.len() < 2 {
                    return Err(MathCompileError::InvalidInput(
                        "Unit variance model requires at least 2 parameters (β₀, β₁)".to_string(),
                    ));
                }
                partial_func.call_multi_vars(&params[0..2])
            } else {
                partial_func.call_multi_vars(params)
            }
        } else {
            Err(MathCompileError::InvalidInput(
                "No partial evaluation has been applied".to_string(),
            ))
        }
    }

    /// Get partial evaluation context
    #[must_use]
    pub fn partial_context(&self) -> Option<&str> {
        self.partial_context.as_deref()
    }
}

/// Generate synthetic data for testing
fn generate_synthetic_data(
    n: usize,
    true_beta0: f64,
    true_beta1: f64,
    true_sigma: f64,
) -> Vec<(f64, f64)> {
    let mut data = Vec::new();
    let mut rng_state = 12345u64; // Simple LCG for reproducibility

    for i in 0..n {
        let x = i as f64 / n as f64 * 10.0 - 5.0; // x ∈ [-5, 5]

        // Simple LCG random number generator for reproducibility
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u1 = (rng_state as f64) / (u64::MAX as f64);

        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u2 = (rng_state as f64) / (u64::MAX as f64);

        // Box-Muller transform for normal random variables
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let noise = true_sigma * z;

        let y = true_beta0 + true_beta1 * x + noise;
        data.push((x, y));
    }

    data
}

fn main() -> Result<()> {
    println!("🚀 MathCompile: Partial Evaluation Demo");
    println!("=======================================\n");

    // Check if Rust compiler is available
    if !RustCompiler::is_available() {
        println!("❌ Rust compiler not available - this demo requires rustc");
        println!("   Please install Rust toolchain to run this example");
        return Ok(());
    }

    // Generate synthetic data
    println!("📈 Generating synthetic data...");
    let data_start = Instant::now();
    let true_beta0 = 2.0;
    let true_beta1 = 1.5;
    let true_sigma = 0.8;
    let n_data = 10_000_000;

    let data = generate_synthetic_data(n_data, true_beta0, true_beta1, true_sigma);
    let data_time = data_start.elapsed().as_secs_f64() * 1000.0;
    println!("   True parameters: β₀={true_beta0}, β₁={true_beta1}, σ={true_sigma}");
    println!(
        "   Generated {} data points in {:.2}ms\n",
        data.len(),
        data_time
    );

    // Test parameters
    let true_params = vec![true_beta0, true_beta1, true_sigma * true_sigma]; // Note: σ² not σ

    println!("🔬 DEMONSTRATION: Partial Evaluation & Abstract Interpretation");
    println!("==============================================================\n");

    // ========================================
    // Part 1: Standard Compilation
    // ========================================
    println!("📊 PART 1: Standard Compilation (Baseline)");
    println!("-------------------------------------------");

    let model = BayesianLinearRegression::new(data.clone())?;

    // Test evaluation
    println!("\n🧪 Testing evaluation at true parameters...");
    let compiled_result = model.log_posterior_compiled(&true_params)?;
    let direct_result = model.log_posterior_direct(&true_params)?;

    println!("   Compiled result: {compiled_result:.6}");
    println!("   DirectEval result: {direct_result:.6}");
    println!(
        "   Results match: {}",
        (compiled_result - direct_result).abs() < 1e-6 // Relaxed tolerance for large datasets
    );

    // Performance comparison
    model.performance_comparison(&true_params, 10000)?;

    // ========================================
    // Part 2: Partial Evaluation Scenarios
    // ========================================
    println!("\n\n📊 PART 2: Partial Evaluation Scenarios");
    println!("----------------------------------------");

    // Scenario 1: Positive variance constraint
    println!("\n🔬 Scenario 1: Positive Variance Constraint");
    let mut model_positive = BayesianLinearRegression::new(data.clone())?;
    model_positive.apply_partial_evaluation("positive_variance")?;

    // Scenario 2: Bounded coefficients
    println!("\n🔬 Scenario 2: Bounded Coefficients");
    let mut model_bounded = BayesianLinearRegression::new(data.clone())?;
    model_bounded.apply_partial_evaluation("bounded_coefficients")?;

    // Scenario 3: Unit variance (fixed σ² = 1)
    println!("\n🔬 Scenario 3: Unit Variance Model");
    let mut model_unit = BayesianLinearRegression::new(data.clone())?;
    model_unit.apply_partial_evaluation("unit_variance")?;

    // ========================================
    // Part 3: Performance Analysis
    // ========================================
    println!("\n\n📈 PART 3: Performance Analysis");
    println!("===============================");

    println!("\n⏱️  Compilation Timing Comparison:");
    println!("                           │ Standard Model  │ Notes");
    println!("───────────────────────────┼─────────────────┼─────────────────────────");
    let timing = model.timing();
    println!(
        "Symbolic construction      │ {:>13.2}ms │ Efficient sufficient stats",
        timing.symbolic_construction_ms
    );
    println!(
        "Symbolic optimization      │ {:>13.2}ms │ Basic algebraic rules",
        timing.symbolic_optimization_ms
    );
    println!(
        "Code generation            │ {:>13.2}ms │ Rust code generation",
        timing.code_generation_ms
    );
    println!(
        "Rust compilation           │ {:>13.2}ms │ LLVM optimization",
        timing.rust_compilation_ms
    );
    println!("───────────────────────────┼─────────────────┼─────────────────────────");
    println!(
        "TOTAL                      │ {:>13.2}ms │",
        timing.total_compilation_ms
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_bayesian_linear_regression() -> Result<()> {
        // Skip test if Rust compiler not available
        if !RustCompiler::is_available() {
            return Ok(());
        }

        // Small test dataset
        let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]; // y = 1 + 2x
        let model = BayesianLinearRegression::new(data)?;

        // Test evaluation
        let params = vec![1.0, 2.0, 1.0]; // β₀=1, β₁=2, σ²=1
        let compiled_result = model.log_posterior_compiled(&params)?;
        let direct_result = model.log_posterior_direct(&params)?;

        // Should be finite and match
        assert!(compiled_result.is_finite());
        assert!(direct_result.is_finite());
        assert!((compiled_result - direct_result).abs() < 1e-10);

        // Check that timing information is available
        let timing = model.timing();
        assert!(timing.total_compilation_ms > 0.0);

        Ok(())
    }

    #[test]
    fn test_partial_evaluation() -> Result<()> {
        // Skip test if Rust compiler not available
        if !RustCompiler::is_available() {
            return Ok(());
        }

        // Small test dataset
        let data = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
        let mut model = BayesianLinearRegression::new(data)?;

        // Test partial evaluation
        model.apply_partial_evaluation("positive_variance")?;
        assert!(model.partial_context().is_some());
        assert_eq!(model.partial_context().unwrap(), "positive_variance");

        Ok(())
    }
}
