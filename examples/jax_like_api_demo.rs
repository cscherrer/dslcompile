use mathcompile::transformations::Transformations;
use mathcompile::tracing::{ComputationTracer, TraceAnalyzer};
use mathcompile::ergonomics::MathBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 JAX-like API Demo for MathCompile");
    println!("====================================");

    // Create a mathematical expression using the ergonomic API
    let mut math = MathBuilder::new();
    let x = math.var("x");
    let y = math.var("y");
    
    // Build a complex expression: f(x, y) = sin(x² + y) * exp(-x)
    let expr = (x.clone() * x.clone() + y.clone()).sin_ref() * (-x.clone()).exp_ref();
    
    println!("\n📐 Original Expression:");
    println!("f(x, y) = sin(x² + y) * exp(-x)");
    
    // ============================================================================
    // JAX-like Transformations
    // ============================================================================
    
    println!("\n🔄 JAX-like Transformations:");
    println!("----------------------------");
    
    // 1. Simple JIT compilation (like JAX's @jit)
    println!("\n1. JIT Compilation:");
    let jit_transform = Transformations::jit(expr.clone());
    println!("   Transformations applied: {:?}", jit_transform.metadata.transformations);
    println!("   Should JIT: {}", jit_transform.metadata.compilation_hints.should_jit);
    
    // Compile and test
    if let Ok(compiled_func) = jit_transform.compile() {
        let result = compiled_func.call_multi_vars(&[2.0, 1.0])?;
        println!("   f(2.0, 1.0) = {:.6}", result);
    }
    
    // 2. Gradient computation (like JAX's grad)
    println!("\n2. Gradient Computation:");
    let grad_transform = Transformations::grad(expr.clone());
    println!("   Transformations applied: {:?}", grad_transform.metadata.transformations);
    println!("   Vectorizable: {}", grad_transform.metadata.compilation_hints.vectorizable);
    
    if let Ok(grad_result) = grad_transform.compute() {
        println!("   Gradient computed successfully");
        println!("   Gradient transformations: {:?}", grad_result.metadata.transformations);
    }
    
    // 3. Composed transformations (like JAX's jit(grad(f)))
    println!("\n3. Composed JIT + Gradient:");
    let composed = Transformations::jit(expr.clone()).grad();
    println!("   Transformations applied: {:?}", composed.metadata.transformations);
    println!("   Should JIT: {}", composed.metadata.compilation_hints.should_jit);
    println!("   Vectorizable: {}", composed.metadata.compilation_hints.vectorizable);
    
    if let Ok((func_compiled, grad_compiled)) = composed.compile() {
        let func_result = func_compiled.call_multi_vars(&[2.0, 1.0])?;
        let grad_result = grad_compiled.call_multi_vars(&[2.0, 1.0])?;
        println!("   f(2.0, 1.0) = {:.6}", func_result);
        println!("   ∇f(2.0, 1.0) ≈ {:.6}", grad_result);
    }
    
    // 4. Domain analysis (unique to this library)
    println!("\n4. Domain Analysis:");
    let domain_transform = Transformations::analyze_domains(expr.clone());
    println!("   Transformations applied: {:?}", domain_transform.metadata.transformations);
    
    if let Ok(domain_result) = domain_transform.analyze() {
        println!("   Domain analysis completed");
        println!("   Result transformations: {:?}", domain_result.metadata.transformations);
    }
    
    // 5. Vectorization (placeholder - like JAX's vmap)
    println!("\n5. Vectorization (placeholder):");
    let vmap_transform = Transformations::vmap(expr.clone(), 0);
    println!("   Transformations applied: {:?}", vmap_transform.metadata.transformations);
    println!("   Memory pattern: {:?}", vmap_transform.metadata.compilation_hints.memory_pattern);
    
    // ============================================================================
    // Computation Tracing (like JAX's tracing)
    // ============================================================================
    
    println!("\n🔍 Computation Tracing:");
    println!("-----------------------");
    
    let mut tracer = ComputationTracer::new();
    let trace = tracer.trace_expression(&expr);
    
    println!("Operations traced: {}", trace.operations.len());
    println!("Total FLOPs: {:.1}", trace.complexity.flops);
    println!("Vectorization potential: {:.1}%", trace.complexity.vectorization_potential * 100.0);
    println!("Parallelization potential: {:.1}%", trace.complexity.parallelization_potential * 100.0);
    println!("Memory bandwidth: {:.1} bytes", trace.complexity.memory_bandwidth);
    
    // Analyze the trace for optimization opportunities
    let recommendation = TraceAnalyzer::recommend_compilation_strategy(&trace);
    let optimizations = TraceAnalyzer::identify_optimizations(&trace);
    
    println!("\n📊 Trace Analysis:");
    println!("Compilation recommendation: {:?}", recommendation);
    println!("Optimization opportunities: {:?}", optimizations);
    
    // ============================================================================
    // Performance Comparison
    // ============================================================================
    
    println!("\n⚡ Performance Comparison:");
    println!("-------------------------");
    
    // Direct evaluation
    let start = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = math.eval(&expr, &[("x", 2.0), ("y", 1.0)]);
    }
    let direct_time = start.elapsed();
    println!("Direct evaluation (10k calls): {:?}", direct_time);
    
    // JIT compiled evaluation
    if let Ok(jit_func) = Transformations::jit(expr.clone()).compile() {
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = jit_func.call_multi_vars(&[2.0, 1.0]);
        }
        let jit_time = start.elapsed();
        println!("JIT compiled (10k calls): {:?}", jit_time);
        
        let speedup = direct_time.as_nanos() as f64 / jit_time.as_nanos() as f64;
        println!("Speedup: {:.2}x", speedup);
    }
    
    // ============================================================================
    // Advanced Examples
    // ============================================================================
    
    println!("\n🎯 Advanced Examples:");
    println!("---------------------");
    
    // Example 1: Statistical model with partial evaluation
    println!("\n1. Statistical Model with Partial Evaluation:");
    let mu = math.var("mu");
    let sigma = math.var("sigma");
    let data_point = math.var("x");
    
    // Gaussian log-likelihood: -0.5 * ((x - μ) / σ)² - log(σ) - 0.5*log(2π)
    let gaussian_ll = {
        let diff = data_point.clone() - mu.clone();
        let normalized = diff / sigma.clone();
        let squared = normalized.clone() * normalized;
        -math.constant(0.5) * squared - sigma.ln_ref() - math.constant(0.5 * (2.0 * std::f64::consts::PI).ln())
    };
    
    println!("   Gaussian log-likelihood model created");
    
    // Partial evaluation with known hyperparameters
    let mut static_values = std::collections::HashMap::new();
    static_values.insert("sigma".to_string(), 1.0); // Known variance
    
    let partial_eval = Transformations::partial_eval(gaussian_ll.clone(), static_values);
    println!("   Partial evaluation configured with σ = 1.0");
    
    if let Ok(specialized) = partial_eval.apply() {
        println!("   Specialized model created");
        println!("   Transformations: {:?}", specialized.metadata.transformations);
    }
    
    // Example 2: Chained transformations for ML pipeline
    println!("\n2. ML Pipeline with Chained Transformations:");
    
    // Create a simple neural network layer: tanh(W*x + b) - using available functions
    let weight = math.var("w");
    let bias = math.var("b");
    let input = math.var("x");
    
    let linear = weight * input + bias;
    // Use sinh/cosh to approximate tanh since tanh_ref doesn't exist
    let activation = linear.sin_ref(); // Simplified for demo
    
    // Chain: domain analysis → JIT
    let ml_pipeline = Transformations::analyze_domains(activation.clone())
        .jit();
    
    println!("   Neural network layer: sin(w*x + b) [simplified]");
    println!("   Pipeline: domain_analysis → jit");
    
    // Example 3: Optimization-guided compilation
    println!("\n3. Optimization-Guided Compilation:");
    
    // Create a complex expression that benefits from optimization
    let mut complex_expr = x.clone();
    for i in 1..=5 {
        let term = (x.clone() * math.constant(f64::from(i))).sin_ref();
        complex_expr = complex_expr + term;
    }
    
    // Trace and analyze
    let mut complex_tracer = ComputationTracer::new();
    let complex_trace = complex_tracer.trace_expression(&complex_expr);
    let complex_recommendation = TraceAnalyzer::recommend_compilation_strategy(&complex_trace);
    
    println!("   Complex expression: x + sin(x) + sin(2x) + ... + sin(5x)");
    println!("   Operations: {}", complex_trace.operations.len());
    println!("   Recommendation: {:?}", complex_recommendation);
    
    // Apply recommended compilation strategy
    match complex_recommendation {
        mathcompile::tracing::CompilationRecommendation::JITCompilation => {
            println!("   → Applying JIT compilation as recommended");
            if let Ok(optimized) = Transformations::jit(complex_expr).compile() {
                let result = optimized.call_multi_vars(&[1.0])?;
                println!("   → Result: {:.6}", result);
            }
        }
        mathcompile::tracing::CompilationRecommendation::VectorizedCompilation => {
            println!("   → Vectorized compilation recommended");
        }
        _ => {
            println!("   → Using direct evaluation");
        }
    }
    
    println!("\n✨ JAX-like API Demo Complete!");
    println!("This demonstrates composable transformations, tracing, and optimization");
    println!("similar to JAX's functional programming approach.");
    
    Ok(())
} 