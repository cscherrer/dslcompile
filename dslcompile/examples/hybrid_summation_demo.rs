//! # Hybrid Summation Demo
//!
//! This example demonstrates both index-based and iterator-based summation approaches to illustrate the hybrid design.

use dslcompile::Result;
use dslcompile::final_tagless::{ExpressionBuilder, IntRange};
use dslcompile::symbolic::summation::SummationProcessor;

/// Iterator-based summation extension
pub trait IteratorSummation {
    fn sum_over<I, F>(&self, data: I, f: F) -> Result<f64>
    where
        I: IntoIterator<Item = f64>,
        F: Fn(f64) -> f64;
        
    fn sum_over_with_factor<I>(&self, data: I, factor: f64) -> Result<f64>
    where
        I: IntoIterator<Item = f64>;
}

impl IteratorSummation for SummationProcessor {
    fn sum_over<I, F>(&self, data: I, f: F) -> Result<f64>
    where
        I: IntoIterator<Item = f64>,
        F: Fn(f64) -> f64,
    {
        Ok(data.into_iter().map(f).sum())
    }
    
    fn sum_over_with_factor<I>(&self, data: I, factor: f64) -> Result<f64>
    where
        I: IntoIterator<Item = f64>,
    {
        Ok(factor * data.into_iter().sum::<f64>())
    }
}

fn main() -> Result<()> {
    println!("🔄 Hybrid Summation Demo");
    println!("========================\n");

    let mut processor = SummationProcessor::new()?;
    
    // Demo 1: Index-based mathematical summation
    println!("📊 Demo 1: Index-Based Mathematical Summation");
    println!("Use case: Pure mathematical expressions with indices");
    println!("Expression: Σ(i=1 to 5) (i² + 2i + 1)");
    
    let range = IntRange::new(1, 5);
    let result = processor.sum(range, |i| {
        let _math = ExpressionBuilder::new();
        // Mathematical expression: i² + 2i + 1
        i.clone() * &i + 2.0 * &i + 1.0
    })?;
    
    println!("Pattern recognized: {:?}", result.pattern);
    let value = result.evaluate(&[])?;
    println!("Result: {value}");
    
    // Manual verification: (1²+2·1+1) + (2²+2·2+1) + ... + (5²+2·5+1)
    //                    = 4 + 9 + 16 + 25 + 36 = 90
    println!("Expected: 90 (verified manually)\n");
    
    // Demo 2: Iterator-based data processing
    println!("📊 Demo 2: Iterator-Based Data Processing");
    println!("Use case: Processing runtime data arrays");
    
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let factor = 2.5;
    
    println!("Data: {:?}", data);
    println!("Processing: sum(k * x for x in data) where k = {factor}");
    
    // Method 1: Direct factor extraction
    let result1 = processor.sum_over_with_factor(data.iter().copied(), factor)?;
    println!("Method 1 (factor extraction): {result1}");
    
    // Method 2: General function application
    let result2 = processor.sum_over(data.iter().copied(), |x| factor * x)?;
    println!("Method 2 (function mapping): {result2}");
    
    // Expected: 2.5 * (1+2+3+4+5) = 2.5 * 15 = 37.5
    println!("Expected: 37.5\n");
    
    // Demo 3: When you need both approaches
    println!("📊 Demo 3: Combining Both Approaches");
    println!("Use case: Mathematical expressions with runtime data");
    
    let weights = vec![0.1, 0.2, 0.3, 0.2, 0.2];
    let observations = vec![10.0, 15.0, 12.0, 18.0, 14.0];
    
    println!("Weights: {:?}", weights);
    println!("Observations: {:?}", observations);
    println!("Computing: Σ(wᵢ * xᵢ) - weighted sum");
    
    // Iterator-based approach for this use case
    let weighted_sum = weights.iter()
        .zip(observations.iter())
        .map(|(&w, &x)| w * x)
        .sum::<f64>();
    
    println!("Weighted sum: {weighted_sum}");
    
    // Alternative: Process in chunks if you need mathematical analysis
    let chunk_size = 2;
    let mut total = 0.0;
    
    for (chunk_idx, chunk) in observations.chunks(chunk_size).enumerate() {
        let weight_chunk = &weights[chunk_idx * chunk_size..];
        
        // Use mathematical summation for each chunk if needed
        let chunk_contribution = chunk.iter()
            .zip(weight_chunk.iter())
            .map(|(&x, &w)| w * x)
            .sum::<f64>();
            
        total += chunk_contribution;
        
        println!("Chunk {}: {:?} -> contribution: {chunk_contribution}", 
                chunk_idx + 1, chunk);
    }
    
    println!("Total via chunked processing: {total}\n");
    
    // Demo 4: Performance comparison
    println!("📊 Demo 4: Performance Characteristics");
    println!("Comparing approaches for different use cases");
    
    let large_data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    
    let start = std::time::Instant::now();
    let iter_result = processor.sum_over(large_data.iter().copied(), |x| x * x)?;
    let iter_time = start.elapsed();
    
    println!("Iterator approach (1000 elements):");
    println!("  Time: {:?}", iter_time);
    println!("  Result: {iter_result}");
    
    // Mathematical approach for comparison
    let start = std::time::Instant::now();
    let range = IntRange::new(1, 1000);
    let math_result = processor.sum(range, |i| {
        let _math = ExpressionBuilder::new();
        i.clone() * &i
    })?;
    let math_value = math_result.evaluate(&[])?;
    let math_time = start.elapsed();
    
    println!("Mathematical approach (1000 terms):");
    println!("  Time: {:?}", math_time);
    println!("  Pattern: {:?}", math_result.pattern);
    println!("  Result: {math_value}");
    println!("  Optimized: {}", math_result.is_optimized);
    
    println!("\n✅ Demo Complete!");
    println!("\n🎯 Key Takeaways:");
    println!("• Use index-based summation for pure mathematical expressions");
    println!("• Use iterator-based summation for runtime data processing");
    println!("• Mathematical approach can optimize to closed forms");
    println!("• Iterator approach is more direct for data manipulation");
    println!("• Choose based on whether you need symbolic optimization");

    Ok(())
} 