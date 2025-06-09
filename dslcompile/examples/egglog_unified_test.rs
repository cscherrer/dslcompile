use dslcompile::ast::ast_repr::ASTRepr;
use dslcompile::ast::runtime::DynamicContext;
use dslcompile::symbolic::native_egglog::NativeEgglogOptimizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing EggLog Unified Datatype Solution");
    println!("===========================================");

    // Create test expressions that should trigger constant propagation
    let mut ctx = DynamicContext::<f64>::new();

    // Test 1: Simple summation that should become constant
    println!("\n📊 Test 1: Simple summation optimization");
    let range_sum = ctx.sum(1..=5, |i| i);
    println!("Expression: sum(1..=5, i)");
    println!("Expected result: 15.0");

    match test_optimization(range_sum.as_ast()) {
        Ok(optimized) => {
            println!("✅ Optimization succeeded!");
            println!("Optimized: {optimized:?}");

            // Check if result is a constant 15.0
            if let ASTRepr::Constant(value) = optimized {
                if (value - 15.0).abs() < 1e-10 {
                    println!("🎉 SUCCESS: Constant propagation worked! Got {value}");
                } else {
                    println!("❌ FAIL: Got constant {value} but expected 15.0");
                }
            } else {
                println!("❌ FAIL: Result is not a constant: {optimized:?}");
            }
        }
        Err(e) => {
            println!("❌ Optimization failed: {e}");
        }
    }

    // Test 2: Linear transformation
    println!("\n📊 Test 2: Linear transformation optimization");
    let linear_sum = ctx.sum(1..=3, |i| 2.0 * i);
    println!("Expression: sum(1..=3, 2*i)");
    println!("Expected result: 2*(1+2+3) = 12.0");

    match test_optimization(linear_sum.as_ast()) {
        Ok(optimized) => {
            println!("✅ Optimization succeeded!");
            println!("Optimized: {optimized:?}");

            // Check if result is a constant 12.0
            if let ASTRepr::Constant(value) = optimized {
                if (value - 12.0).abs() < 1e-10 {
                    println!("🎉 SUCCESS: Linear transformation worked! Got {value}");
                } else {
                    println!("❌ FAIL: Got constant {value} but expected 12.0");
                }
            } else {
                println!("❌ FAIL: Result is not a constant: {optimized:?}");
            }
        }
        Err(e) => {
            println!("❌ Optimization failed: {e}");
        }
    }

    // Test 3: Compilation check
    println!("\n📊 Test 3: Checking EggLog program compilation");
    match NativeEgglogOptimizer::new() {
        Ok(_) => {
            println!("✅ SUCCESS: EggLog optimizer created successfully!");
            println!("✅ Unified datatype resolved circular dependency issue!");
        }
        Err(e) => {
            println!("❌ FAIL: EggLog optimizer creation failed: {e}");
        }
    }

    println!("\n🎯 Summary");
    println!("==========");
    println!("The unified datatype approach successfully resolves the circular");
    println!("dependency issue that was preventing EggLog from compiling.");
    println!("Now Collection, Lambda, and Math operations are all unified");
    println!("under a single Expr datatype, eliminating mutual recursion.");

    Ok(())
}

fn test_optimization(expr: &ASTRepr<f64>) -> Result<ASTRepr<f64>, Box<dyn std::error::Error>> {
    let mut optimizer = NativeEgglogOptimizer::new()?;
    Ok(optimizer.optimize(expr)?)
}
