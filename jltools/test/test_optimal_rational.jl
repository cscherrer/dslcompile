using Test
using DSLCompile

@testset "Optimal Rational Approximation Tests" begin
    
    @testset "Basic functionality" begin
        # Test with a simple function that should work well
        f(x) = x^2 + x + 1  # Simple polynomial
        result = find_optimal_rational(f, (0.0, 1.0), 1e-6)
        
        # Should find some reasonable approximation
        @test result.error <= 1e-6
        @test result.total_degree >= 0
        @test length(result.N) == result.degree_n + 1
        @test length(result.D) == result.degree_d + 1
        @test result.D[1] ≈ 1.0  # First coefficient of denominator should be 1
        
        # Test evaluation
        test_x = 0.5
        expected = f(test_x)
        actual = evaluate_rational(test_x, result.N, result.D)
        @test abs(expected - actual) < 1e-6
    end
    
    @testset "Exponential function" begin
        f(x) = exp(x)
        result = find_optimal_rational(f, (0.0, 1.0), 1e-6)
        
        # Should find some reasonable approximation
        @test result.error <= 1e-6
        @test result.total_degree > 0
        @test length(result.N) == result.degree_n + 1
        @test length(result.D) == result.degree_d + 1
        @test result.D[1] ≈ 1.0  # First coefficient of denominator should be 1
        
        # Test that the approximation is actually good
        test_error = test_approximation_error(f, result, (0.0, 1.0))
        @test test_error.max_error <= 1e-6 * 1.1  # Allow small numerical tolerance
    end
    
    @testset "Custom weight function" begin
        f(x) = sin(x)
        weight_fn(x, y) = BigFloat(2)  # Constant weight of 2
        
        result = find_optimal_rational(f, (0.0, 1.0), 1e-4, w=weight_fn)
        
        # Should still find a valid approximation
        @test result.error <= 1e-4
        @test result.total_degree >= 0
    end
    
    @testset "Error handling" begin
        f(x) = exp(x)
        
        # Test with impossible tolerance and low max_degree
        @test_throws ErrorException find_optimal_rational(f, (0.0, 1.0), 1e-20, max_degree=1)
    end
    
    @testset "Rational function evaluation" begin
        # Test evaluation with known coefficients
        N = [1.0, 2.0, 1.0]  # 1 + 2x + x^2
        D = [1.0, 1.0]       # 1 + x
        
        x = 2.0
        expected = (1 + 2*x + x^2) / (1 + x)  # (1 + 4 + 4) / (1 + 2) = 9/3 = 3
        actual = evaluate_rational(x, N, D)
        @test actual ≈ expected
    end
    
    @testset "Degree progression" begin
        # Test that higher tolerance requirements lead to higher degrees
        f(x) = exp(x)
        
        result_loose = find_optimal_rational(f, (0.0, 1.0), 1e-2)
        result_tight = find_optimal_rational(f, (0.0, 1.0), 1e-8)
        
        # Tighter tolerance should require higher degree (or at least not lower)
        @test result_tight.total_degree >= result_loose.total_degree
    end
end 