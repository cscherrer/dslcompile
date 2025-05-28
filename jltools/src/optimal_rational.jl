using Remez

"""
    find_optimal_rational(f, interval, tolerance; w=nothing, max_degree=20)

Find the fastest (lowest degree) rational function approximation that satisfies the error tolerance.

# Arguments
- `f`: The function to be approximated. Maps BigFloat -> BigFloat.
- `interval`: A tuple giving the endpoints of the interval (in either order) on which to approximate f.
- `tolerance`: Maximum allowed error tolerance (BigFloat or Float64).
- `w`: Optional error-weighting function. Takes two BigFloat arguments x,y and returns a scaling factor.
- `max_degree`: Maximum total degree (n+d) to search up to. Default is 20.

# Returns
A named tuple with fields:
- `N`: Coefficients of the numerator polynomial
- `D`: Coefficients of the denominator polynomial  
- `error`: The actual maximum weighted error achieved
- `degree_n`: Degree of numerator
- `degree_d`: Degree of denominator
- `total_degree`: Total degree (n+d)
- `alternation_points`: Points where the error alternates sign

# Example
```julia
f(x) = exp(x)
result = find_optimal_rational(f, (0.0, 1.0), 1e-6)
println("Optimal approximation: degree (\$(result.degree_n), \$(result.degree_d))")
```
"""
function find_optimal_rational(f, interval, tolerance; w=nothing, max_degree=20)
    # Convert tolerance to BigFloat for consistency
    tol = BigFloat(tolerance)
    
    # Default weight function (always returns 1)
    weight_fn = w === nothing ? (x, y) -> BigFloat(1) : w
    
    best_result = nothing
    best_total_degree = Inf
    
    # Search through all combinations of numerator and denominator degrees
    # Start with low degrees and work up
    for total_deg in 0:max_degree
        best_result_this_degree = nothing
        best_error_this_degree = Inf
        
        # Try all combinations for this total degree
        for n in 0:total_deg
            d = total_deg - n
            
            try
                # Attempt to find rational approximation with degrees (n, d)
                N, D, E, X = ratfn_minimax(f, interval, n, d, weight_fn)
                
                # Check if this approximation meets our tolerance
                if E <= tol
                    # This is a valid approximation - check if it's the best for this degree
                    if E < best_error_this_degree
                        best_result_this_degree = (
                            N = N,
                            D = D,
                            error = E,
                            degree_n = n,
                            degree_d = d,
                            total_degree = total_deg,
                            alternation_points = X
                        )
                        best_error_this_degree = E
                    end
                else 
                    @info "($n, $d): $(Float64(E))"
                end
            catch e
                # ratfn_minimax might fail for some degree combinations
                # (e.g., if the problem is ill-conditioned), so we continue
                @info "($n, $d): $e"
                continue
            end
        end
        
        # If we found a valid approximation with this total degree, we're done
        # since we're searching in order of increasing total degree
        if best_result_this_degree !== nothing
            best_result = best_result_this_degree
            break
        end
    end
    
    if best_result === nothing
        error("Could not find a rational approximation meeting tolerance $tolerance within maximum degree $max_degree")
    end
    
    return best_result
end

"""
    evaluate_rational(x, N, D)

Evaluate a rational function with numerator coefficients N and denominator coefficients D at point x.

# Arguments
- `x`: Point at which to evaluate the function
- `N`: Coefficients of numerator polynomial (constant term first)
- `D`: Coefficients of denominator polynomial (constant term first)

# Returns
The value of the rational function N(x)/D(x)
"""
function evaluate_rational(x, N, D)
    # Evaluate numerator
    num = N[1]  # constant term
    x_power = x
    for i in 2:length(N)
        num += N[i] * x_power
        x_power *= x
    end
    
    # Evaluate denominator  
    den = D[1]  # constant term (should be 1)
    x_power = x
    for i in 2:length(D)
        den += D[i] * x_power
        x_power *= x
    end
    
    return num / den
end

"""
    test_approximation_error(f, result, interval; num_points=1000)

Test the actual error of a rational approximation across the interval.

# Arguments
- `f`: Original function
- `result`: Result from find_optimal_rational
- `interval`: Interval to test over
- `num_points`: Number of test points

# Returns
A named tuple with maximum absolute error and the point where it occurs
"""
function test_approximation_error(f, result, interval; num_points=1000)
    a, b = interval
    points = range(BigFloat(a), BigFloat(b), length=num_points)
    
    max_error = BigFloat(0)
    max_error_point = BigFloat(a)
    
    for x in points
        actual = f(x)
        approx = evaluate_rational(x, result.N, result.D)
        error = abs(actual - approx)
        
        if error > max_error
            max_error = error
            max_error_point = x
        end
    end
    
    return (max_error = max_error, max_error_point = max_error_point)
end 