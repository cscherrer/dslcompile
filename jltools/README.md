# DSLCompile.jl

A Julia package for finding optimal rational function approximations with minimal computational cost.

## Overview

DSLCompile provides functionality to find the fastest (lowest degree) rational function approximation that satisfies a given error tolerance. This is particularly useful for applications where computational efficiency is critical, as lower degree rational functions require fewer operations to evaluate.

## Key Features

- **Optimal Degree Selection**: Automatically finds the lowest total degree (numerator + denominator) rational function that meets your error tolerance
- **Custom Error Weighting**: Support for custom error-weighting functions to emphasize accuracy in specific regions
- **Comprehensive Testing**: Built-in functions to verify approximation quality
- **BigFloat Precision**: Uses high-precision arithmetic for accurate coefficient computation

## Installation

```julia
using Pkg
Pkg.add(url="path/to/DSLCompile")
```

## Quick Start

```julia
using DSLCompile

# Approximate exp(x) on [0, 1] with tolerance 1e-6
f(x) = exp(x)
result = find_optimal_rational(f, (0.0, 1.0), 1e-6)

println("Optimal degrees: ($(result.degree_n), $(result.degree_d))")
println("Total degree: $(result.total_degree)")
println("Achieved error: $(result.error)")

# Evaluate the approximation at a point
x = 0.5
approx_value = evaluate_rational(x, result.N, result.D)
actual_value = f(x)
println("Approximation error at x=0.5: $(abs(actual_value - approx_value))")
```

## Main Functions

### `find_optimal_rational(f, interval, tolerance; w=nothing, max_degree=20)`

Finds the fastest rational function approximation meeting the error tolerance.

**Arguments:**
- `f`: Function to approximate (maps BigFloat → BigFloat)
- `interval`: Tuple of interval endpoints
- `tolerance`: Maximum allowed error
- `w`: Optional error-weighting function (default: uniform weighting)
- `max_degree`: Maximum total degree to search (default: 20)

**Returns:** Named tuple with:
- `N`: Numerator coefficients
- `D`: Denominator coefficients  
- `error`: Achieved maximum weighted error
- `degree_n`, `degree_d`: Degrees of numerator and denominator
- `total_degree`: Total degree (n + d)
- `alternation_points`: Chebyshev alternation points

### `evaluate_rational(x, N, D)`

Evaluates a rational function with given coefficients.

### `test_approximation_error(f, result, interval; num_points=1000)`

Tests the actual approximation error across the interval.

## Examples

### Basic Usage
```julia
# Approximate sin(x) on [0, π/2]
f(x) = sin(x)
result = find_optimal_rational(f, (0.0, π/2), 1e-8)
```

### Custom Error Weighting
```julia
# Emphasize accuracy near endpoints
f(x) = 1/(1 + x^2)
weight_fn(x, y) = 1 + abs(x)  # Higher weight near x = ±1
result = find_optimal_rational(f, (-1.0, 1.0), 1e-6, w=weight_fn)
```

### Tolerance Comparison
```julia
tolerances = [1e-3, 1e-6, 1e-9]
for tol in tolerances
    result = find_optimal_rational(exp, (0.0, 1.0), tol)
    println("Tolerance: $tol, Total degree: $(result.total_degree)")
end
```

## Algorithm

The algorithm works by:

1. **Systematic Search**: Iterates through all combinations of numerator and denominator degrees in order of increasing total degree
2. **Early Termination**: Stops as soon as a valid approximation is found, ensuring minimal computational cost
3. **Robust Error Handling**: Gracefully handles cases where certain degree combinations fail
4. **Chebyshev Optimality**: Uses the Remez exchange algorithm to ensure optimal approximations

## Performance Considerations

- Lower degree rational functions are always preferred for speed
- The search is optimized to find the minimal degree solution first
- BigFloat arithmetic ensures high precision but may be slower than Float64 for some applications
- Consider the trade-off between approximation accuracy and evaluation speed for your specific use case

## Dependencies

- [Remez.jl](https://github.com/simonbyrnes/Remez.jl): For computing minimax rational approximations

## Testing

Run the test suite with:
```julia
using Pkg
Pkg.test("DSLCompile")
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

[Add your license information here] 