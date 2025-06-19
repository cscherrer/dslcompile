//! Property-based tests for interval domain analysis
//!
//! Tests the interval domain representation to ensure mathematical correctness,
//! proper interval arithmetic, and domain propagation properties.

use dslcompile::{
    ast::ast_repr::ASTRepr,
    interval_domain::{IntervalDomain, Endpoint},
    prelude::*,
};
use proptest::prelude::*;
use std::collections::HashMap;

/// Generator for interval endpoints
fn endpoint_strategy() -> BoxedStrategy<Endpoint<f64>> {
    prop_oneof![
        (-100.0..100.0).prop_map(Endpoint::Open),
        (-100.0..100.0).prop_map(Endpoint::Closed),
        Just(Endpoint::Unbounded),
    ].boxed()
}

/// Generator for interval domains  
fn interval_domain_strategy() -> BoxedStrategy<IntervalDomain<f64>> {
    prop_oneof![
        Just(IntervalDomain::Bottom),
        Just(IntervalDomain::Top),
        (-100.0..100.0).prop_map(IntervalDomain::Constant),
        (endpoint_strategy(), endpoint_strategy()).prop_map(|(lower, upper)| {
            IntervalDomain::new_interval(lower, upper)
        }),
    ].boxed()
}

/// Generator for valid intervals (lower <= upper)
fn valid_interval_strategy() -> BoxedStrategy<IntervalDomain<f64>> {
    prop_oneof![
        Just(IntervalDomain::Bottom),
        Just(IntervalDomain::Top),
        (-50.0..50.0).prop_map(IntervalDomain::Constant),
        (-50.0..0.0).prop_flat_map(|lower| {
            (0.0..50.0).prop_map(move |upper| {
                IntervalDomain::new_interval(
                    Endpoint::Closed(lower),
                    Endpoint::Closed(upper)
                )
            })
        }),
    ].boxed()
}

proptest! {
    /// Test that interval domain operations preserve mathematical properties
    #[test] 
    fn prop_interval_contains_consistency(
        domain in valid_interval_strategy(),
        test_value in -100.0..100.0f64
    ) {
        // Contains should be consistent with mathematical definition
        let contains = domain.contains(test_value);
        
        match &domain {
            IntervalDomain::Bottom => {
                prop_assert!(!contains, "Bottom domain should contain no values");
            }
            IntervalDomain::Top => {
                prop_assert!(contains, "Top domain should contain all values");
            }
            IntervalDomain::Constant(c) => {
                let expected = (test_value - c).abs() < 1e-15;
                prop_assert_eq!(contains, expected, "Constant domain contains only the constant");
            }
            IntervalDomain::Interval { lower, upper } => {
                // Test lower bound
                let satisfies_lower = match lower {
                    Endpoint::Open(l) => test_value > *l,
                    Endpoint::Closed(l) => test_value >= *l,
                    Endpoint::Unbounded => true,
                };
                
                // Test upper bound
                let satisfies_upper = match upper {
                    Endpoint::Open(u) => test_value < *u,
                    Endpoint::Closed(u) => test_value <= *u,
                    Endpoint::Unbounded => true,
                };
                
                let expected = satisfies_lower && satisfies_upper;
                prop_assert_eq!(contains, expected, 
                             "Interval contains should match endpoint constraints");
            }
        }
    }

    /// Test interval meet (intersection) properties
    #[test]
    fn prop_interval_meet_properties(
        domain1 in interval_domain_strategy(),
        domain2 in interval_domain_strategy()
    ) {
        let meet = domain1.meet(&domain2);
        
        // Meet should be commutative
        let meet_rev = domain2.meet(&domain1);
        prop_assert_eq!(meet, meet_rev, "Meet should be commutative");
        
        // Meet with self should be identity
        let self_meet = domain1.meet(&domain1);
        prop_assert_eq!(self_meet, domain1.clone(), "Meet with self should be identity");
        
        // Meet with bottom should be bottom
        let bottom_meet = domain1.meet(&IntervalDomain::Bottom);
        prop_assert_eq!(bottom_meet, IntervalDomain::Bottom, 
                       "Meet with bottom should be bottom");
        
        // Meet with top should be the original domain
        let top_meet = domain1.meet(&IntervalDomain::Top);
        prop_assert_eq!(top_meet, domain1.clone(), 
                       "Meet with top should be original domain");
    }

    /// Test interval join (union) properties
    #[test]
    fn prop_interval_join_properties(
        domain1 in interval_domain_strategy(),
        domain2 in interval_domain_strategy()
    ) {
        let join = domain1.join(&domain2);
        
        // Join should be commutative
        let join_rev = domain2.join(&domain1);
        prop_assert_eq!(join, join_rev, "Join should be commutative");
        
        // Join with self should be identity
        let self_join = domain1.join(&domain1);
        prop_assert_eq!(self_join, domain1.clone(), "Join with self should be identity");
        
        // Join with bottom should be the original domain
        let bottom_join = domain1.join(&IntervalDomain::Bottom);
        prop_assert_eq!(bottom_join, domain1.clone(), 
                       "Join with bottom should be original domain");
        
        // Join with top should be top
        let top_join = domain1.join(&IntervalDomain::Top);
        prop_assert_eq!(top_join, IntervalDomain::Top, "Join with top should be top");
    }

    /// Test specific interval construction patterns
    #[test]
    fn prop_interval_construction_patterns(zero_val in -1.0..1.0f64) {
        // Test positive domain
        let positive = IntervalDomain::positive(zero_val);
        prop_assert!(!positive.contains(zero_val), "Positive domain excludes zero (open)");
        if zero_val < 100.0 {
            prop_assert!(positive.contains((zero_val + 1.0)), "Positive domain includes values > zero");
        }
        
        // Test non-negative domain
        let non_negative = IntervalDomain::non_negative(zero_val);
        prop_assert!(non_negative.contains(zero_val), "Non-negative domain includes zero (closed)");
        if zero_val < 100.0 {
            prop_assert!(non_negative.contains((zero_val + 1.0)), "Non-negative domain includes values >= zero");
        }
        
        // Test unit interval [0, 1]
        let unit = IntervalDomain::closed_interval(0.0, 1.0);
        prop_assert!(unit.contains(0.0), "Unit interval contains 0");
        prop_assert!(unit.contains(1.0), "Unit interval contains 1");
        prop_assert!(unit.contains(0.5), "Unit interval contains 0.5");
        prop_assert!(!unit.contains(-0.1), "Unit interval excludes negative values");
        prop_assert!(!unit.contains(1.1), "Unit interval excludes values > 1");
    }

    /// Test basic domain properties
    #[test]
    fn prop_domain_basic_properties(
        domain in interval_domain_strategy(),
        test_value in -50.0..50.0f64
    ) {
        // Test that domain operations are well-defined
        let contains = domain.contains(test_value);
        
        // Test is_positive and is_non_negative consistency
        if domain.is_positive(0.0) {
            prop_assert!(!domain.contains(0.0), "Positive domain excludes zero");
            prop_assert!(!domain.contains(-1.0), "Positive domain excludes negative");
        }
        
        if domain.is_non_negative(0.0) {
            prop_assert!(!domain.contains(-1.0), "Non-negative domain excludes negative");
        }
        
        // Test contains_zero consistency
        let contains_zero = domain.contains_zero(0.0);
        prop_assert_eq!(contains_zero, domain.contains(0.0), "contains_zero should match contains(0)");
    }

    /// Test endpoint basic functionality
    #[test]
    fn prop_endpoint_basic(
        val1 in -100.0..100.0f64
    ) {
        let open = Endpoint::Open(val1);
        let closed = Endpoint::Closed(val1);
        let unbounded = Endpoint::<f64>::Unbounded;
        
        // Test that endpoints can be created and compared for equality
        prop_assert_eq!(open.clone(), Endpoint::Open(val1));
        prop_assert_eq!(closed.clone(), Endpoint::Closed(val1));
        prop_assert_eq!(unbounded, Endpoint::Unbounded);
        
        // Test that different endpoint types with same value are different
        prop_assert_ne!(open, closed);
    }
}

/// Unit tests for specific interval domain functionality
#[cfg(test)]
mod interval_domain_unit_tests {
    use super::*;

    #[test]
    fn test_basic_interval_construction() {
        // Test constant domain
        let const_domain = IntervalDomain::Constant(5.0);
        assert!(const_domain.contains(5.0));
        assert!(!const_domain.contains(4.9));
        assert!(!const_domain.contains(5.1));
        
        // Test bottom domain
        let bottom = IntervalDomain::Bottom;
        assert!(!bottom.contains(0.0));
        assert!(!bottom.contains(f64::INFINITY));
        assert!(!bottom.contains(f64::NEG_INFINITY));
        
        // Test top domain
        let top = IntervalDomain::Top;
        assert!(top.contains(0.0));
        assert!(top.contains(1000.0));
        assert!(top.contains(-1000.0));
    }

    #[test]
    fn test_interval_endpoint_semantics() {
        // Test open interval (1, 3)
        let open_interval = IntervalDomain::Interval {
            lower: Endpoint::Open(1.0),
            upper: Endpoint::Open(3.0),
        };
        
        assert!(!open_interval.contains(1.0)); // Endpoint excluded
        assert!(open_interval.contains(2.0));  // Interior included
        assert!(!open_interval.contains(3.0)); // Endpoint excluded
        
        // Test closed interval [1, 3]
        let closed_interval = IntervalDomain::Interval {
            lower: Endpoint::Closed(1.0),
            upper: Endpoint::Closed(3.0),
        };
        
        assert!(closed_interval.contains(1.0)); // Endpoint included
        assert!(closed_interval.contains(2.0)); // Interior included
        assert!(closed_interval.contains(3.0)); // Endpoint included
        
        // Test half-open interval [1, 3)
        let half_open = IntervalDomain::Interval {
            lower: Endpoint::Closed(1.0),
            upper: Endpoint::Open(3.0),
        };
        
        assert!(half_open.contains(1.0));  // Lower endpoint included
        assert!(half_open.contains(2.0));  // Interior included
        assert!(!half_open.contains(3.0)); // Upper endpoint excluded
    }

    #[test]
    fn test_interval_meet_edge_cases() {
        let interval1 = IntervalDomain::Interval {
            lower: Endpoint::Closed(1.0),
            upper: Endpoint::Closed(3.0),
        };
        
        let interval2 = IntervalDomain::Interval {
            lower: Endpoint::Closed(2.0),
            upper: Endpoint::Closed(4.0),
        };
        
        let meet = interval1.meet(&interval2);
        
        // Should be [2, 3]
        assert!(meet.contains(2.0));
        assert!(meet.contains(3.0));
        assert!(!meet.contains(1.5));
        assert!(!meet.contains(3.5));
    }

    #[test]
    fn test_interval_join_non_overlapping() {
        let interval1 = IntervalDomain::Interval {
            lower: Endpoint::Closed(1.0),
            upper: Endpoint::Closed(2.0),
        };
        
        let interval2 = IntervalDomain::Interval {
            lower: Endpoint::Closed(3.0),
            upper: Endpoint::Closed(4.0),
        };
        
        let join = interval1.join(&interval2);
        
        // Join of non-overlapping intervals should create larger interval [1, 4]
        assert!(join.contains(1.0));
        assert!(join.contains(2.0));
        assert!(join.contains(2.5)); // Gap gets included in convex hull
        assert!(join.contains(3.0));
        assert!(join.contains(4.0));
    }

    #[test]
    fn test_special_domain_constructors() {
        // Test positive domain
        let positive = IntervalDomain::positive(0.0);
        assert!(!positive.contains(0.0));   // Open at zero
        assert!(!positive.contains(-1.0));  // Excludes negative
        assert!(positive.contains(1.0));    // Includes positive
        
        // Test non-negative domain
        let non_negative = IntervalDomain::non_negative(0.0);
        assert!(non_negative.contains(0.0));  // Closed at zero
        assert!(!non_negative.contains(-1.0)); // Excludes negative
        assert!(non_negative.contains(1.0));   // Includes positive
        
        // Test unit interval [0, 1]
        let unit = IntervalDomain::closed_interval(0.0, 1.0);
        assert!(unit.contains(0.0));
        assert!(unit.contains(0.5));
        assert!(unit.contains(1.0));
        assert!(!unit.contains(-0.1));
        assert!(!unit.contains(1.1));
    }

    #[test]
    fn test_domain_basic_operations() {
        let interval1 = IntervalDomain::Interval {
            lower: Endpoint::Closed(1.0),
            upper: Endpoint::Closed(3.0),
        };
        
        let interval2 = IntervalDomain::Interval {
            lower: Endpoint::Closed(2.0),
            upper: Endpoint::Closed(4.0),
        };
        
        // Test meet operation
        let meet = interval1.meet(&interval2);
        assert!(meet.contains(2.0));
        assert!(meet.contains(3.0));
        assert!(!meet.contains(1.5));
        assert!(!meet.contains(3.5));
        
        // Test join operation
        let join = interval1.join(&interval2);
        assert!(join.contains(1.0));
        assert!(join.contains(4.0));
        assert!(join.contains(2.5));
    }

    #[test]
    fn test_domain_properties() {
        let positive = IntervalDomain::positive(0.0);
        let non_negative = IntervalDomain::non_negative(0.0);
        let negative = IntervalDomain::negative(0.0);
        let non_positive = IntervalDomain::non_positive(0.0);
        
        // Test positivity properties
        assert!(positive.is_positive(0.0));
        assert!(positive.is_non_negative(0.0)); // Positive numbers are also non-negative
        assert!(!positive.contains(0.0));
        assert!(positive.contains(1.0));
        
        assert!(non_negative.is_non_negative(0.0));
        assert!(non_negative.contains(0.0));
        assert!(non_negative.contains(1.0));
        assert!(!non_negative.contains(-1.0));
        
        // Test negative domains
        assert!(!negative.contains(0.0));
        assert!(!negative.contains(1.0));
        assert!(negative.contains(-1.0));
        
        assert!(non_positive.contains(0.0));
        assert!(!non_positive.contains(1.0));
        assert!(non_positive.contains(-1.0));
    }

    #[test]
    fn test_constant_domain_operations() {
        let const5 = IntervalDomain::Constant(5.0);
        let const3 = IntervalDomain::Constant(3.0);
        
        // Test contains
        assert!(const5.contains(5.0));
        assert!(!const5.contains(4.9));
        assert!(!const5.contains(5.1));
        
        // Test meet with constant
        let meet = const5.meet(&const3);
        assert_eq!(meet, IntervalDomain::Bottom); // Different constants
        
        let self_meet = const5.meet(&const5);
        assert_eq!(self_meet, const5); // Same constant
        
        // Test join with constant
        let join = const5.join(&const3);
        // Should create interval [3, 5]
        assert!(join.contains(3.0));
        assert!(join.contains(4.0));
        assert!(join.contains(5.0));
    }
}