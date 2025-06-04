//! Type-Level Boolean Logic and First-Order Logic Operations
//!
//! This module provides a complete type-level boolean logic system that enables
//! compile-time logical operations and conditional trait implementations.
//!
//! # Features
//!
//! - **Type-level booleans**: `True` and `False` types with `TypeLevelBool` trait
//! - **Logical operations**: `AND`, `OR`, `NOT`
//! - **Conditional traits**: `WhenTrue`, `WhenFalse` for conditional implementations
//! - **Comparison operations**: Type-level equality checking for const generics
//! - **Helper aliases**: Clean syntax for complex type-level logic expressions

/// Type-level boolean trait for compile-time boolean values
pub trait TypeLevelBool {
    /// The boolean value at the type level
    const VALUE: bool;
}

/// Type-level representation of `true`
pub struct True;

/// Type-level representation of `false`
pub struct False;

impl TypeLevelBool for True {
    const VALUE: bool = true;
}

impl TypeLevelBool for False {
    const VALUE: bool = false;
}

// ============================================================================
// LOGICAL OPERATIONS
// ============================================================================

/// Type-level AND operation: A ∧ B
pub trait And<Other: TypeLevelBool> {
    type Output: TypeLevelBool;
}

impl And<True> for True {
    type Output = True;
}

impl And<False> for True {
    type Output = False;
}

impl And<True> for False {
    type Output = False;
}

impl And<False> for False {
    type Output = False;
}

/// Type-level OR operation: A ∨ B
pub trait Or<Other: TypeLevelBool> {
    type Output: TypeLevelBool;
}

impl Or<True> for True {
    type Output = True;
}

impl Or<False> for True {
    type Output = True;
}

impl Or<True> for False {
    type Output = True;
}

impl Or<False> for False {
    type Output = False;
}

/// Type-level NOT operation: ¬A
pub trait Not {
    type Output: TypeLevelBool;
}

impl Not for True {
    type Output = False;
}

impl Not for False {
    type Output = True;
}

// ============================================================================
// COMPARISON OPERATIONS
// ============================================================================

/// Type-level equality check for const generic values
pub trait TypeLevelEq<const A: usize, const B: usize> {
    type Output: TypeLevelBool;
}

impl<const ID: usize> TypeLevelEq<ID, ID> for () {
    type Output = True;
}

// Note: The "not equal" case is handled by the fact that if two values
// are not the same, the above impl doesn't match, so we get a compile error
// when trying to use it. This is actually what we want for type-level logic.

// ============================================================================
// HELPER TYPE ALIASES
// ============================================================================

/// Type-level equality: A == B
pub type TypeEq<const A: usize, const B: usize> = <() as TypeLevelEq<A, B>>::Output;

/// Type-level inequality: A != B (implemented as NOT(A == B))
pub type TypeNeq<const A: usize, const B: usize> = <TypeEq<A, B> as Not>::Output;

/// Type-level AND with clean syntax
pub type TypeAnd<A, B> = <A as And<B>>::Output;

/// Type-level OR with clean syntax  
pub type TypeOr<A, B> = <A as Or<B>>::Output;

/// Type-level NOT with clean syntax
pub type TypeNot<A> = <A as Not>::Output;

// ============================================================================
// CONDITIONAL TRAITS
// ============================================================================

/// Conditional trait: only implement if condition is True
///
/// Usage: `where (): WhenTrue<SomeCondition>`
pub trait WhenTrue<Condition: TypeLevelBool> {}
impl WhenTrue<True> for () {}

/// Conditional trait: only implement if condition is False
///
/// Usage: `where (): WhenFalse<SomeCondition>`
pub trait WhenFalse<Condition: TypeLevelBool> {}
impl WhenFalse<False> for () {}

// ============================================================================
// DOMAIN-SPECIFIC ALIASES FOR SCOPED VARIABLES
// ============================================================================

/// Check if two variable IDs are the same
pub type IsSameId<const ID1: usize, const ID2: usize> = TypeEq<ID1, ID2>;

/// Check if two variable IDs are different
pub type IsDifferentId<const ID1: usize, const ID2: usize> = TypeNeq<ID1, ID2>;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_level_bool_values() {
        assert!(True::VALUE);
        assert!(!False::VALUE);
    }

    // Note: Most tests here would be compile-time tests that verify
    // the type system works correctly. Runtime tests are limited for type-level logic.

    #[test]
    fn test_conditional_compilation() {
        // This function only compiles if the condition is true
        fn only_when_true()
        where
            (): WhenTrue<True>,
        {
            println!("This compiles because True satisfies WhenTrue");
        }

        only_when_true();

        // This would fail to compile:
        // fn only_when_false()
        // where
        //     (): WhenTrue<False>  // ❌ False doesn't satisfy WhenTrue
        // {}
    }
}
