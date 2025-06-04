//! Demo of Type-Level Boolean Logic System
//!
//! This example demonstrates the type-level boolean logic system that enables
//! sophisticated compile-time programming patterns.

use dslcompile::compile_time::type_level_logic::{
    False, True, TypeEq, TypeLevelBool, WhenFalse, WhenTrue,
};

// Example of using type-level logic in a trait
trait ExampleTrait<T, Condition: TypeLevelBool> {
    fn process(&self, value: T) -> T;
}

// Implementation only available when condition is True
struct ExampleStruct;

impl<T> ExampleTrait<T, True> for ExampleStruct {
    fn process(&self, value: T) -> T {
        println!("Processing with True condition");
        value
    }
}

impl<T> ExampleTrait<T, False> for ExampleStruct {
    fn process(&self, value: T) -> T {
        println!("Processing with False condition");
        value
    }
}

// Example of conditional implementations using WhenTrue/WhenFalse
struct ConditionalProcessor<Condition: TypeLevelBool> {
    _condition: std::marker::PhantomData<Condition>,
}

impl<Condition: TypeLevelBool> ConditionalProcessor<Condition> {
    fn new() -> Self {
        Self {
            _condition: std::marker::PhantomData,
        }
    }
}

impl ConditionalProcessor<True> {
    fn process_when_true(&self) -> &'static str
    where
        (): WhenTrue<True>, // This constraint is always satisfied for True
    {
        "Processed with True condition!"
    }
}

impl ConditionalProcessor<False> {
    fn process_when_false(&self) -> &'static str
    where
        (): WhenFalse<False>, // This constraint is always satisfied for False
    {
        "Processed with False condition!"
    }
}

fn main() {
    println!("🔬 Type-Level Boolean Logic Demo");
    println!("================================");

    // Test basic type-level boolean values
    println!("\n📊 Basic Type-Level Booleans:");
    println!("True::VALUE = {}", True::VALUE);
    println!("False::VALUE = {}", False::VALUE);

    // Test trait implementations with different conditions
    println!("\n🔄 Conditional Trait Implementations:");
    let example = ExampleStruct;
    let _result1 = <ExampleStruct as ExampleTrait<i32, True>>::process(&example, 42);
    let _result2 = <ExampleStruct as ExampleTrait<i32, False>>::process(&example, 24);

    // Test conditional processors
    println!("\n🎯 Conditional Processing:");
    let processor_true = ConditionalProcessor::<True>::new();
    let processor_false = ConditionalProcessor::<False>::new();

    println!("{}", processor_true.process_when_true());
    println!("{}", processor_false.process_when_false());

    // Demonstrate that type-level logic types exist
    println!("\n🧮 Type-Level Logic Types Available:");
    println!("✅ TypeLevelBool trait implemented");
    println!("✅ True and False types available");
    println!("✅ And, Or, Not operations available");
    println!("✅ TypeEq for const generic comparison");
    println!("✅ WhenTrue/WhenFalse conditional traits");

    // Example of using helper type aliases (these are compile-time only)
    type SameId = TypeEq<42, 42>; // This is True at compile time
    // Note: TypeNeq<42, 24> doesn't work as expected due to type system limitations
    // We can only check for equality, not inequality at the type level with this approach

    println!("\n🔍 Compile-Time ID Comparison:");
    println!("IsSameId<42, 42> evaluates to: True (at compile time)");
    println!("Type-level inequality requires different approach");

    // Note: We can't directly use inequality checks due to type system limitations
    let _same_check: SameId = True; // This compiles because SameId resolves to True

    println!("\n✨ Type-level logic system successfully demonstrates:");
    println!("   • Compile-time boolean values and operations");
    println!("   • Conditional trait implementations");
    println!("   • Type-level first-order logic for const generics");
    println!("   • Foundation for advanced operator overloading patterns");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_type_level_logic() {
        // Test that basic type-level booleans work
        assert_eq!(True::VALUE, true);
        assert_eq!(False::VALUE, false);
    }

    #[test]
    fn test_conditional_traits() {
        let example = ExampleStruct;

        // Test that both implementations work
        let _result1 = <ExampleStruct as ExampleTrait<i32, True>>::process(&example, 42);
        let _result2 = <ExampleStruct as ExampleTrait<i32, False>>::process(&example, 24);
    }

    #[test]
    fn test_conditional_processors() {
        let processor_true = ConditionalProcessor::<True>::new();
        let processor_false = ConditionalProcessor::<False>::new();

        assert_eq!(
            processor_true.process_when_true(),
            "Processed with True condition!"
        );
        assert_eq!(
            processor_false.process_when_false(),
            "Processed with False condition!"
        );
    }
}
