use std::marker::PhantomData;

/// Core trait for zero-overhead storage with compile-time type specialization
pub trait DirectStorage<T>: std::fmt::Debug {
    /// Get value with compile-time type specialization - O(1) access
    fn get_typed(&self, var_id: usize) -> T;
}

/// Extended trait for mutable storage operations
pub trait DirectStorageMut<T>: DirectStorage<T> {
    /// Add value with compile-time type specialization
    fn add_typed(&mut self, var_id: usize, value: T);
}

/// Zero-overhead inputs with fixed-size arrays for O(1) access
///
/// This provides the core storage pattern used across multiple modules
/// with compile-time bounds checking and zero runtime dispatch.
#[derive(Debug)]
pub struct ZeroOverheadInputs<const MAX_VARS: usize> {
    /// Fixed-size arrays for O(1) access - NO VEC LOOKUP!
    pub f64_values: [Option<f64>; MAX_VARS],
    pub f32_values: [Option<f32>; MAX_VARS],
    pub usize_values: [Option<usize>; MAX_VARS],
    pub vec_f64_values: [Option<Vec<f64>>; MAX_VARS],
    var_count: usize,
}

impl<const MAX_VARS: usize> Default for ZeroOverheadInputs<MAX_VARS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_VARS: usize> ZeroOverheadInputs<MAX_VARS> {
    /// Create new zero-overhead inputs storage
    #[must_use]
    pub fn new() -> Self {
        Self {
            f64_values: [None; MAX_VARS],
            f32_values: [None; MAX_VARS],
            usize_values: [None; MAX_VARS],
            vec_f64_values: std::array::from_fn(|_| None),
            var_count: 0,
        }
    }

    /// Add f64 value with O(1) access and bounds checking
    pub fn add_f64(&mut self, var_id: usize, value: f64) {
        assert!(
            var_id < MAX_VARS,
            "Variable ID {var_id} exceeds maximum {MAX_VARS}"
        );
        self.f64_values[var_id] = Some(value);
        self.var_count = self.var_count.max(var_id + 1);
    }

    /// Add f32 value with O(1) access and bounds checking
    pub fn add_f32(&mut self, var_id: usize, value: f32) {
        assert!(
            var_id < MAX_VARS,
            "Variable ID {var_id} exceeds maximum {MAX_VARS}"
        );
        self.f32_values[var_id] = Some(value);
        self.var_count = self.var_count.max(var_id + 1);
    }

    /// Add usize value with O(1) access and bounds checking
    pub fn add_usize(&mut self, var_id: usize, value: usize) {
        assert!(
            var_id < MAX_VARS,
            "Variable ID {var_id} exceeds maximum {MAX_VARS}"
        );
        self.usize_values[var_id] = Some(value);
        self.var_count = self.var_count.max(var_id + 1);
    }

    /// Add Vec<f64> value with O(1) access and bounds checking
    pub fn add_vec_f64(&mut self, var_id: usize, value: Vec<f64>) {
        assert!(
            var_id < MAX_VARS,
            "Variable ID {var_id} exceeds maximum {MAX_VARS}"
        );
        self.vec_f64_values[var_id] = Some(value);
        self.var_count = self.var_count.max(var_id + 1);
    }

    /// Get current variable count for validation
    #[must_use]
    pub fn var_count(&self) -> usize {
        self.var_count
    }
}

// ============================================================================
// COMPILE-TIME TRAIT SPECIALIZATION - O(1) ACCESS, ZERO RUNTIME DISPATCH!
// ============================================================================

impl<const MAX_VARS: usize> DirectStorage<f64> for ZeroOverheadInputs<MAX_VARS> {
    fn get_typed(&self, var_id: usize) -> f64 {
        // O(1) ARRAY ACCESS - NO VEC LOOKUP!
        self.f64_values[var_id].expect("f64 variable not found or wrong type")
    }
}

impl<const MAX_VARS: usize> DirectStorageMut<f64> for ZeroOverheadInputs<MAX_VARS> {
    fn add_typed(&mut self, var_id: usize, value: f64) {
        self.add_f64(var_id, value);
    }
}

impl<const MAX_VARS: usize> DirectStorage<f32> for ZeroOverheadInputs<MAX_VARS> {
    fn get_typed(&self, var_id: usize) -> f32 {
        // O(1) ARRAY ACCESS - NO VEC LOOKUP!
        self.f32_values[var_id].expect("f32 variable not found or wrong type")
    }
}

impl<const MAX_VARS: usize> DirectStorageMut<f32> for ZeroOverheadInputs<MAX_VARS> {
    fn add_typed(&mut self, var_id: usize, value: f32) {
        self.add_f32(var_id, value);
    }
}

impl<const MAX_VARS: usize> DirectStorage<usize> for ZeroOverheadInputs<MAX_VARS> {
    fn get_typed(&self, var_id: usize) -> usize {
        // O(1) ARRAY ACCESS - NO VEC LOOKUP!
        self.usize_values[var_id].expect("usize variable not found or wrong type")
    }
}

impl<const MAX_VARS: usize> DirectStorageMut<usize> for ZeroOverheadInputs<MAX_VARS> {
    fn add_typed(&mut self, var_id: usize, value: usize) {
        self.add_usize(var_id, value);
    }
}

impl<const MAX_VARS: usize> DirectStorage<Vec<f64>> for ZeroOverheadInputs<MAX_VARS> {
    fn get_typed(&self, var_id: usize) -> Vec<f64> {
        // O(1) ARRAY ACCESS - NO VEC LOOKUP!
        self.vec_f64_values[var_id]
            .as_ref()
            .expect("Vec<f64> variable not found or wrong type")
            .clone()
    }
}

impl<const MAX_VARS: usize> DirectStorageMut<Vec<f64>> for ZeroOverheadInputs<MAX_VARS> {
    fn add_typed(&mut self, var_id: usize, value: Vec<f64>) {
        self.add_vec_f64(var_id, value);
    }
}

/// Core trait for zero-overhead expression evaluation
pub trait ZeroOverheadExpr<T> {
    /// Evaluate with ZERO runtime dispatch - pure compile-time specialization
    fn eval_zero<S>(&self, inputs: &S) -> T
    where
        S: DirectStorage<T>;
}

/// Zero-overhead variable reference
#[derive(Debug, Clone)]
pub struct ZeroOverheadVar<T> {
    id: usize,
    _type: PhantomData<T>,
}

impl<T> ZeroOverheadVar<T> {
    /// Create new zero-overhead variable reference
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _type: PhantomData,
        }
    }

    /// Get variable ID
    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }
}

impl<T: Clone> ZeroOverheadExpr<T> for ZeroOverheadVar<T> {
    fn eval_zero<S>(&self, inputs: &S) -> T
    where
        S: DirectStorage<T>,
    {
        inputs.get_typed(self.id)
    }
}

/// Zero-overhead constant value
#[derive(Debug, Clone)]
pub struct ZeroOverheadConst<T> {
    value: T,
}

impl<T> ZeroOverheadConst<T> {
    /// Create new zero-overhead constant
    #[must_use]
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Get constant value
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: Clone> ZeroOverheadExpr<T> for ZeroOverheadConst<T> {
    fn eval_zero<S>(&self, _inputs: &S) -> T
    where
        S: DirectStorage<T>,
    {
        self.value.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_overhead_storage() {
        let mut storage = ZeroOverheadInputs::<8>::new();

        // Test f64 storage
        storage.add_f64(0, 2.5);
        assert_eq!(DirectStorage::<f64>::get_typed(&storage, 0), 2.5);

        // Test usize storage
        storage.add_usize(1, 42);
        assert_eq!(DirectStorage::<usize>::get_typed(&storage, 1), 42);

        // Test Vec<f64> storage
        let test_vec = vec![1.0, 2.0, 3.0];
        storage.add_vec_f64(2, test_vec.clone());
        assert_eq!(DirectStorage::<Vec<f64>>::get_typed(&storage, 2), test_vec);

        assert_eq!(storage.var_count(), 3);
    }

    #[test]
    fn test_zero_overhead_expressions() {
        let mut storage = ZeroOverheadInputs::<4>::new();
        storage.add_f64(0, 10.0);
        storage.add_f64(1, 20.0);

        let var_x = ZeroOverheadVar::<f64>::new(0);
        let var_y = ZeroOverheadVar::<f64>::new(1);
        let const_val = ZeroOverheadConst::new(5.0);

        assert_eq!(var_x.eval_zero(&storage), 10.0);
        assert_eq!(var_y.eval_zero(&storage), 20.0);
        assert_eq!(const_val.eval_zero(&storage), 5.0);
    }

    #[test]
    #[should_panic(expected = "Variable ID 8 exceeds maximum 4")]
    fn test_bounds_checking() {
        let mut storage = ZeroOverheadInputs::<4>::new();
        storage.add_f64(8, 1.0); // Should panic
    }
}
