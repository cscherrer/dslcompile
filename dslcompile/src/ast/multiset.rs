//! Enhanced multiset implementation with exact coefficient arithmetic
//!
//! Provides a deterministic multiset backed by `BTreeMap` for proper
//! associative/commutative operations in mathematical expressions.
//! Now uses Multiplicity for exact coefficient and exponent handling.

use crate::ast::multiplicity::Multiplicity;
use std::{cmp::Ordering, collections::BTreeMap, fmt};

#[cfg(feature = "egg_optimization")]
use egg::{Id, LanguageChildren};

/// Wrapper that provides total ordering for any `PartialOrd` type
/// by placing NaN values at the end consistently
#[derive(Clone, Debug)]
struct OrderedWrapper<T>(T);

impl<T: PartialEq> PartialEq for OrderedWrapper<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialEq> Eq for OrderedWrapper<T> {}

impl<T: PartialOrd> PartialOrd for OrderedWrapper<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for OrderedWrapper<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ord) => ord,
            None => {
                // Handle NaN cases: if both are NaN, they're equal;
                // otherwise NaN goes to the end
                if self.0.partial_cmp(&self.0).is_none() && other.0.partial_cmp(&other.0).is_none()
                {
                    Ordering::Equal
                } else if self.0.partial_cmp(&self.0).is_none() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        }
    }
}

/// A multiset (bag) with exact coefficient arithmetic via Multiplicity
///
/// Maintains deterministic ordering via `BTreeMap` and uses Multiplicity
/// for exact coefficient handling (Integer → Rational → Float promotion).
/// This enables combining like terms: 0.5*x + x → 1.5*x
#[derive(Clone, PartialEq, Eq)]
pub struct MultiSet<T> {
    map: BTreeMap<OrderedWrapper<T>, Multiplicity>,
}

impl<T> MultiSet<T>
where
    T: PartialOrd + Clone,
{
    /// Create a new empty multiset
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Create a multiset from a single element with multiplicity 1
    pub fn singleton(element: T) -> Self {
        let mut map = BTreeMap::new();
        map.insert(OrderedWrapper(element), Multiplicity::one());
        Self { map }
    }

    /// Create a multiset from two elements
    pub fn pair(a: T, b: T) -> Self {
        let mut multiset = Self::new();
        multiset.insert(a);
        multiset.insert(b);
        multiset
    }

    /// Create a multiset from an iterator of elements
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut multiset = Self::new();
        for item in iter {
            multiset.insert(item);
        }
        multiset
    }

    /// Insert an element into the multiset with multiplicity 1
    pub fn insert(&mut self, element: T) {
        self.insert_with_multiplicity(element, Multiplicity::one());
    }

    /// Insert an element with specific multiplicity (coefficient/exponent)
    pub fn insert_with_multiplicity(&mut self, element: T, multiplicity: Multiplicity) {
        let wrapped = OrderedWrapper(element);
        let current = self
            .map
            .get(&wrapped)
            .cloned()
            .unwrap_or(Multiplicity::zero());
        let new_multiplicity = current.add(multiplicity);

        if new_multiplicity.is_zero() {
            self.map.remove(&wrapped);
        } else {
            self.map.insert(wrapped, new_multiplicity);
        }
    }

    /// Remove one occurrence of an element from the multiset
    /// Returns true if the element was present
    pub fn remove(&mut self, element: &T) -> bool {
        self.remove_with_multiplicity(element, Multiplicity::one())
    }

    /// Remove specific multiplicity of an element from the multiset
    /// Returns true if the element was present
    pub fn remove_with_multiplicity(&mut self, element: &T, multiplicity: Multiplicity) -> bool {
        let wrapped = OrderedWrapper(element.clone());
        if let Some(current) = self.map.get(&wrapped).cloned() {
            let new_multiplicity = current.add(multiplicity.multiply(Multiplicity::from_i64(-1)));

            if new_multiplicity.is_zero() || new_multiplicity.to_f64() < 0.0 {
                self.map.remove(&wrapped);
            } else {
                self.map.insert(wrapped, new_multiplicity);
            }
            true
        } else {
            false
        }
    }

    /// Get the multiplicity (coefficient/exponent) of an element in the multiset
    pub fn multiplicity(&self, element: &T) -> Multiplicity {
        self.map
            .get(&OrderedWrapper(element.clone()))
            .cloned()
            .unwrap_or(Multiplicity::zero())
    }

    /// Get the count of an element in the multiset (for backward compatibility)
    /// Note: This converts Multiplicity to usize, losing precision for non-integer multiplicities
    pub fn count(&self, element: &T) -> usize {
        let mult = self.multiplicity(element);
        match mult {
            Multiplicity::Integer(i) if i >= 0 => i as usize,
            _ => mult.to_f64().max(0.0) as usize,
        }
    }

    /// Check if the multiset contains an element
    pub fn contains(&self, element: &T) -> bool {
        self.map.contains_key(&OrderedWrapper(element.clone()))
    }

    /// Get the total number of elements (counting multiplicities)
    /// Note: This converts Multiplicity to usize, losing precision for non-integer multiplicities
    #[must_use]
    pub fn len(&self) -> usize {
        self.map
            .values()
            .map(|mult| match mult {
                Multiplicity::Integer(i) if *i >= 0 => *i as usize,
                _ => mult.to_f64().max(0.0) as usize,
            })
            .sum()
    }

    /// Check if the multiset is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get the number of distinct elements
    #[must_use]
    pub fn distinct_len(&self) -> usize {
        self.map.len()
    }

    /// Iterate over (element, multiplicity) pairs in sorted order
    pub fn iter_with_multiplicity(&self) -> impl Iterator<Item = (&T, &Multiplicity)> {
        self.map.iter().map(|(k, v)| (&k.0, v))
    }

    /// Iterate over (element, count) pairs in sorted order (for backward compatibility)
    /// Note: This converts Multiplicity to usize, losing precision for non-integer multiplicities
    pub fn iter(&self) -> impl Iterator<Item = (&T, usize)> + '_ {
        self.map.iter().map(|(k, v)| {
            let count = match v {
                Multiplicity::Integer(i) if *i >= 0 => *i as usize,
                _ => v.to_f64().max(0.0) as usize,
            };
            (&k.0, count)
        })
    }

    /// Iterate over all elements (with repetition) in sorted order
    /// Note: For fractional multiplicities, rounds down to nearest integer
    pub fn elements(&self) -> impl Iterator<Item = &T> + '_ {
        self.map.iter().flat_map(|(k, multiplicity)| {
            let count = match multiplicity {
                Multiplicity::Integer(i) if *i >= 0 => *i as usize,
                _ => multiplicity.to_f64().max(0.0) as usize,
            };
            std::iter::repeat_n(&k.0, count)
        })
    }

    /// Iterate over unique elements (no repetition) in sorted order
    /// This is more appropriate for multisets with fractional multiplicities
    pub fn unique_elements(&self) -> impl Iterator<Item = &T> {
        self.map.keys().map(|k| &k.0)
    }

    /// Convert to a Vec of all elements (with repetition)
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        self.elements().cloned().collect()
    }

    /// Union of two multisets (element multiplicities are added)
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (element, multiplicity) in &other.map {
            let current = result
                .map
                .get(element)
                .cloned()
                .unwrap_or(Multiplicity::zero());
            let new_multiplicity = current.add(multiplicity.clone());

            if new_multiplicity.is_zero() {
                result.map.remove(element);
            } else {
                result.map.insert(element.clone(), new_multiplicity);
            }
        }
        result
    }

    /// Get an arbitrary element (useful for single-element multisets)
    /// Returns None if empty
    #[must_use]
    pub fn pick(&self) -> Option<&T> {
        self.map.keys().next().map(|wrapped| &wrapped.0)
    }

    /// Remove all occurrences of an element, returning the previous multiplicity
    pub fn remove_all(&mut self, element: &T) -> Multiplicity {
        self.map
            .remove(&OrderedWrapper(element.clone()))
            .unwrap_or(Multiplicity::zero())
    }

    /// Remove all occurrences of an element, returning the previous count (for backward compatibility)
    pub fn remove_all_count(&mut self, element: &T) -> usize {
        let multiplicity = self.remove_all(element);
        match multiplicity {
            Multiplicity::Integer(i) if i >= 0 => i as usize,
            _ => multiplicity.to_f64().max(0.0) as usize,
        }
    }

    /// Get the first element (for single-element access patterns)
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.map.keys().next().map(|wrapped| &wrapped.0)
    }

    /// Get two elements as a tuple (for binary operation patterns)
    #[must_use]
    pub fn as_pair(&self) -> Option<(&T, &T)> {
        if self.distinct_len() == 2 {
            let mut iter = self.map.keys();
            let first = &iter.next()?.0;
            let second = &iter.next()?.0;
            Some((first, second))
        } else {
            None
        }
    }

    /// Convert to Vec (for indexing operations)
    #[must_use]
    pub fn as_vec(&self) -> Vec<&T> {
        self.elements().collect()
    }

    /// Convert to owned Vec
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.to_vec()
    }
}

impl<T> Default for MultiSet<T>
where
    T: PartialOrd + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for MultiSet<T>
where
    T: fmt::Debug + PartialOrd + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiSet{{")?;
        let mut first = true;
        for (element, multiplicity) in &self.map {
            if !first {
                write!(f, ", ")?;
            }
            if multiplicity.is_one() {
                write!(f, "{:?}", element.0)?;
            } else {
                write!(f, "{:?}×{}", element.0, multiplicity)?;
            }
            first = false;
        }
        write!(f, "}}")
    }
}

impl<T> FromIterator<T> for MultiSet<T>
where
    T: PartialOrd + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut multiset = Self::new();
        for item in iter {
            multiset.insert(item);
        }
        multiset
    }
}

/// Implementation of egg's LanguageChildren trait for MultiSet<Id>
/// 
/// This allows MultiSet<Id> to be used directly in egg's define_language! macro.
/// The implementation expands multiplicities: {x: 2, y: 1} becomes [x, x, y]
#[cfg(feature = "egg_optimization")]
impl LanguageChildren for MultiSet<Id> {
    fn len(&self) -> usize {
        // Total number of children = sum of all multiplicities
        self.map.values()
            .map(|m| m.to_f64().max(0.0) as usize)
            .sum()
    }
    
    fn can_be_length(_n: usize) -> bool {
        // MultiSet can represent any number of children
        true
    }
    
    fn from_vec(v: Vec<Id>) -> Self {
        // Create multiset from vector by counting occurrences
        MultiSet::from_iter(v)
    }
    
    fn as_slice(&self) -> &[Id] {
        // This is problematic - egg expects slice access but MultiSet stores differently
        // For now, panic with a helpful message if this is called
        panic!("MultiSet<Id>::as_slice() not supported - MultiSet stores counts, not expanded slices. Use to_id_vec() instead.")
    }
    
    fn as_mut_slice(&mut self) -> &mut [Id] {
        // Similarly problematic for mutation  
        panic!("MultiSet<Id>::as_mut_slice() not supported - MultiSet stores counts, not expanded slices. Use from_id_vec() instead.")
    }
}

/// Helper trait for expanding MultiSet<Id> to Vec<Id> and back
#[cfg(feature = "egg_optimization")]
impl MultiSet<Id> {
    /// Convert to Vec<Id> by expanding multiplicities
    pub fn to_id_vec(&self) -> Vec<Id> {
        self.map.iter()
            .flat_map(|(id_wrapper, multiplicity)| {
                let count = multiplicity.to_f64().max(0.0) as usize;
                std::iter::repeat_n(id_wrapper.0, count)
            })
            .collect()
    }
    
    /// Create from Vec<Id> by counting occurrences  
    pub fn from_id_vec(ids: Vec<Id>) -> Self {
        Self::from_iter(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut ms = MultiSet::new();
        assert!(ms.is_empty());
        assert_eq!(ms.len(), 0);

        ms.insert(1);
        ms.insert(2);
        ms.insert(1);

        assert_eq!(ms.len(), 3);
        assert_eq!(ms.distinct_len(), 2);
        assert_eq!(ms.count(&1), 2);
        assert_eq!(ms.count(&2), 1);
        assert_eq!(ms.count(&3), 0);
    }

    #[test]
    fn test_deterministic_ordering() {
        let ms1 = MultiSet::from_iter([3, 1, 2, 1]);
        let ms2 = MultiSet::from_iter([1, 2, 3, 1]);

        assert_eq!(ms1, ms2);

        let elements1: Vec<_> = ms1.elements().collect();
        let elements2: Vec<_> = ms2.elements().collect();
        assert_eq!(elements1, elements2); // Should be [1, 1, 2, 3]
    }

    #[test]
    fn test_union() {
        let ms1 = MultiSet::from_iter([1, 2]);
        let ms2 = MultiSet::from_iter([2, 3]);
        let union = ms1.union(&ms2);

        assert_eq!(union.count(&1), 1);
        assert_eq!(union.count(&2), 2);
        assert_eq!(union.count(&3), 1);
    }

    #[test]
    fn test_debug_format() {
        let ms = MultiSet::from_iter([1, 2, 1, 3]);
        let debug_str = format!("{ms:?}");
        // Should show counts for repeated elements
        assert!(debug_str.contains("1×2") || debug_str.contains("1, 1"));
    }
}
