use crate::ast::{
    ASTRepr, Scalar,
    ast_repr::{Collection, Lambda},
    multiset::MultiSet,
};

/// Work items for the explicit traversal stack
#[derive(Debug, Clone)]
enum WorkItem<T: Scalar + Clone> {
    /// Visit a node (pre-order)
    Visit(ASTRepr<T>),
    /// Process a node after its children have been visited (post-order)
    Process(ASTRepr<T>),
    /// Visit a collection
    VisitCollection(Collection<T>),
}

/// Non-recursive visitor trait using explicit stack
///
/// This trait eliminates stack overflow issues by using a heap-allocated
/// Vec as an explicit stack instead of relying on the call stack.
pub trait StackBasedVisitor<T: Scalar + Clone> {
    type Output;
    type Error;

    /// Visit a node - implement this for custom behavior
    fn visit_node(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error>;

    /// Visit a collection - implement this for custom collection handling
    fn visit_collection(
        &mut self,
        _collection: &Collection<T>,
    ) -> Result<Self::Output, Self::Error> {
        // Default: just visit as empty (override for custom behavior)
        self.visit_empty_collection()
    }

    /// Visit empty collection - implement this
    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error>;

    /// Main traversal method - no stack overflow!
    fn traverse(&mut self, expr: ASTRepr<T>) -> Result<Vec<Self::Output>, Self::Error> {
        let mut stack = Vec::new();
        let mut results = Vec::new();

        // Start with the root expression
        stack.push(WorkItem::Visit(expr));

        // Process stack until empty - no recursion!
        while let Some(work_item) = stack.pop() {
            match work_item {
                WorkItem::Visit(expr) => {
                    // Visit this node
                    let result = self.visit_node(&expr)?;
                    results.push(result);

                    // Push children onto stack for later processing
                    // Note: We push in reverse order so they're processed left-to-right
                    match expr {
                        ASTRepr::Add(terms) => {
                            let terms_vec: Vec<_> = terms.elements().collect();
                            for term in terms_vec.iter().rev() {
                                stack.push(WorkItem::Visit((*term).clone()));
                            }
                        }
                        ASTRepr::Mul(factors) => {
                            let factors_vec: Vec<_> = factors.elements().collect();
                            for factor in factors_vec.iter().rev() {
                                stack.push(WorkItem::Visit((*factor).clone()));
                            }
                        }
                        ASTRepr::Sub(left, right)
                        | ASTRepr::Div(left, right)
                        | ASTRepr::Pow(left, right) => {
                            stack.push(WorkItem::Visit(*right));
                            stack.push(WorkItem::Visit(*left));
                        }
                        ASTRepr::Neg(inner)
                        | ASTRepr::Sin(inner)
                        | ASTRepr::Cos(inner)
                        | ASTRepr::Ln(inner)
                        | ASTRepr::Exp(inner)
                        | ASTRepr::Sqrt(inner) => {
                            stack.push(WorkItem::Visit(*inner));
                        }
                        ASTRepr::Sum(collection) => {
                            stack.push(WorkItem::VisitCollection(*collection));
                        }
                        ASTRepr::Lambda(lambda) => {
                            stack.push(WorkItem::Visit(*lambda.body));
                        }
                        ASTRepr::Let(_binding_id, expr, body) => {
                            stack.push(WorkItem::Visit(*body));
                            stack.push(WorkItem::Visit(*expr));
                        }
                        // Leaf nodes - no children to visit
                        ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => {
                            // No children to process
                        }
                    }
                }
                WorkItem::VisitCollection(collection) => {
                    let result = self.visit_collection(&collection)?;
                    results.push(result);

                    // Push collection children onto stack
                    match collection {
                        Collection::Singleton(expr) => {
                            stack.push(WorkItem::Visit(*expr));
                        }
                        Collection::Range { start, end } => {
                            stack.push(WorkItem::Visit(*end));
                            stack.push(WorkItem::Visit(*start));
                        }

                        Collection::Filter {
                            collection,
                            predicate,
                        } => {
                            stack.push(WorkItem::Visit(*predicate));
                            stack.push(WorkItem::VisitCollection(*collection));
                        }
                        Collection::Map { lambda, collection } => {
                            stack.push(WorkItem::VisitCollection(*collection));
                            stack.push(WorkItem::Visit(*lambda.body));
                        }
                        // Leaf collections
                        Collection::Empty | Collection::Variable(_) | Collection::DataArray(_) => {
                            // No children to process
                        }
                    }
                }
                WorkItem::Process(_expr) => {
                    // For post-order processing if needed
                    // Currently unused but available for extension
                }
            }
        }

        Ok(results)
    }
}

/// Mutable stack-based visitor for transformations
pub trait StackBasedMutVisitor<T: Scalar + Clone> {
    type Error;

    /// Transform a node - implement this for custom transformations
    fn transform_node(&mut self, expr: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error>;

    /// Transform a collection - implement this for custom collection transformations
    fn transform_collection(
        &mut self,
        collection: Collection<T>,
    ) -> Result<Collection<T>, Self::Error> {
        // Default: return unchanged (override for custom behavior)
        Ok(collection)
    }

    /// Main transformation method - no stack overflow!
    fn transform(&mut self, expr: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        // For transformations, we need to rebuild the tree bottom-up
        // This is more complex and requires careful handling of the stack

        #[derive(Debug)]
        enum TransformWorkItem<T: Scalar + Clone> {
            Transform(ASTRepr<T>),
            Rebuild {
                original: ASTRepr<T>,
                transformed_children: Vec<ASTRepr<T>>,
            },
        }

        let mut stack = Vec::new();
        let mut result_stack = Vec::new();

        stack.push(TransformWorkItem::Transform(expr));

        while let Some(work_item) = stack.pop() {
            match work_item {
                TransformWorkItem::Transform(expr) => {
                    // Check if this node has children
                    let children_count = match &expr {
                        ASTRepr::Add(terms) => terms.len(),
                        ASTRepr::Mul(factors) => factors.len(),
                        ASTRepr::Sub(_, _) | ASTRepr::Div(_, _) | ASTRepr::Pow(_, _) => 2,
                        ASTRepr::Neg(_)
                        | ASTRepr::Sin(_)
                        | ASTRepr::Cos(_)
                        | ASTRepr::Ln(_)
                        | ASTRepr::Exp(_)
                        | ASTRepr::Sqrt(_) => 1,
                        ASTRepr::Lambda(_) => 1,
                        ASTRepr::Let(_, _, _) => 2,
                        ASTRepr::Sum(_) => 0, // Sum expressions are treated as atomic/leaf nodes
                        ASTRepr::Constant(_) | ASTRepr::Variable(_) | ASTRepr::BoundVar(_) => 0,
                    };

                    if children_count == 0 {
                        // Leaf node - transform directly
                        let transformed = self.transform_node(expr)?;
                        result_stack.push(transformed);
                    } else {
                        // Non-leaf node - need to transform children first
                        stack.push(TransformWorkItem::Rebuild {
                            original: expr.clone(),
                            transformed_children: Vec::new(),
                        });

                        // Push children for transformation (in reverse order)
                        match expr {
                            ASTRepr::Add(terms) => {
                                let terms_vec: Vec<_> = terms.elements().collect();
                                for term in terms_vec.iter().rev() {
                                    stack.push(TransformWorkItem::Transform((*term).clone()));
                                }
                            }
                            ASTRepr::Mul(factors) => {
                                let factors_vec: Vec<_> = factors.elements().collect();
                                for factor in factors_vec.iter().rev() {
                                    stack.push(TransformWorkItem::Transform((*factor).clone()));
                                }
                            }
                            ASTRepr::Sub(left, right)
                            | ASTRepr::Div(left, right)
                            | ASTRepr::Pow(left, right) => {
                                stack.push(TransformWorkItem::Transform(*right));
                                stack.push(TransformWorkItem::Transform(*left));
                            }
                            ASTRepr::Neg(inner)
                            | ASTRepr::Sin(inner)
                            | ASTRepr::Cos(inner)
                            | ASTRepr::Ln(inner)
                            | ASTRepr::Exp(inner)
                            | ASTRepr::Sqrt(inner) => {
                                stack.push(TransformWorkItem::Transform(*inner));
                            }
                            ASTRepr::Lambda(lambda) => {
                                stack.push(TransformWorkItem::Transform(*lambda.body));
                            }
                            ASTRepr::Let(_binding_id, expr, body) => {
                                stack.push(TransformWorkItem::Transform(*body));
                                stack.push(TransformWorkItem::Transform(*expr));
                            }
                            ASTRepr::Sum(collection) => {
                                // For now, treat collection as atomic
                                // TODO: Implement proper collection transformation
                                let transformed_collection =
                                    self.transform_collection(*collection)?;
                                result_stack.push(ASTRepr::Sum(Box::new(transformed_collection)));
                                continue;
                            }
                            _ => unreachable!("Already handled leaf nodes"),
                        }
                    }
                }
                TransformWorkItem::Rebuild {
                    original,
                    mut transformed_children,
                } => {
                    // Pop the required number of children from result stack
                    let children_count = match &original {
                        ASTRepr::Add(terms) => terms.len(),
                        ASTRepr::Mul(factors) => factors.len(),
                        ASTRepr::Sub(_, _)
                        | ASTRepr::Div(_, _)
                        | ASTRepr::Pow(_, _)
                        | ASTRepr::Let(_, _, _) => 2,
                        ASTRepr::Neg(_)
                        | ASTRepr::Sin(_)
                        | ASTRepr::Cos(_)
                        | ASTRepr::Ln(_)
                        | ASTRepr::Exp(_)
                        | ASTRepr::Sqrt(_)
                        | ASTRepr::Lambda(_) => 1,
                        _ => 0,
                    };

                    for _ in 0..children_count {
                        if let Some(child) = result_stack.pop() {
                            transformed_children.insert(0, child);
                        }
                    }

                    // Rebuild the node with transformed children
                    let rebuilt = match original {
                        ASTRepr::Add(_) => {
                            ASTRepr::Add(MultiSet::from_iter(transformed_children.clone()))
                        }
                        ASTRepr::Sub(_, _) => ASTRepr::Sub(
                            Box::new(transformed_children[0].clone()),
                            Box::new(transformed_children[1].clone()),
                        ),
                        ASTRepr::Mul(_) => {
                            ASTRepr::Mul(MultiSet::from_iter(transformed_children.clone()))
                        }
                        ASTRepr::Div(_, _) => ASTRepr::Div(
                            Box::new(transformed_children[0].clone()),
                            Box::new(transformed_children[1].clone()),
                        ),
                        ASTRepr::Pow(_, _) => ASTRepr::Pow(
                            Box::new(transformed_children[0].clone()),
                            Box::new(transformed_children[1].clone()),
                        ),
                        ASTRepr::Neg(_) => ASTRepr::Neg(Box::new(transformed_children[0].clone())),
                        ASTRepr::Sin(_) => ASTRepr::Sin(Box::new(transformed_children[0].clone())),
                        ASTRepr::Cos(_) => ASTRepr::Cos(Box::new(transformed_children[0].clone())),
                        ASTRepr::Ln(_) => ASTRepr::Ln(Box::new(transformed_children[0].clone())),
                        ASTRepr::Exp(_) => ASTRepr::Exp(Box::new(transformed_children[0].clone())),
                        ASTRepr::Sqrt(_) => {
                            ASTRepr::Sqrt(Box::new(transformed_children[0].clone()))
                        }
                        ASTRepr::Lambda(lambda) => ASTRepr::Lambda(Box::new(Lambda {
                            var_indices: lambda.var_indices.clone(),
                            body: Box::new(transformed_children[0].clone()),
                        })),
                        ASTRepr::Let(binding_id, _, _) => ASTRepr::Let(
                            binding_id,
                            Box::new(transformed_children[0].clone()),
                            Box::new(transformed_children[1].clone()),
                        ),
                        _ => original, // Shouldn't happen
                    };

                    // Apply final transformation to the rebuilt node
                    let final_transformed = self.transform_node(rebuilt)?;
                    result_stack.push(final_transformed);
                }
            }
        }

        // Should have exactly one result
        result_stack.pop().ok_or_else(|| {
            // Return a default error - implement proper error handling
            panic!("Stack-based transformation failed: no result")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NodeCounter {
        count: usize,
    }

    impl StackBasedVisitor<f64> for NodeCounter {
        type Output = String;
        type Error = ();

        fn visit_node(&mut self, expr: &ASTRepr<f64>) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            match expr {
                ASTRepr::Constant(val) => Ok(format!("Const({val})")),
                ASTRepr::Variable(idx) => Ok(format!("Var({idx})")),
                ASTRepr::Add(terms) => Ok(format!("Add({})", terms.len())),
                ASTRepr::Mul(factors) => Ok("Mul".to_string()),
                _ => Ok("Other".to_string()),
            }
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            Ok("EmptyCollection".to_string())
        }
    }

    #[test]
    fn test_stack_based_visitor_no_overflow() {
        // Build a deep expression iteratively to avoid stack overflow during construction
        // Create nested Sin(Cos(Ln(Exp(...)))) operations
        let mut expr = ASTRepr::Variable(0);

        // Wrap in 250 levels of unary operations (1000 operations total cycling through 4 types)
        for i in 0..250 {
            expr = match i % 4 {
                0 => ASTRepr::Sin(Box::new(expr)),
                1 => ASTRepr::Cos(Box::new(expr)),
                2 => ASTRepr::Ln(Box::new(expr)),
                _ => ASTRepr::Exp(Box::new(expr)),
            };
        }

        // Now traverse this deep expression with our stack-based visitor
        let mut visitor = NodeCounter { count: 0 };
        let results = visitor.traverse(expr.clone()).unwrap();

        // Should have visited 251 nodes (250 operations + 1 variable)
        assert_eq!(visitor.count, 251);
        assert_eq!(results.len(), 251);

        // Create an even deeper expression to really test stack safety
        let mut deep_expr = ASTRepr::Variable(0);
        for i in 0..10000 {
            deep_expr = match i % 4 {
                0 => ASTRepr::Neg(Box::new(deep_expr)),
                1 => ASTRepr::Sqrt(Box::new(deep_expr)),
                2 => ASTRepr::Exp(Box::new(deep_expr)),
                _ => ASTRepr::Sin(Box::new(deep_expr)),
            };
        }

        // This would definitely overflow a recursive visitor's stack
        let mut deep_visitor = NodeCounter { count: 0 };
        let deep_results = deep_visitor.traverse(deep_expr).unwrap();

        // Should handle 10000 levels without stack overflow
        assert_eq!(deep_visitor.count, 10001);
        assert_eq!(deep_results.len(), 10001);
    }

    struct ConstantDoubler;

    impl StackBasedMutVisitor<f64> for ConstantDoubler {
        type Error = ();

        fn transform_node(&mut self, expr: ASTRepr<f64>) -> Result<ASTRepr<f64>, Self::Error> {
            match expr {
                ASTRepr::Constant(val) => Ok(ASTRepr::Constant(val * 2.0)),
                other => Ok(other),
            }
        }
    }

    #[test]
    fn test_stack_based_mut_visitor() {
        use crate::ast::multiset::MultiSet;
        let expr = ASTRepr::Add(MultiSet::from_iter([
            ASTRepr::Constant(5.0),
            ASTRepr::Constant(10.0),
        ]));

        let mut transformer = ConstantDoubler;
        let result = transformer.transform(expr).unwrap();

        // Constants should be doubled
        match result {
            ASTRepr::Add(terms) => {
                assert_eq!(terms.len(), 2);
                let elements: Vec<_> = terms.to_vec();
                assert!(
                    matches!(elements[0], ASTRepr::Constant(val) if val == 10.0 || val == 20.0)
                );
                assert!(
                    matches!(elements[1], ASTRepr::Constant(val) if val == 10.0 || val == 20.0)
                );
                // Just check that we have the right constants (order might vary due to sorting)
                let values: Vec<f64> = elements
                    .iter()
                    .filter_map(|e| match e {
                        ASTRepr::Constant(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                assert!(values.contains(&10.0) && values.contains(&20.0));
            }
            _ => panic!("Expected Add node"),
        }
    }
}
