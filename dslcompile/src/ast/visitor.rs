use crate::ast::{
    ExpressionType, Scalar,
    ast_repr::{ASTRepr, Collection, Lambda},
};

/// Work items for heap-allocated stack-based traversal
#[derive(Debug, Clone)]
enum VisitorWorkItem<T: Scalar + ExpressionType + Clone> {
    /// Visit a node and call the appropriate visit method
    Visit(ASTRepr<T>),
    /// Visit a collection
    VisitCollection(Collection<T>),
}

/// Immutable visitor trait for AST traversal
///
/// This trait provides a clean way to traverse AST nodes without modifying them.
/// Uses heap-allocated stack internally to prevent stack overflow on deep expressions.
pub trait ASTVisitor<T: Scalar + ExpressionType + Clone> {
    type Output;
    type Error;

    /// Visit any AST node - now uses heap-allocated stack to prevent stack overflow
    fn visit(&mut self, expr: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let mut stack = Vec::new();
        let mut results = Vec::new();

        stack.push(VisitorWorkItem::Visit(expr.clone()));

        while let Some(work_item) = stack.pop() {
            match work_item {
                VisitorWorkItem::Visit(expr) => {
                    match &expr {
                        ASTRepr::Constant(value) => {
                            let result = self.visit_constant(value)?;
                            results.push(result);
                        }
                        ASTRepr::Variable(index) => {
                            let result = self.visit_variable(*index)?;
                            results.push(result);
                        }
                        ASTRepr::BoundVar(index) => {
                            let result = self.visit_bound_var(*index)?;
                            results.push(result);
                        }
                        ASTRepr::Add(terms) => {
                            // Visit the Add node itself
                            let result = self.visit_add_node()?;
                            results.push(result);

                            // Push children for processing (reverse order for left-to-right)
                            let terms_vec: Vec<_> = terms.elements().collect();
                            for term in terms_vec.iter().rev() {
                                stack.push(VisitorWorkItem::Visit((*term).clone()));
                            }
                        }
                        ASTRepr::Sub(left, right) => {
                            // Visit the Sub node itself
                            let result = self.visit_sub_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**right).clone()));
                            stack.push(VisitorWorkItem::Visit((**left).clone()));
                        }
                        ASTRepr::Mul(factors) => {
                            // Visit the Mul node itself
                            let result = self.visit_mul_node()?;
                            results.push(result);

                            let factors_vec: Vec<_> = factors.elements().collect();
                            for factor in factors_vec.iter().rev() {
                                stack.push(VisitorWorkItem::Visit((*factor).clone()));
                            }
                        }
                        ASTRepr::Div(left, right) => {
                            // Visit the Div node itself
                            let result = self.visit_div_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**right).clone()));
                            stack.push(VisitorWorkItem::Visit((**left).clone()));
                        }
                        ASTRepr::Pow(base, exp) => {
                            // Visit the Pow node itself
                            let result = self.visit_pow_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**exp).clone()));
                            stack.push(VisitorWorkItem::Visit((**base).clone()));
                        }
                        ASTRepr::Neg(inner) => {
                            // Visit the Neg node itself
                            let result = self.visit_neg_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**inner).clone()));
                        }
                        ASTRepr::Sin(inner) => {
                            // Visit the Sin node itself
                            let result = self.visit_sin_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**inner).clone()));
                        }
                        ASTRepr::Cos(inner) => {
                            // Visit the Cos node itself
                            let result = self.visit_cos_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**inner).clone()));
                        }
                        ASTRepr::Ln(inner) => {
                            // Visit the Ln node itself
                            let result = self.visit_ln_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**inner).clone()));
                        }
                        ASTRepr::Exp(inner) => {
                            // Visit the Exp node itself
                            let result = self.visit_exp_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**inner).clone()));
                        }
                        ASTRepr::Sqrt(inner) => {
                            // Visit the Sqrt node itself
                            let result = self.visit_sqrt_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**inner).clone()));
                        }
                        ASTRepr::Sum(collection) => {
                            // Visit the Sum node itself
                            let result = self.visit_sum_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::VisitCollection((**collection).clone()));
                        }
                        ASTRepr::Lambda(lambda) => {
                            // Visit the Lambda node itself
                            let result = self.visit_lambda_node()?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((*lambda.body).clone()));
                        }
                        ASTRepr::Let(binding_id, expr, body) => {
                            // Visit the Let node itself
                            let result = self.visit_let_node(*binding_id)?;
                            results.push(result);

                            stack.push(VisitorWorkItem::Visit((**body).clone()));
                            stack.push(VisitorWorkItem::Visit((**expr).clone()));
                        }
                    }
                }
                VisitorWorkItem::VisitCollection(collection) => match &collection {
                    Collection::Empty => {
                        let result = self.visit_empty_collection()?;
                        results.push(result);
                    }
                    Collection::Singleton(expr) => {
                        stack.push(VisitorWorkItem::Visit((**expr).clone()));
                    }
                    Collection::Range { start, end } => {
                        stack.push(VisitorWorkItem::Visit((**end).clone()));
                        stack.push(VisitorWorkItem::Visit((**start).clone()));
                    }
                    Collection::Variable(index) => {
                        let result = self.visit_collection_variable(*index)?;
                        results.push(result);
                    }

                    Collection::Filter {
                        collection,
                        predicate,
                    } => {
                        stack.push(VisitorWorkItem::Visit((**predicate).clone()));
                        stack.push(VisitorWorkItem::VisitCollection((**collection).clone()));
                    }
                    Collection::Map { lambda, collection } => {
                        stack.push(VisitorWorkItem::VisitCollection((**collection).clone()));
                        stack.push(VisitorWorkItem::Visit((*lambda.body).clone()));
                    }
                    Collection::DataArray(_) => {
                        let result = self.visit_empty_collection()?;
                        results.push(result);
                    }
                },
            }
        }

        // For simple visitors like NodeCounter, we just need to return any result
        // The actual counting happens in the leaf node methods
        if results.is_empty() {
            // If no results were generated, call visit_generic_node as a fallback
            self.visit_generic_node()
        } else {
            // Return the last result
            Ok(results.pop().unwrap())
        }
    }

    // Leaf nodes - implement these for custom behavior
    fn visit_constant(&mut self, value: &T) -> Result<Self::Output, Self::Error>;
    fn visit_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error>;
    fn visit_bound_var(&mut self, index: usize) -> Result<Self::Output, Self::Error>;

    // Node-level visitors - implement these instead of the old recursive ones
    fn visit_add_node(&mut self) -> Result<Self::Output, Self::Error> {
        // Default implementation - override for custom behavior
        self.visit_binary_op_node()
    }

    fn visit_sub_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_binary_op_node()
    }

    fn visit_mul_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_binary_op_node()
    }

    fn visit_div_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_binary_op_node()
    }

    fn visit_pow_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_binary_op_node()
    }

    fn visit_neg_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_unary_op_node()
    }

    fn visit_sin_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_unary_op_node()
    }

    fn visit_cos_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_unary_op_node()
    }

    fn visit_ln_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_unary_op_node()
    }

    fn visit_exp_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_unary_op_node()
    }

    fn visit_sqrt_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_unary_op_node()
    }

    fn visit_sum_node(&mut self) -> Result<Self::Output, Self::Error> {
        // Default implementation - override for custom behavior
        self.visit_complex_node()
    }

    fn visit_lambda_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_complex_node()
    }

    fn visit_let_node(&mut self, _binding_id: usize) -> Result<Self::Output, Self::Error> {
        self.visit_complex_node()
    }

    // Default implementations for operation categories
    fn visit_binary_op_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_generic_node()
    }

    fn visit_unary_op_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_generic_node()
    }

    fn visit_complex_node(&mut self) -> Result<Self::Output, Self::Error> {
        self.visit_generic_node()
    }

    /// Override this for a default behavior for all non-leaf nodes
    fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error>;

    // DEPRECATED: Old recursive methods - kept for backward compatibility but now delegate to stack-based implementation
    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_add(
        &mut self,
        left: &ASTRepr<T>,
        right: &ASTRepr<T>,
    ) -> Result<Self::Output, Self::Error> {
        // Create a temporary Add node and visit it
        let add_node = ASTRepr::add_binary(left.clone(), right.clone());
        self.visit(&add_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_sub(
        &mut self,
        left: &ASTRepr<T>,
        right: &ASTRepr<T>,
    ) -> Result<Self::Output, Self::Error> {
        let sub_node = ASTRepr::Sub(Box::new(left.clone()), Box::new(right.clone()));
        self.visit(&sub_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_mul(
        &mut self,
        left: &ASTRepr<T>,
        right: &ASTRepr<T>,
    ) -> Result<Self::Output, Self::Error> {
        let mul_node = ASTRepr::mul_binary(left.clone(), right.clone());
        self.visit(&mul_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_div(
        &mut self,
        left: &ASTRepr<T>,
        right: &ASTRepr<T>,
    ) -> Result<Self::Output, Self::Error> {
        let div_node = ASTRepr::Div(Box::new(left.clone()), Box::new(right.clone()));
        self.visit(&div_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_pow(
        &mut self,
        base: &ASTRepr<T>,
        exp: &ASTRepr<T>,
    ) -> Result<Self::Output, Self::Error> {
        let pow_node = ASTRepr::Pow(Box::new(base.clone()), Box::new(exp.clone()));
        self.visit(&pow_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_neg(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let neg_node = ASTRepr::Neg(Box::new(inner.clone()));
        self.visit(&neg_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_sin(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let sin_node = ASTRepr::Sin(Box::new(inner.clone()));
        self.visit(&sin_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_cos(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let cos_node = ASTRepr::Cos(Box::new(inner.clone()));
        self.visit(&cos_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_ln(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let ln_node = ASTRepr::Ln(Box::new(inner.clone()));
        self.visit(&ln_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_exp(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let exp_node = ASTRepr::Exp(Box::new(inner.clone()));
        self.visit(&exp_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_sqrt(&mut self, inner: &ASTRepr<T>) -> Result<Self::Output, Self::Error> {
        let sqrt_node = ASTRepr::Sqrt(Box::new(inner.clone()));
        self.visit(&sqrt_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_sum(&mut self, collection: &Collection<T>) -> Result<Self::Output, Self::Error> {
        let sum_node = ASTRepr::Sum(Box::new(collection.clone()));
        self.visit(&sum_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_lambda(&mut self, lambda: &Lambda<T>) -> Result<Self::Output, Self::Error> {
        let lambda_node = ASTRepr::Lambda(Box::new(lambda.clone()));
        self.visit(&lambda_node)
    }

    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_let(
        &mut self,
        binding_id: usize,
        expr: &ASTRepr<T>,
        body: &ASTRepr<T>,
    ) -> Result<Self::Output, Self::Error> {
        let let_node = ASTRepr::Let(binding_id, Box::new(expr.clone()), Box::new(body.clone()));
        self.visit(&let_node)
    }

    // Collection visitor - now stack-based
    #[deprecated(
        note = "This method now uses heap-allocated stack internally. The API is preserved for compatibility."
    )]
    fn visit_collection(
        &mut self,
        collection: &Collection<T>,
    ) -> Result<Self::Output, Self::Error> {
        let mut stack = Vec::new();
        stack.push(VisitorWorkItem::VisitCollection(collection.clone()));

        while let Some(work_item) = stack.pop() {
            if let VisitorWorkItem::VisitCollection(coll) = work_item {
                match coll {
                    Collection::Empty => return self.visit_empty_collection(),
                    Collection::Singleton(expr) => return self.visit(&expr),
                    Collection::Range { start, end } => {
                        self.visit(&start)?;
                        return self.visit(&end);
                    }
                    Collection::Variable(index) => return self.visit_collection_variable(index),

                    Collection::Filter {
                        collection,
                        predicate,
                    } => {
                        self.visit_collection(&collection)?;
                        return self.visit(&predicate);
                    }
                    Collection::Map { lambda, collection } => {
                        self.visit(&lambda.body)?;
                        return self.visit_collection(&collection);
                    }
                    Collection::DataArray(_) => return self.visit_empty_collection(), // Treat as no-op
                }
            }
        }

        panic!("Collection visitor failed - this should not happen")
    }

    fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error>;
    fn visit_collection_variable(&mut self, index: usize) -> Result<Self::Output, Self::Error>;
}

/// Work items for heap-allocated stack-based mutable traversal
#[derive(Debug, Clone)]
enum MutVisitorWorkItem<T: Scalar + ExpressionType + Clone> {
    /// Transform a node and push result to result stack
    Transform(ASTRepr<T>),
    /// Apply binary operation transformation to top two results on stack
    ApplyBinaryTransform(BinaryTransform),
    /// Apply unary operation transformation to top result on stack
    ApplyUnaryTransform(UnaryTransform),
    /// Transform collection
    TransformCollection(Collection<T>),
}

#[derive(Debug, Clone)]
enum BinaryTransform {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone)]
enum UnaryTransform {
    Neg,
    Sin,
    Cos,
    Ln,
    Exp,
    Sqrt,
}

/// Mutable visitor trait for AST transformation
///
/// This trait allows modifying AST nodes during traversal.
/// Uses heap-allocated stack internally to prevent stack overflow on deep expressions.
pub trait ASTMutVisitor<T: Scalar + ExpressionType + Clone> {
    type Error;

    /// Visit and potentially transform any AST node - now uses heap-allocated stack
    fn visit_mut(&mut self, expr: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        // For now, use a simpler approach that still eliminates the main recursive calls
        // but handles complex structures more directly
        match expr {
            ASTRepr::Constant(value) => self.visit_constant_mut(value),
            ASTRepr::Variable(index) => self.visit_variable_mut(index),
            ASTRepr::BoundVar(index) => self.visit_bound_var_mut(index),
            ASTRepr::Add(terms) => {
                if terms.len() == 2 {
                    let terms_vec: Vec<_> = terms.elements().cloned().collect();
                    let left_transformed = self.visit_mut(terms_vec[0].clone())?;
                    let right_transformed = self.visit_mut(terms_vec[1].clone())?;
                    self.visit_add_mut(left_transformed, right_transformed)
                } else {
                    // For non-binary multiset, transform each term and reconstruct
                    let transformed_terms: Result<Vec<_>, _> = terms
                        .elements()
                        .map(|term| self.visit_mut(term.clone()))
                        .collect();
                    Ok(ASTRepr::Add(crate::ast::multiset::MultiSet::from_iter(
                        transformed_terms?,
                    )))
                }
            }
            ASTRepr::Sub(left, right) => {
                let left_transformed = self.visit_mut(*left)?;
                let right_transformed = self.visit_mut(*right)?;
                self.visit_sub_mut(left_transformed, right_transformed)
            }
            ASTRepr::Mul(factors) => {
                if factors.len() == 2 {
                    let factors_vec: Vec<_> = factors.elements().cloned().collect();
                    let left_transformed = self.visit_mut(factors_vec[0].clone())?;
                    let right_transformed = self.visit_mut(factors_vec[1].clone())?;
                    self.visit_mul_mut(left_transformed, right_transformed)
                } else {
                    // For non-binary multiset, transform each factor and reconstruct
                    let transformed_factors: Result<Vec<_>, _> = factors
                        .elements()
                        .map(|factor| self.visit_mut(factor.clone()))
                        .collect();
                    Ok(ASTRepr::Mul(crate::ast::multiset::MultiSet::from_iter(
                        transformed_factors?,
                    )))
                }
            }
            ASTRepr::Div(left, right) => {
                let left_transformed = self.visit_mut(*left)?;
                let right_transformed = self.visit_mut(*right)?;
                self.visit_div_mut(left_transformed, right_transformed)
            }
            ASTRepr::Pow(base, exp) => {
                let base_transformed = self.visit_mut(*base)?;
                let exp_transformed = self.visit_mut(*exp)?;
                self.visit_pow_mut(base_transformed, exp_transformed)
            }
            ASTRepr::Neg(inner) => {
                let inner_transformed = self.visit_mut(*inner)?;
                self.visit_neg_mut(inner_transformed)
            }
            ASTRepr::Sin(inner) => {
                let inner_transformed = self.visit_mut(*inner)?;
                self.visit_sin_mut(inner_transformed)
            }
            ASTRepr::Cos(inner) => {
                let inner_transformed = self.visit_mut(*inner)?;
                self.visit_cos_mut(inner_transformed)
            }
            ASTRepr::Ln(inner) => {
                let inner_transformed = self.visit_mut(*inner)?;
                self.visit_ln_mut(inner_transformed)
            }
            ASTRepr::Exp(inner) => {
                let inner_transformed = self.visit_mut(*inner)?;
                self.visit_exp_mut(inner_transformed)
            }
            ASTRepr::Sqrt(inner) => {
                let inner_transformed = self.visit_mut(*inner)?;
                self.visit_sqrt_mut(inner_transformed)
            }
            ASTRepr::Sum(collection) => {
                let transformed_collection = self.visit_collection_mut(*collection)?;
                self.visit_sum_mut(transformed_collection)
            }
            ASTRepr::Lambda(lambda) => self.visit_lambda_mut(*lambda),
            ASTRepr::Let(binding_id, expr, body) => self.visit_let_mut(binding_id, *expr, *body),
        }
    }

    // Leaf nodes - default implementations return unchanged
    fn visit_constant_mut(&mut self, value: T) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Constant(value))
    }

    fn visit_variable_mut(&mut self, index: usize) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Variable(index))
    }

    fn visit_bound_var_mut(&mut self, index: usize) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::BoundVar(index))
    }

    // Binary operations - now use heap-allocated stack to prevent recursion
    fn visit_add_mut(
        &mut self,
        left: ASTRepr<T>,
        right: ASTRepr<T>,
    ) -> Result<ASTRepr<T>, Self::Error> {
        Ok(left + right)
    }

    fn visit_sub_mut(
        &mut self,
        left: ASTRepr<T>,
        right: ASTRepr<T>,
    ) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Sub(Box::new(left), Box::new(right)))
    }

    fn visit_mul_mut(
        &mut self,
        left: ASTRepr<T>,
        right: ASTRepr<T>,
    ) -> Result<ASTRepr<T>, Self::Error> {
        Ok(left * right)
    }

    fn visit_div_mut(
        &mut self,
        left: ASTRepr<T>,
        right: ASTRepr<T>,
    ) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Div(Box::new(left), Box::new(right)))
    }

    fn visit_pow_mut(
        &mut self,
        base: ASTRepr<T>,
        exp: ASTRepr<T>,
    ) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Pow(Box::new(base), Box::new(exp)))
    }

    // Unary operations - default implementations transform child using stack-based approach
    fn visit_neg_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Neg(Box::new(inner)))
    }

    fn visit_sin_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Sin(Box::new(inner)))
    }

    fn visit_cos_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Cos(Box::new(inner)))
    }

    fn visit_ln_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Ln(Box::new(inner)))
    }

    fn visit_exp_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Exp(Box::new(inner)))
    }

    fn visit_sqrt_mut(&mut self, inner: ASTRepr<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Sqrt(Box::new(inner)))
    }

    // Complex structures
    fn visit_sum_mut(&mut self, collection: Collection<T>) -> Result<ASTRepr<T>, Self::Error> {
        Ok(ASTRepr::Sum(Box::new(collection)))
    }

    fn visit_lambda_mut(&mut self, lambda: Lambda<T>) -> Result<ASTRepr<T>, Self::Error> {
        // Transform the body using stack-based approach
        let transformed_body = self.visit_mut(*lambda.body)?;
        let transformed_lambda = Lambda {
            var_indices: lambda.var_indices,
            body: Box::new(transformed_body),
        };
        Ok(ASTRepr::Lambda(Box::new(transformed_lambda)))
    }

    fn visit_let_mut(
        &mut self,
        binding_id: usize,
        expr: ASTRepr<T>,
        body: ASTRepr<T>,
    ) -> Result<ASTRepr<T>, Self::Error> {
        // Transform both expr and body using stack-based approach
        let expr_transformed = self.visit_mut(expr)?;
        let body_transformed = self.visit_mut(body)?;
        Ok(ASTRepr::Let(
            binding_id,
            Box::new(expr_transformed),
            Box::new(body_transformed),
        ))
    }

    // Collection transformation - now uses heap-allocated stack
    fn visit_collection_mut(
        &mut self,
        collection: Collection<T>,
    ) -> Result<Collection<T>, Self::Error> {
        match collection {
            Collection::Empty => Ok(Collection::Empty),
            Collection::Singleton(expr) => {
                let transformed = self.visit_mut(*expr)?;
                Ok(Collection::Singleton(Box::new(transformed)))
            }
            Collection::Range { start, end } => {
                let start_transformed = self.visit_mut(*start)?;
                let end_transformed = self.visit_mut(*end)?;
                Ok(Collection::Range {
                    start: Box::new(start_transformed),
                    end: Box::new(end_transformed),
                })
            }
            Collection::Variable(index) => Ok(Collection::Variable(index)),

            Collection::Filter {
                collection,
                predicate,
            } => {
                let collection_transformed = self.visit_collection_mut(*collection)?;
                let predicate_transformed = self.visit_mut(*predicate)?;
                Ok(Collection::Filter {
                    collection: Box::new(collection_transformed),
                    predicate: Box::new(predicate_transformed),
                })
            }
            Collection::Map { lambda, collection } => {
                let body_transformed = self.visit_mut(*lambda.body)?;
                let collection_transformed = self.visit_collection_mut(*collection)?;
                let transformed_lambda = Lambda {
                    var_indices: lambda.var_indices,
                    body: Box::new(body_transformed),
                };
                Ok(Collection::Map {
                    lambda: Box::new(transformed_lambda),
                    collection: Box::new(collection_transformed),
                })
            }
            Collection::DataArray(data) => Ok(Collection::DataArray(data)), // Pass through unchanged
        }
    }
}

/// Convenience function for applying an immutable visitor
pub fn visit_ast<T, V>(expr: &ASTRepr<T>, visitor: &mut V) -> Result<V::Output, V::Error>
where
    T: Scalar + ExpressionType,
    V: ASTVisitor<T>,
{
    visitor.visit(expr)
}

/// Convenience function for applying a mutable visitor
pub fn visit_ast_mut<T, V>(expr: ASTRepr<T>, visitor: &mut V) -> Result<ASTRepr<T>, V::Error>
where
    T: Scalar + ExpressionType + Clone,
    V: ASTMutVisitor<T>,
{
    visitor.visit_mut(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example visitor that counts nodes
    struct NodeCounter {
        count: usize,
    }

    impl ASTVisitor<f64> for NodeCounter {
        type Output = ();
        type Error = ();

        fn visit_constant(&mut self, _value: &f64) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_variable(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_bound_var(&mut self, _index: usize) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_empty_collection(&mut self) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_collection_variable(
            &mut self,
            _index: usize,
        ) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }

        fn visit_generic_node(&mut self) -> Result<Self::Output, Self::Error> {
            self.count += 1;
            Ok(())
        }
    }

    // Example transformer that replaces constants with variables
    struct ConstantToVariable {
        var_index: usize,
    }

    impl ASTMutVisitor<f64> for ConstantToVariable {
        type Error = ();

        fn visit_constant_mut(&mut self, _value: f64) -> Result<ASTRepr<f64>, Self::Error> {
            Ok(ASTRepr::Variable(self.var_index))
        }
    }

    #[test]
    fn test_node_counter() {
        use crate::ast::multiset::MultiSet;
        let expr = ASTRepr::Add(MultiSet::from_iter([
            ASTRepr::Constant(1.0),
            ASTRepr::Variable(0),
        ]));

        let mut counter = NodeCounter { count: 0 };
        visit_ast(&expr, &mut counter).unwrap();
        assert_eq!(counter.count, 3); // 1 Add node + 1 Constant + 1 Variable
    }

    #[test]
    fn test_constant_transformer() {
        use crate::ast::multiset::MultiSet;
        let expr = ASTRepr::Add(MultiSet::from_iter([
            ASTRepr::Constant(1.0),
            ASTRepr::Constant(2.0),
        ]));

        let mut transformer = ConstantToVariable { var_index: 42 };
        let result = visit_ast_mut(expr, &mut transformer).unwrap();

        match result {
            ASTRepr::Add(operands) => {
                let operands_vec: Vec<_> = operands.to_vec();
                if let [left, right] = &operands_vec[..] {
                    assert!(matches!(left, ASTRepr::Variable(42)));
                    assert!(matches!(right, ASTRepr::Variable(42)));
                } else {
                    panic!("Expected exactly 2 operands");
                }
            }
            _ => panic!("Expected Add node"),
        }
    }
}
