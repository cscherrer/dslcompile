use dslcompile::ast::{ASTRepr, Scalar};

/// Non-recursive visitor using explicit stack
pub struct StackBasedVisitor<T: Scalar> {
    stack: Vec<WorkItem<T>>,
}

/// Work items for the explicit stack
enum WorkItem<T: Scalar> {
    Visit(ASTRepr<T>),
    ProcessAdd(ASTRepr<T>, ASTRepr<T>),
    ProcessMul(ASTRepr<T>, ASTRepr<T>),
    // ... other operations
}

impl<T: Scalar + Clone> Default for StackBasedVisitor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Clone> StackBasedVisitor<T> {
    #[must_use]
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    /// Non-recursive traversal - no stack overflow!
    pub fn visit(&mut self, expr: ASTRepr<T>) -> Vec<String> {
        let mut results = Vec::new();
        self.stack.push(WorkItem::Visit(expr));

        while let Some(work_item) = self.stack.pop() {
            match work_item {
                WorkItem::Visit(expr) => {
                    match expr {
                        ASTRepr::Constant(value) => {
                            results.push(format!("Constant({value})"));
                        }
                        ASTRepr::Variable(index) => {
                            results.push(format!("Variable({index})"));
                        }
                        ASTRepr::Add(operands) => {
                            // Push children onto stack for later processing
                            for operand in operands.into_iter().rev() {
                                self.stack.push(WorkItem::Visit(operand));
                            }
                            results.push("Add".to_string());
                        }
                        ASTRepr::Mul(operands) => {
                            for operand in operands.into_iter().rev() {
                                self.stack.push(WorkItem::Visit(operand));
                            }
                            results.push("Mul".to_string());
                        }
                        ASTRepr::Sin(inner) => {
                            self.stack.push(WorkItem::Visit(*inner));
                            results.push("Sin".to_string());
                        }
                        // Handle other variants...
                        _ => {
                            results.push("Other".to_string());
                        }
                    }
                }
                WorkItem::ProcessAdd(left, right) => {
                    // Custom processing after children are visited
                    results.push("ProcessedAdd".to_string());
                }
                WorkItem::ProcessMul(left, right) => {
                    results.push("ProcessedMul".to_string());
                }
            }
        }

        results
    }
}

fn main() {
    // Create a VERY deep expression that would blow the stack with recursion
    let mut expr: ASTRepr<f64> = ASTRepr::Variable(0);

    // Build: ((((x + 1) + 2) + 3) + ... + 10000)
    for i in 1..=10000 {
        expr = expr + ASTRepr::Constant(f64::from(i));
    }

    println!("Created expression with depth: 10000");
    println!("This would cause stack overflow with recursive traversal!");

    let mut visitor = StackBasedVisitor::new();
    let results = visitor.visit(expr);

    println!(
        "Successfully traversed {} nodes without stack overflow!",
        results.len()
    );
    println!("First few operations: {:?}", &results[0..10]);
}
