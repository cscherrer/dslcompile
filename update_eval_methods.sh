#!/bin/bash

# Update all .eval( calls that use the deprecated API to .eval_old(
# These are the ones that use IntoEvalArray (Vec<f64>, arrays)

echo "Step 1: Updating deprecated .eval( calls to .eval_old( for array-based evaluation..."

# Update tests and examples that use array syntax like &[3.0, 4.0]
find . -name "*.rs" -exec sed -i 's/\.eval(&\([^,]*\), &\[\([^]]*\)\])/\.eval_old(\&\1, \&[\2])/g' {} \;
find . -name "*.rs" -exec sed -i 's/\.eval(&\([^,]*\), \[\([^]]*\)\])/\.eval_old(\&\1, [\2])/g' {} \;

echo "Step 2: Updating .eval_hlist( calls to .eval( for HList-based evaluation..."

# Update all .eval_hlist( calls to .eval(
find . -name "*.rs" -exec sed -i 's/\.eval_hlist(/\.eval(/g' {} \;

echo "Step 3: Updating documentation references..."

# Update documentation and comments
find . -name "*.rs" -exec sed -i 's/eval_hlist(/eval(/g' {} \;
find . -name "*.rs" -exec sed -i 's/eval_hlist\b/eval/g' {} \;

echo "Done! All eval method calls have been updated."
echo "- .eval( with arrays → .eval_old("
echo "- .eval_hlist( → .eval("
echo "- Documentation updated to reflect new naming" 