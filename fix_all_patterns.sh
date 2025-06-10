#!/bin/bash

echo "Adding wildcard patterns to fix compilation..."

# Find all Rust files with ASTRepr match statements
find src -name "*.rs" -exec grep -l "ASTRepr::" {} \; | while read file; do
    echo "Processing $file..."
    
    # Add wildcard pattern before the last closing brace of match statements
    # This is a simple approach - add catch-all patterns
    sed -i '/ASTRepr::Sum.*{/,/^[[:space:]]*}/ {
        /^[[:space:]]*}/ i\
            ASTRepr::BoundVar(_) => todo!("BoundVar not implemented"),\
            ASTRepr::Let(_, _, _) => todo!("Let not implemented"),
    }' "$file"
done

echo "Done! Testing compilation..."
cargo check --all-features --all-targets 2>&1 | grep "not covered" | wc -l 