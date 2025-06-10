#!/usr/bin/env python3

import re
import os

# Files that need CSE pattern fixes
files_to_fix = [
    "src/ast/ast_utils.rs",
    "src/ast/normalization.rs", 
    "src/ast/runtime/expression_builder.rs",
    "src/ast/runtime/typed_registry.rs",
    "src/backends/rust_codegen.rs",
    "src/symbolic/anf.rs",
    "src/symbolic/symbolic.rs",
    "src/symbolic/symbolic_ad.rs"
]

def add_cse_patterns_to_file(filepath):
    """Add CSE pattern placeholders to a Rust file"""
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist, skipping...")
        return
    
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find match statements that have ASTRepr patterns
    # Look for patterns like "ASTRepr::Sum" followed by "}" to find end of match
    pattern = r'(ASTRepr::Sum[^}]*})'
    
    def add_cse_after_sum(match):
        sum_pattern = match.group(1)
        # Add CSE patterns after the Sum pattern
        cse_patterns = '''
            ASTRepr::BoundVar(_) => todo!("BoundVar not implemented"),
            ASTRepr::Let(_, _, _) => todo!("Let not implemented"),'''
        return sum_pattern + cse_patterns
    
    # Apply the replacement
    new_content = re.sub(pattern, add_cse_after_sum, content, flags=re.DOTALL)
    
    # Write back if changed
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  Updated {filepath}")
    else:
        print(f"  No changes needed for {filepath}")

def main():
    print("Adding CSE pattern placeholders...")
    
    for filepath in files_to_fix:
        add_cse_patterns_to_file(filepath)
    
    print("Done! Run 'cargo check' to see remaining compilation errors.")

if __name__ == "__main__":
    main() 