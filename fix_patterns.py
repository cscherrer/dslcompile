#!/usr/bin/env python3

import re
import glob
import os

def fix_file(filepath):
    """Add CSE patterns to a file if needed"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already has BoundVar patterns
    if 'BoundVar' in content:
        return False
    
    # Skip if no ASTRepr patterns
    if 'ASTRepr::' not in content:
        return False
    
    # Find match statements and add patterns before closing braces
    # Look for patterns like:
    # ASTRepr::Sum(...) => { ... }
    # }  <- add patterns before this
    
    lines = content.split('\n')
    new_lines = []
    in_match = False
    match_depth = 0
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Detect match statements with ASTRepr
        if 'match' in line and any(f'ASTRepr::{variant}' in lines[j] for j in range(i, min(i+20, len(lines))) for variant in ['Variable', 'Constant', 'Add']):
            in_match = True
            match_depth = 0
        
        if in_match:
            # Track braces
            match_depth += line.count('{') - line.count('}')
            
            # If we're closing the match and haven't added CSE patterns
            if match_depth == 0 and '}' in line and 'ASTRepr::Sum' in content[content.find('match'):content.find(line)]:
                # Add CSE patterns before this closing brace
                indent = ' ' * (len(line) - len(line.lstrip()))
                new_lines.insert(-1, f'{indent}ASTRepr::BoundVar(_) => todo!("BoundVar not implemented"),')
                new_lines.insert(-1, f'{indent}ASTRepr::Let(_, _, _) => todo!("Let not implemented"),')
                in_match = False
    
    new_content = '\n'.join(new_lines)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    
    return False

def main():
    print("Fixing CSE patterns in all Rust files...")
    
    updated_files = []
    for filepath in glob.glob('src/**/*.rs', recursive=True):
        if fix_file(filepath):
            updated_files.append(filepath)
    
    print(f"Updated {len(updated_files)} files:")
    for f in updated_files:
        print(f"  {f}")
    
    print("Testing compilation...")
    os.system("cargo check 2>&1 | grep 'not covered' | wc -l")

if __name__ == "__main__":
    main() 