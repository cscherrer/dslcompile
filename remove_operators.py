#!/usr/bin/env python3

import re

# Read the file
with open('dslcompile/src/ast/runtime/expression_builder.rs', 'r') as f:
    content = f.read()

# Find the start and end of operator implementations
start_marker = "// OPERATOR OVERLOADING FOR VariableExpr - AUTOMATIC CONVERSION"
end_marker = "#[cfg(test)]"

# Find the positions
start_pos = content.find(start_marker)
end_pos = content.find(end_marker)

if start_pos == -1 or end_pos == -1:
    print("Could not find markers")
    exit(1)

# Find the start of the line containing the start marker
start_line_start = content.rfind('\n', 0, start_pos) + 1

# Keep everything before the operators and everything after
new_content = content[:start_line_start] + "// All operator implementations moved to operators module\n\n" + content[end_pos:]

# Write back
with open('dslcompile/src/ast/runtime/expression_builder.rs', 'w') as f:
    f.write(new_content)

print("Successfully removed operator implementations")
print(f"Removed from position {start_line_start} to {end_pos}") 