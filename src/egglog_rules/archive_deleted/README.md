# Archived Egglog Rules

This directory contains egglog files that were deleted in the "drop old files" cleanup but preserved for their potential value.

## Files Archived

### `dependency_aware_core.egg` (251 lines)
**Why preserved**: Contains substantial work on variable dependency tracking and analysis. Implements grammar-embedded dependency analysis to eliminate variable capture bugs. Based on ideas from toomuch.diff and includes:
- VarSet datatype for tracking variable dependencies  
- Automatic dependency computation rules
- Independence checking for safe optimizations
- Basic cost model integration

**Status**: This was experimental work but contains valuable concepts that may be referenced for future dependency analysis features.

### `test_staged_math.egg` (70 lines)
**Why preserved**: Comprehensive test suite for staged mathematical optimization rules. Tests critical patterns like:
- Constant folding
- Variable collection (2*x + 3*x = 5*x)
- Division simplification ((x*x + 2*x) / x → x + 2)
- Basic algebraic identities

**Status**: These test patterns should be covered in current test suites, but this provides reference for optimization verification.

### `simple_dependency_test.egg` (142 lines)
**Why preserved**: Working demonstration of basic dependency tracking concepts. Shows how to:
- Track variable dependencies in expressions
- Check variable independence
- Apply safe optimization rules based on dependency analysis

**Status**: Simpler version of dependency_aware_core.egg, useful as a learning reference.

### `test_dependency_aware_grammar.egg` (117 lines)
**Why preserved**: Comprehensive test suite for the dependency-aware grammar system. Verifies that dependency tracking works correctly across various expression types.

**Status**: Test coverage for dependency analysis concepts that may be reimplemented.

## Recovery Information

These files were recovered from git commit `8bc8f08` (before the "drop old files" cleanup in `4f1210c`).

## Usage Notes

These files are archived for reference only. They are not part of the active build system and may reference outdated APIs or concepts. If you need to reference dependency analysis concepts, these files provide working examples of the approach. 