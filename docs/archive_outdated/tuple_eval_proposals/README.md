# Tuple Evaluation Proposals Archive

This directory contains substantial proposals for replacing HList-based evaluation with tuple-based evaluation that were deleted in the "drop old files" cleanup but preserved for their research value.

## Files Archived

### `tuple_eval_implementation.rs` (422 lines)
**Why preserved**: Comprehensive implementation of tuple-based evaluation with:
- Universal TupleEval trait abstracting over all tuple sizes
- O(1) variable access using const generics + arrays
- Zero match arms approach through trait interface
- Backward compatibility bridge from HLists to tuples
- Complete working examples and test coverage

**Key benefits proposed**:
- O(1) variable access instead of O(n) HList traversal
- Natural syntax: `(x, y, z)` instead of `hlist![x, y, z]`
- Better compile times and error messages
- Elimination of complex nested types

### `tuple_eval_proposal.rs` (468 lines)  
**Why preserved**: Detailed design proposal for tuple-based evaluation including:
- Macro-generated implementations for tuples up to size 12
- Integration with existing DynamicContext
- Lambda composition support with tuple destructuring
- Migration plan from HLists to tuples
- Performance analysis and usage examples

**Status**: According to memory, HList evaluation is the current approach, suggesting these proposals were not adopted.

## Research Value

These proposals contain substantial research into alternative evaluation approaches that could be valuable for:
- Performance optimization investigations
- Alternative API design considerations  
- Understanding trade-offs between different type-level approaches
- Future evaluation system redesign

## Recovery Information

These files were recovered from git commit `8bc8f08` (before the "drop old files" cleanup in `4f1210c`).

## Current Status

The DSLCompile system currently uses HList-based evaluation as documented in the memories. These tuple-based approaches represent alternative designs that were explored but not adopted. The HList approach was chosen for its heterogeneous type support and zero-cost abstractions.

## Usage Notes

These files are archived for research reference only. They may reference outdated APIs and are not part of the active build system. If tuple-based evaluation is reconsidered in the future, these files provide a comprehensive starting point. 