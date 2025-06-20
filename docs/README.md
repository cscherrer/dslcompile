# DSL Compile Documentation

## 📚 CURRENT DOCUMENTATION (Use These)

### Essential Reading
- **[CURRENT_STATE.md](../dslcompile/CURRENT_STATE.md)** - **START HERE** - What works right now
- **[ROADMAP.md](../dslcompile/ROADMAP.md)** - Project status and current priorities
- **[DEVELOPER_NOTES.md](DEVELOPER_NOTES.md)** - Architecture overview

### Technical References  
- **[DSL_System_Architecture.md](DSL_System_Architecture.md)** - Complete system overview and data flow
- **[ANF_CHEATSHEET.md](ANF_CHEATSHEET.md)** - A-Normal Form reference
- **[FRUNK_ZERO_COST_EXPLANATION.md](FRUNK_ZERO_COST_EXPLANATION.md)** - HList architecture and zero-cost abstractions

### Working Examples
See `dslcompile/examples/` for verified, working code:
- `comprehensive_iid_gaussian_demo.rs` - Complete probabilistic programming
- `anf_cse_performance_test.rs` - ANF and CSE optimization  
- `collection_codegen_demo.rs` - Code generation patterns
- `static_context_iid_test.rs` - Static context usage

## 🗄️ ARCHIVED DOCUMENTATION

### Theoretical and Completed Work
See `archive_outdated/` for:
- API unification design documents (COMPLETE)
- Summation system migration guides (COMPLETE)  
- Collection design documents (COMPLETE)
- Variable composition theory documents
- Type-level scoping designs
- Trait-based compile-time system documentation

**These are kept for reference but describe theoretical work or completed migrations.**

## 🧭 NAVIGATION GUIDE

**New to DSLCompile?** → Start with [CURRENT_STATE.md](../CURRENT_STATE.md) ⚠️ (Note: compilation issues)
**Want to understand architecture?** → Read [DEVELOPER_NOTES.md](DEVELOPER_NOTES.md)
**Working on features?** → Check [ROADMAP.md](../dslcompile/ROADMAP.md) for current status
**Need examples?** → Browse `dslcompile/examples/` directory (may need updates)
**Understanding data flow?** → See [DSL_System_Architecture.md](DSL_System_Architecture.md)
**Want to contribute?** → Help fix compilation issues first

---

**Rule**: When documentation conflicts, trust CURRENT_STATE.md over everything else. Note that examples may not work due to current compilation issues. 