# Documentation

This directory contains research documentation for the entropy_nn project.

## 📚 Documentation Structure

### [research_direction.md](research_direction.md)
**Detailed research plan and literature review**

For researchers and collaborators who want to understand:
- Core research questions and hypotheses
- Related work and literature gaps
- Implementation roadmap
- Success metrics
- Open questions

**Best for**: Understanding the "why" and "what's novel" about this project.

### [concepts_explained.md](concepts_explained.md)
**Accessible explanations of key concepts**

For researchers without deep ML/compression background who want to understand:
- What quantization, entropy, and compression mean
- How decode-on-the-fly execution works
- Why memory matters for training/inference
- Practical implications for robotics/edge deployment

**Best for**: Getting up to speed on fundamentals before diving into research.

---

## 🚀 Quick Start by Role

### I want to understand the motivation and approach
→ Start with [concepts_explained.md](concepts_explained.md) (explains energy/memory angle)
→ Read [research_direction.md](research_direction.md) for detailed plan

### I'm implementing the next feature
→ Check [research_direction.md](research_direction.md) § "Implementation Roadmap"
→ See [../CLAUDE.md](../CLAUDE.md) for current priorities
→ See [../README.md](../README.md) for code usage

### I'm writing up results for publication
→ Use [research_direction.md](research_direction.md) § "Related Work & Literature Gap"
→ Reference "Success Metrics" for what constitutes publishable results
→ Cite papers linked in "References" section

### I'm evaluating if this approach makes sense
→ Read [research_direction.md](research_direction.md) § "Why This Matters" (energy analysis)
→ Check "Core Research Questions" and "Success Metrics"

---

## 🔗 Key Links

- **Project README**: [../README.md](../README.md) - Usage and workflow
- **CLAUDE.md**: [../CLAUDE.md](../CLAUDE.md) - Project conventions and current priorities
- **Configs**: [../configs/](../configs/) - YAML experiment configurations
- **Code**: [../src/entropy_nn/](../src/entropy_nn/) - Source code

---

## 📅 Documentation Maintenance

**Last Major Update**: 2026-02-12

When to update:
- **After completing a research phase**: Update "Implementation Roadmap" status
- **New related work published**: Add to "Related Work" section
- **Major findings/insights**: Add to "Open Questions" or create new doc
- **Pivot/change direction**: Update "Core Research Questions"

