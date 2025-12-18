# Elixir Graph

Ontology-augmented code generation for Elixir using RDF/OWL semantic annotations.

## Overview

This project explores combining structured knowledge (RDF/OWL ontologies) with Elixir source code to improve neural code generation. By encoding OTP behaviors, type specifications, and module relationships as semantic annotations, the model can leverage domain knowledge that would otherwise require massive training corpora to learn implicitly.

Research suggests this approach can yield **7-25% improvements** on semantic understanding tasks, with the greatest gains in type inference and pattern recognition.

## Approach

### Linearized Triple Representations

Rather than complex graph neural network architectures, this project uses linearized triples that standard transformers can process:

```
[CODE] def handle_call(request, from, state) [/CODE]
[ONTO] <module>GenServer</module> <pattern>handle_call</pattern> <type>sync_request</type> [/ONTO]
```

### Ontology Structure

Four ontology files will define Elixir semantics:

| File | Purpose |
|------|---------|
| `elixir-core.ttl` | Core language primitives and types |
| `elixir-otp.ttl` | OTP behaviours (GenServer, Supervisor, Agent) |
| `elixir-structure.ttl` | Module and function relationships |
| `elixir-shapes.ttl` | SHACL shapes for validation |

### Training Pipeline

1. **Corpus**: ~10,000 annotated Elixir functions with ontology coverage
2. **Curriculum Learning**: Gradually introduce ontological complexity
3. **Multi-task Training**: MLM + triple prediction + contrastive learning
4. **Parameter-efficient Fine-tuning**: LoRA for limited data scenarios

## Tech Stack

- **Axon** - Neural network layers with custom graph attention
- **Bumblebee** - Tokenization via Rust bindings
- **EXLA** - GPU acceleration
- **rdf-ex** - RDF parsing and manipulation

## Status

Currently in research and planning phase. See `notes/research/` for detailed implementation analysis.

## References

Key influences:
- GraphCodeBERT (data flow graph integration)
- K-BERT (knowledge triple injection)
- TyFlow (type-constrained generation)
- Code Graph Model (semantic node compression)
