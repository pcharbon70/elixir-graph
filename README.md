# Elixir Graph

Ontology-augmented code generation for Elixir combining RDF/OWL semantic annotations with neural language models to produce code that is functionally correct, idiomatic, and secure.

## Project Goals

This project builds a code generation model that leverages structured knowledge about Elixir—OTP behaviors, type specifications, module relationships, code quality rules, and security patterns—to compensate for the relatively small Elixir training corpus compared to languages like Python or JavaScript.

Research on knowledge-enhanced language models (ERNIE, K-BERT, GraphCodeBERT) demonstrates that structured knowledge injection can improve code generation by **7-25%** on semantic understanding tasks. The greatest gains appear in:

- **Type inference** (+25% accuracy)
- **OTP pattern recognition** (+18% pass@1)
- **Security vulnerability prevention** (35-50% F1 improvement with contrastive learning)
- **Code quality compliance** (2.3x improvement on difficult tasks with curriculum learning)

## Architecture

### Linearized Triple Representations

Rather than complex graph neural network architectures requiring substantial custom implementation, this project uses linearized triples that standard transformers can process directly:

```
[CODE] def handle_call(request, from, state) [/CODE]
[ONTO] <module>GenServer</module> <pattern>handle_call</pattern> <type>sync_request</type> [/ONTO]
```

This approach maintains compatibility with existing transformer architectures while preserving ontological structure. Graph embeddings via RDF2Vec or OWL2Vec* can be precomputed and incorporated as additional embedding dimensions.

### Ontology Foundation

Four ontology files define Elixir semantics (available at [pcharbon70/elixir-ontologies](https://github.com/pcharbon70/elixir-ontologies)):

| File | Purpose |
|------|---------|
| `elixir-core.ttl` | Core language primitives, types, and operators |
| `elixir-otp.ttl` | OTP behaviours (GenServer, Supervisor, Agent, Task) |
| `elixir-structure.ttl` | Module hierarchy, function signatures, dependencies |
| `elixir-shapes.ttl` | SHACL validation shapes (extending to Credo and security rules) |

### Multi-Task Learning Architecture

The model uses a shared transformer encoder with task-specific heads:

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Transformer Encoder                │
│                  (Code + Ontology Embeddings)                │
└─────────────────────────────────────────────────────────────┘
        │                │                │                │
        ▼                ▼                ▼                ▼
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│ Generation│    │  Quality  │    │ Security  │    │Explanation│
│    Head   │    │   Head    │    │   Head    │    │   Head    │
│  (LM)     │    │ (Credo)   │    │(Sobelow)  │    │ (Seq2Seq) │
└───────────┘    └───────────┘    └───────────┘    └───────────┘
```

## Training Pipeline

### Data Sources

- **Minimum viable corpus**: ~10,000 annotated Elixir functions with ontology coverage
- **Sources**: GitHub repositories (filtered by stars/forks), HexDocs, Elixir Forum, Stack Overflow
- **Augmentation target**: 50,000 total samples via synthetic generation

### Training Objectives

1. **Code Generation** - Masked language modeling with type-specific masking (separating identifiers, operators, keywords)

2. **Quality Compliance** - Classification of Credo rule violations across 5 categories:
   - Consistency (cross-file uniformity)
   - Design (architectural concerns)
   - Readability (convention adherence)
   - Refactor (simplification opportunities)
   - Warning (likely bugs)

3. **Security Detection** - Classification of Sobelow findings mapped to CWE:
   - Command Injection (CWE-78)
   - SQL Injection (CWE-89)
   - XSS (CWE-79)
   - Directory Traversal (CWE-22)
   - Denial of Service / Atom Exhaustion (CWE-400)
   - Unsafe Deserialization (CWE-502)
   - Remote Code Execution (CWE-94)

4. **Explanation Generation** - Natural language explanations grounded in rule documentation via retrieval-augmented generation

### Curriculum Learning Schedule

Training progresses through phases of increasing complexity:

| Phase | Epochs | Focus |
|-------|--------|-------|
| 1 | 1-10 | Code-only MLM |
| 2 | 11-30 | Code + simple single-relation ontology annotations |
| 3 | 31-50 | Code + full multi-hop ontology graphs |
| 4 | 51+ | Fine-tuning on complex examples with complete augmentation |

For Credo rules, the curriculum orders by pattern complexity:
1. Surface token patterns (FunctionNames, IoInspect)
2. Local structural patterns (ModuleDoc, MaxLineLength)
3. Cross-function analysis (Nesting, CyclomaticComplexity)
4. Corpus-level patterns (DuplicatedCode, Consistency checks)

### Contrastive Learning

The model learns quality distinctions through programmatically generated code pairs:

```elixir
# Clean code (positive)
def process(data), do: transform(data)

# Violating code (negative) - IO.inspect injected
def process(data), do: data |> IO.inspect(label: :debug) |> transform()
```

InfoNCE loss pushes clean code representations away from violating variants in embedding space.

## Inference Pipeline

### Constrained Decoding

Three layers of constraint enforcement during generation:

| Layer | Approach | Overhead | Error Reduction |
|-------|----------|----------|-----------------|
| Syntax | Grammar-constrained via DFA mask stores | ~10% | 96% syntax errors |
| Semantic | Monitor-guided decoding at trigger points | Variable | 19-25% compilation |
| Quality/Security | Beam search with rejection sampling | 5x candidates | Depends on beam size |

### Generate-Check-Repair Loop

```
┌──────────┐     ┌─────────┐     ┌──────────┐
│ Generate │────▶│  Check  │────▶│  Clean?  │───Yes──▶ Return
└──────────┘     │ (Credo/ │     └──────────┘
                 │Sobelow) │           │
                 └─────────┘           No
                                       │
                 ┌─────────┐           ▼
                 │ Repair  │◀────┌──────────┐
                 │ Prompt  │     │ Refine & │
                 └─────────┘────▶│  Retry   │
                                 └──────────┘
```

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Neural Networks | Axon | Custom layers, graph attention |
| Tokenization | Bumblebee | BPE via Rust bindings |
| Acceleration | EXLA | GPU/TPU backend |
| RDF Processing | rdf-ex | Ontology parsing and manipulation |
| Code Quality | Credo | 83+ checks, programmatic API |
| Security | Sobelow | 30+ vulnerability types, JSON output |

### Hybrid Architecture

- **Python**: Graph preprocessing, embedding generation (pyRDF2Vec, OWL2Vec*)
- **Elixir**: Training loop, inference, deployment

This balances implementation effort against the goal of native Elixir ML tooling.

## Evaluation

### Primary Metrics

- **pass@k**: Probability that at least one of k samples passes all tests
- **secure-pass@k**: Probability of generating code that is both secure AND functionally correct
- **Credo clean rate**: Percentage of generations with zero quality violations
- **CodeBLEU**: Semantic similarity combining n-gram, AST, and data-flow matching

### Expected Improvements

| Dimension | Code-Only Baseline | With Ontology | Expected Δ |
|-----------|-------------------|---------------|------------|
| pass@1 (basic patterns) | ~65% | ~72% | +7% |
| pass@1 (OTP-specific) | ~40% | ~58% | +18% |
| Type inference accuracy | ~45% | ~70% | +25% |
| Semantic probing tasks | ~60% | ~80% | +20% |

### Ablation Studies

Isolating component contributions:

| Configuration | Component Removed | Expected Impact |
|--------------|-------------------|-----------------|
| Full model | None (baseline) | Reference |
| No type specs | @spec, @type | Type inference degradation |
| No behaviours | Behaviour semantics | Pattern recognition loss |
| No OTP patterns | Supervisor/GenServer ontology | Concurrency understanding loss |
| No Credo training | Quality compliance head | Increased style violations |
| No security training | Security detection head | Increased vulnerabilities |

## Project Status

**Current Phase**: Research and planning

See `notes/research/` for detailed analysis:
- `ontology-augmented-code-generation.md` - Core architecture and training approach
- `respecting-credo-rules.md` - Credo integration for code quality
- `code-security-enhancement.md` - Security patterns via Sobelow

## Research Foundation

Key influences on the architecture:

| Paper/System | Contribution |
|--------------|--------------|
| GraphCodeBERT | Data flow graph integration, joint pre-training |
| K-BERT | Knowledge triple injection without retraining |
| TyFlow | Type-constrained generation via synthesis rules |
| Code Graph Model | Semantic node compression (512x context extension) |
| ContraCode | Contrastive pre-training for code |
| Monitor-Guided Decoding | Static analysis in the decoding loop |
| SynCode | Grammar-constrained generation via DFA masks |
| VulLLM | Multi-task learning for vulnerability detection |
| CodeGuard+ | Secure-pass@k evaluation metric |

## License

TBD
