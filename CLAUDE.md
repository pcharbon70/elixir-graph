# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project aims to build an ontology-augmented code generation model for Elixir. The goal is to combine RDF/OWL semantic annotations with Elixir source code to improve code generation through structured knowledge injection.

**Current Status**: Research/planning phase. No implementation code exists yet.

## Planned Architecture

The system will use:
- **Linearized triple representations** for combining RDF/OWL ontologies with code tokens
- **Axon** for neural network layers (with custom graph attention implementations)
- **Bumblebee** for tokenization
- **EXLA** for GPU acceleration
- **rdf-ex** for RDF parsing

Expected ontology files (not yet created):
- `elixir-core.ttl` - Core Elixir language semantics
- `elixir-otp.ttl` - OTP behaviours and patterns
- `elixir-structure.ttl` - Module and function structure
- `elixir-shapes.ttl` - SHACL shapes for validation

## Build Commands

Once implementation begins, standard Elixir/Mix commands will apply:
```bash
mix deps.get          # Install dependencies
mix compile           # Compile the project
mix test              # Run tests
mix test path/to/test.exs:42  # Run single test at line
```

## Key Technical Decisions

- Use linearized triples over graph neural networks for initial implementation (simpler architecture)
- Hybrid approach: Python for graph preprocessing/embedding generation, Elixir for training/inference
- Curriculum learning: gradually introduce ontological complexity during training
- Parameter-efficient fine-tuning (LoRA) to work with limited Elixir corpus (~10k annotated functions minimum)
