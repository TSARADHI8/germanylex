# GermanyLex

Domain-Specific RAG for German Legal & Regulatory Intelligence

Capstone project, M.Sc. Data Science, UE Potsdam. Supervisor: Dr. Humera Noor.

An agentic pipeline that answers questions about German visa, work, housing and employment rules
using only official sources, with the exact law, section and source link, and a citation check.

Status: Sprint 1 (data ingestion and chunking). Work in progress.

## Sources
gesetze-im-internet.de, bamf.de, EUR-Lex, bmas.de

## Structure
- data/raw: downloaded official texts
- data/processed: cleaned, chunked sections with metadata
- src: pipeline code
- tests: tests
