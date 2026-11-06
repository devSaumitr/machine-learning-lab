# CHAPTER GENERATION FORMAT — ML ROADMAP

## PURPOSE

This file is the mandatory format specification for generating every chapter Markdown file in the ML Roadmap project.

Before generating, modifying, expanding, or restructuring ANY chapter:

1. Read `AGENTS.md`.
2. Read `CHAPTER_FORMAT.md`.
3. Read `progress/current-state.md`.
4. Read the relevant roadmap/chapter files.
5. Read existing topic files if they already exist.
6. Follow the structure and teaching methodology defined below.

Do NOT generate a chapter from the user's request alone.

The existing project context always takes priority.

---

# 1. CHAPTER STRUCTURE

Every major chapter MUST follow this hierarchy:

```text
CHAPTER
│
├── Chapter Overview
│
├── Why This Chapter Matters
│
├── Prerequisites
│
├── Learning Objectives
│
├── Topic Map
│
├── Dependency Order
│
├── Topic 1
│   ├── Why
│   ├── Mental Model
│   ├── Concept
│   ├── Syntax
│   ├── Examples
│   ├── Internal Behavior
│   ├── Common Mistakes
│   ├── ML Connection
│   ├── Practice
│   ├── Interview
│   └── Cheat Sheet
│
├── Topic 2
│   └── ...
│
├── ...
│
├── Chapter-Level Practice
│
├── Chapter-Level Quiz
│
├── Interview Questions
│
├── Common Mistakes
│
├── Revision Strategy
│
├── Mastery Checklist
│
└── Next Chapter