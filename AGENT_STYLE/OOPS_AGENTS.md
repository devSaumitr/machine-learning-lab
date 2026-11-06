# AGENTS.md — Workspace Rules

## Teaching / Explanation Rule (ALWAYS APPLY)

**Every concept explanation, note file, code review, and revision sheet in this workspace MUST follow `MASTER_TEACHING_STYLE.md`.**

Non-negotiables:
1. Build from zero, numbered sections, one tiny example carried through.
2. Always show memory with ASCII diagrams (`__dict__`, references, MRO).
3. Predict → break → reveal; bust misconceptions explicitly in bold.
4. Prove every claim with runnable code + shown output.
5. Contrast with C++/Java when relevant.
6. End every topic with ONE bolded lock-in line (+ ASCII cheat sheet for big topics).

Note files live in `notes/` using the mandatory structure defined at the bottom of `MASTER_TEACHING_STYLE.md`.

## Rule Update Policy

**NEVER delete or overwrite existing rules.** When adding new teaching rules/styles:
1. Append them to `MASTER_TEACHING_STYLE.md` as a new numbered **Rule Set** (Rule Set 2, 3, ...).
2. On conflict, keep the old rule and mark it `(superseded by Rule Set N)` — never remove it.
3. Log every change in the master file's **Changelog** table.
4. Keep original standalone captures (e.g. `TEACHING_STYLE.md`) as historical records.
