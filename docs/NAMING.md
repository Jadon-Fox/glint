# GLINT naming

This repository is **GLINT, GitHub Project 6** (`users/Jadon-Fox/projects/6`).

Through 2026-11-14 many issues, pull requests, files and folders were named `P7`, `p7_*` or "Project 7". That was a mislabel from the October 2026 board numbering. There is no Project 7 for GLINT, and "Project 7" must not be confused with MMLRPV (Project 5), the former Board 7 (`corpus -> BigQuery -> MERCS`), HSVE, or LLMVE (Project 4).

## What changed on 2026-11-15

- Issue and pull request titles: `P7-` became `GLINT-`, and `(p7)` became `(glint)`. The full old-to-new list is `title_map_old_to_new.tsv` in `/home/workspace/reports/glint_github_cleanup_20261115/`.
- Project 6 draft cards `P7-0` to `P7-11` became `GLINT-0` to `GLINT-11`.
- Items that are not GLINT were removed from Project 6 (training_orchestrator #314-#337), and GLINT items were removed from the LLMVE board (Project 4).

## What did not change

Names that are identifiers keep their spelling, so nothing that depends on them breaks: `docs/p7_mode3_fold/`, `mojo/p7_*.mojo`, `scripts/p7_*`, `libp7_kernels.so`, `p7_plug_res4_v1`, `p7_attach_abi_v1`, branch names, and `lakes/p7_glint/`. Read `p7` in those as "GLINT, legacy name". Renaming them is a separate change.
