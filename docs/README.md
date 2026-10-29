# docs/

Reference material for the package.  Everything here is documentation only; no script
reads these files at run time.

| File | What it is |
|---|---|
| `column_semantics.csv` | One row per dataset, listing how each raw column maps onto the ten semantic slots. Machine-readable form of the **slot** decisions that live in `data/col_mappings.csv`; that table also carries the SMILES pass-through rows and the alias priorities, which are not slots. |
| `reaction_roles.csv` | One row per dataset, describing how the reactant columns are resolved (which column supplies `reactant_1`, which `reactant_2`) and the reaction class. |
| `NUMBERS_AND_PIPELINE.md` | Where every manuscript number comes from (producing script → table) and what `run_all.bat` does, step by step. **Start here.** |
| `NUMBER_UPDATES_20261029.md` | The 2026-10-29 number-sync ledger: every value that moved between the manuscript tables and this repository, old → new, with the file that produced it. |
| `NUMBERS_PROVENANCE.md` | The longer historical provenance note (how each table was first rebuilt). Superseded by `NUMBERS_AND_PIPELINE.md` for day-to-day use. |
| `RELEASING.md` | Checklist for cutting a release (what ships, what is stripped, Zenodo metadata). |

Licence information is **not** here: see `LICENSE` (code) and `LICENSE-DATA` (datasets,
with the per-source table and the preparation each one needs) at the repository root.

The two `.csv` files are the readable form of the column decisions; the Markdown
versions that used to sit next to them were removed because the same facts read better
as a table.
