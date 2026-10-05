# Local legacy artifacts

Large generated data that is not tracked by Git was moved out of the repository
root without deleting it:

- `artifacts/legacy_results/`: approximately 35 GB of historical experiment
  results.
- `artifacts/legacy_documents/overleaf_record_2/`: the ignored second Overleaf
  working copy.
- `artifacts/legacy_root_outputs/`: old root traces, reports, logs, animations,
  and temporary files.

`artifacts/` is ignored by Git. These files are therefore not protected by the
repository history. Confirm an external backup before permanently deleting
them.
