# Standing rule: the dependency stack is pinned (2026-08-22)

Decided: 2026-08-22. Status: accepted. Moved verbatim from `LAB_NOTES.md` on 2026-09-04.


`setup.sh` used to install `trl>=0.26` plus a bare list of package names, so
every fresh environment resolved whatever was newest that week, and the OpenEnv
clone floated on upstream HEAD. Nothing recorded which versions a run trained
against - e9 through e36 all sit on "whatever was installed at the time". That
is the same failure mode as an unrecorded `batch_size`: a library changes, the
numbers move, and nothing on disk says why.

Since 2026-08-22 `requirements.lock.txt` is the only install source and
`pipeline/OPENENV_COMMIT` pins the clone. The live values are in those two files;
do not copy them here, or this section becomes a second, stale answer to the same
question. What e30-e36 trained on was trl 1.6.0 / transformers 5.12.0 / torch
2.10.0+cu130 / vllm 0.19.1+cu130, on OpenEnv `024eedc` - recorded here because
those runs predate the stamp file and nothing else holds it.

Upgrading is a deliberate act, not a side effect of re-running setup: install,
verify against a real reset, then re-freeze the lock in the same commit. A
version bump landing silently between two arms of a comparison confounds them
exactly like a geometry change would. Runs before 2026-08-22 have no recorded
stack; that is a caveat on their reproducibility, not on their numbers.

Two things now enforce the pin outside setup, because setup only runs when
someone runs it: a launch refuses to serve a run from a clone that has moved off
`pipeline/OPENENV_COMMIT`, and each phase writes `runs/<exp>/env_stamp.json` with
the clone HEAD and the load-bearing package versions it actually had. From e37 on,
the run directory answers "which stack produced this" on its own.
