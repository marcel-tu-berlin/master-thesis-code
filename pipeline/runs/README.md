# Historical run evidence

Everything under this directory is completed experiment evidence, not a current
or planned run. Stable `runs/<experiment_id>/` paths are preserved because the
findings, paired analyses and overwrite guards refer to them.

`RUNNING.md` at the repository root is the only source of live GPU state.
Planned experiment configs belong in `pipeline/configs/`; finished and
superseded configs belong in `pipeline/configs/archive/`.
