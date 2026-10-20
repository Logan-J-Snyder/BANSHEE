# Contributing

Open an issue with the command, Python/dependency versions, column names/units,
and the per-file `summary.json` error. Remove tokens, private paths, and unpublished
data before posting. A small synthetic reproducer is better than a full catalogue.

1. Branch from `main`; install with `python -m pip install -e ".[dev]"`.
2. Add a focused regression test; run `python -m pytest`.
3. Separate bug fixes, numerical changes, and threshold changes.
4. Document changes to priors, preprocessing, flags, outputs, or units.
5. Support scientific claims with paired injections, a frozen protocol, fresh
   held-out confirmation, and failed inputs/null controls retained in denominators.

Do not commit FITS, catalogues, credentials, large validation outputs, or private
server settings. Passing a matching-template test does not establish real-data
performance. This project is all rights reserved; obtain permission and discuss
reuse/contribution terms with Logan Snyder before using the code or making
substantial contributions. See [LICENSE](LICENSE). Submitting a contribution
does not automatically transfer its copyright to the maintainer.
