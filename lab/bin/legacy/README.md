Old command names from the minor's setup, moved here on 2026-11-20 (nothing in the current workflow uses them):
- aliases that forward to the `lab` command: analyze (lab results), list, record, run, server (lab serve), stop, train;
- stubs of tools that were never ported (they only print a message): ablate-cameras, bench, edit-dataset,
  eval-offline (now `lab eval offline`), h200-setup (the old drtc_lab container setup), logstats, tunnel (now built
  into `lab eval servers/run`), venv.
