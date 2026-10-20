---
tags: [llama-autoloader, import-time-config, scratch-scripts, cwd]
aliases: [server-import-validates-config, config-yaml-at-import]
related: [[llama-autoloader-pytest-canonical-invocation]]
confidence: verified
author: reviewer_lru-eviction-review_20261020_155658.jsonl
---

# `import server` validates `config.yaml` at import time — and resolves it CWD-relative

Any scratch script that does `from server import ModelManager` **runs the production
config loader and validator at import time**. It is not lazy, and it is not
overridable by the dict you later pass to `ModelManager(cfg)`.

At `llama-autoloader` @ `3926acecf917796ed2fbf2a0073d97685a863219`, module-level
`server.py` executes `ModelManager.validate_config(CFG)` (symbol: `validate_config`,
called from the module body near `server.py:3492`, raising into a
`RuntimeError(f"Failed to load config from '{CONFIG_PATH}'...")` wrapper).

## The trap

`CONFIG_PATH` is **relative to the current working directory**, not to `server.py`.
So running a scratch file placed inside the repo from elsewhere fails with a message
that points at the wrong thing:

```
$ python N:\work\stuff\Beta\llama-autoloader\temp\scratch.py
ValueError: Config missing 'launcher' section
... RuntimeError: Failed to load config from 'config.yaml': Config missing 'launcher' section
```

The repo's `config.yaml` **does** have a `launcher:` section — the loader simply found
a different (or no) file. Reading the error as "config.yaml is malformed" sends you
editing a file that was never the problem.

## Fix

Run with the repo root as cwd, and make the script's own `sys.path` insertion
absolute (it must not depend on cwd either):

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

`tests/conftest.py` does the same thing deliberately (`_REPO_ROOT` insert) — that is
why pytest imports work from any cwd while a bare `python scratch.py` does not.

## Why this matters for reviews

Verifying a behaviour change in this repo usually needs a scratch script (see
[[wrong-shape-double-false-pass]]: build an execution witness, then delete the
script). Budget one wasted run for this, or set cwd from the start. Note that
`ModelManager(cfg)` itself takes an explicit dict, so the import-time validation is
purely a module-body side effect — nothing about your test config is at fault.

## Related

- [[llama-autoloader-pytest-canonical-invocation]] — the pytest-side collection gotcha;
  same repo, different failure mode.
