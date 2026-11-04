---
name: route-decorator-inside-another-endpoint-body
description: "A new endpoint can be syntactically valid, greppable and unit-test-green yet never registered if its @app decorator lands inside another handler's body; prove registration with the live app route table plus an AST enclosing-function check."
metadata:
  type: project
  verified: 2026-11-04
---

On 2026-11-04 the `POST /v1/restart` endpoint (`server.py:5530`) was inserted at **4-space
indentation**, i.e. nested inside `unload_all_ep()` (`server.py:5522-5545`). Every path of
`unload_all_ep` returns at 5527/5528, so the nested `def`/decorator is unreachable dead code and
FastAPI never registers the route. The `# ---------- state save / load ----------` section comment
(line 5547) was swallowed into the function body at the same time — a good corroborating tell that
a whole block got shifted one indent level.

**Why the normal checks all said "fine":**
- `syntax_check server.py` -> valid (a nested `def` is legal Python).
- `grep '@app.post("/v1/restart")' server.py` -> found, with no indication of nesting.
- `ruff check` -> clean.
- `tests/test_server_restart.py` -> 3 passed, because it exercises only the helper module and never
  imports `server`.

**Decisive check (do this for every added endpoint):**

```python
python -c "import server; print(sorted(r.path for r in server.app.routes))"
```

`/v1/restart` was absent; only the unload routes appeared. Second check, which localizes the fault
without eyeballing indentation:

```python
import ast
tree = ast.parse(open("server.py", encoding="utf-8").read())
for n in ast.walk(tree):
    if isinstance(n, ast.FunctionDef) and n.lineno <= 5530 <= n.end_lineno:
        print("line 5530 is inside", n.name, n.lineno, "-", n.end_lineno)
# -> line 5530 is inside unload_all_ep 5522 - 5545
```

**Rule:** a route decorator whose line falls inside a `FunctionDef` body is a bug until proven
otherwise. Also add a one-line registration assertion to the suite:
`assert "/v1/restart" in {r.path for r in server.app.routes}`.

Related: [[put-model-config-route-was-missing]] (the inverse failure mode — handler defined with
no decorator at all; same invisibility to tests and docs).
