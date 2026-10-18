---
name: put-model-config-route-was-missing
description: PUT /v1/models/{id}/config had no @app decorator for its whole life, so the WebUI Edit-modal Save always 404'd — restored 2026-10-18.
metadata:
  type: project
  verified: 2026-10-18
---

`update_model_config()` was defined with **no route decorator**, directly below the body of
`interrupt_model` with no blank line separating them. FastAPI never registered it, so
`PUT /v1/models/{model_id}/config` returned `{"detail":"Not Found"}` (404).

**Blast radius:** the endpoint is documented in `README.md` and `DOCUMENTATION.md` and is
what the WebUI **Edit → Save** button calls (`static/app.js` `modal-save`). Every model
config edit from the UI — name, tags, ctx, mmproj filename, pinned/default — silently
failed. Nothing in the test suite covered it, so the suite stayed green for as long as the
bug existed.

**How it was found:** a new e2e test for model-type pinning did `client.put(.../config)` and
got 404. Grepping the route decorators showed only `@app.put("/v1/settings")`.

**Fix:** added `@app.put("/v1/models/{model_id}/config")`. Covered now by
`tests/test_e2e_startup.py::test_model_type_override_roundtrip`.

**Rule:** when adding an endpoint, grep the decorators and count the handlers:

    rg "@app\.(get|post|put|delete|patch)" server.py

and make sure every `async def` that looks like a handler appears in that list. A handler
missing its decorator is invisible to tests and to the docs.

Related: [[webui-appjs-render-drift]]