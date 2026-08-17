---
tags: [uvicorn, logging, streaming, tail-sync]
aliases: [uvicorn-streamhandler-eager-binding, uvicorn-default-logging-config]
related: []
confidence: verified
---

# Uvicorn Logging Architecture: Eager Stream Binding and Propagation Behavior

## Key Finding 1: StreamHandler Binds to sys.stderr/stdout at Configuration Time

**Fact:** When `uvicorn.run()` is called with default `log_config`, it creates `StreamHandler` instances via `logging.config.dictConfig()`. These handlers store a reference to the stream object **as it exists at configuration time**, not at log emit time.

**Evidence from Uvicorn source (`config.py`):**
- Default `LOGGING_CONFIG` defines handlers with `"stream": "ext://sys.stderr"` and `"stream": "ext://sys.stdout"` (lines 85, 90)
- `Config.configure_logging()` calls `logging.config.dictConfig(self.log_config)` (line 366)
- Empirical testing confirms: if `sys.stderr` is replaced **after** `dictConfig()`, the handler writes to the original stream; if replaced **before**, it writes to the replacement stream.

**Implication for llama-autoloader:** Simply replacing `sys.stdout`/`sys.stderr` before calling `uvicorn.run()` will NOT capture Uvicorn's logs unless the replacement happens **before** any `Config` object is instantiated (i.e., before `uvicorn.run()`). However, because `uvicorn.run()` creates the `Config` internally, the safest approach is to use a tee stream pattern.

## Key Finding 2: Uvicorn Default Loggers Have propagate=False

**Fact:** Uvicorn's default logging configuration explicitly sets `propagate=False` on all its loggers:
```python
"loggers": {
    "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
    "uvicorn.error": {"level": "INFO"},  # inherits from root unless configured
    "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
}
```

**Evidence:** `config.py` lines 93-97 (default `LOGGING_CONFIG`).

**Implication:** Adding a custom `StreamHandler` to the root logger will **not** cause duplication of Uvicorn's banner, access, or error logs. However, application code that does not explicitly set `propagate=False` on its own loggers may still propagate to root.

## Key Finding 3: log_config=None Disables Uvicorn's Default Handlers

**Fact:** When `log_config=None` is passed to `uvicorn.run()`, the `configure_logging()` method skips all configuration and does not install any handlers. The logging system remains in whatever state existed prior to the call.

**Evidence:** `config.py` lines 361-382:
```python
def configure_logging(self) -> None:
    logging.addLevelName(TRACE_LOG_LEVEL, "TRACE")
    if self.log_config is not None:
        # ... configures handlers via dictConfig or fileConfig
```

**Implication:** If full control over logging is desired, `log_config=None` can be used, but then the application must handle all console output via custom stream replacement.

## Recommended Pattern for Capturing All Output

To guarantee ALL console output (Uvicorn banner + access logs + app logs + print statements) goes to both terminal and rotating log file without duplication:

1. **Create a `TeeStream`** that writes to multiple destinations.
2. **Replace `sys.stdout` and `sys.stderr` BEFORE creating any `Config` or calling `uvicorn.run()`**.
3. Optionally, use `log_config=None` if you want to avoid Uvicorn's default logging entirely.

This pattern works because the tee stream becomes the "original" stream that Uvicorn's handlers bind to at configuration time.

## Source References

- Uvicorn LOGGING_CONFIG: `C:\Python312\lib\site-packages\uvicorn\config.py` lines 67-98
- Config.__init__: line 274 calls `self.configure_logging()`
- configure_logging method: lines 358-394
