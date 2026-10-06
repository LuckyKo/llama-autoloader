---
tags: [windows-batch, findstr, netstat, taskkill, gotcha]
aliases: [findstr-substring-port-trap, batch-for-variable-syntax, windows-port-kill-script]
related: [[start-bat-port-wait-relaunch]]
confidence: verified
---

# Windows batch gotchas when writing port-kill scripts (netstat/findstr/taskkill)

Hard-won lessons from writing `start.bat` for llama-autoloader. All verified empirically on this host.

## 1. findstr substring matching defeats "trailing space" port guards
`findstr ":1234 "` does NOT anchor to the end of the port. findstr splits the pattern on whitespace and
matches if ANY token appears, so it effectively becomes a bare substring match on `:1234`.
**It matches `:12345` and `:12340` too.** Verified: a synthetic netstat file with lines for 1234/12345/12340
returned ALL THREE. Do not use trailing-space findstr as a port guard.

**Correct approach:** extract the local-address column and compare exactly:
```bat
for /f "tokens=2,5" %%B in ('netstat -ano ^| findstr "LISTENING"') do (
    if /i "%%B"=="127.0.0.1:1234" ...  :: token2=local addr, token5=PID
)
```

## 2. `%~P` is INVALID inside a FOR loop
`%~P` is batch-parameter substitution syntax (valid for script args `%1`–`%9`), NOT FOR-variable syntax.
Using it inside `for %%P in (...) do ... findstr ":%~P "` errors:
"The following usage of the path operator in batch-parameter substitution is invalid: %~P ]".
Use plain `%%P`.

## 3. `timeout /t N` fails under redirected stdin
In background/redirected contexts, `timeout` prints "Input redirection is not supported, exiting the
process immediately." Use `ping -n <N+1> 127.0.0.1 >nul` for a no-input sleep.

## 4. taskkill /F is async — verify port release before rebind
See [[start-bat-port-wait-relaunch]]. A killed process can keep its socket LISTENING for many seconds.

## 5. Validate batch logic OFFLINE, not against the live service
Testing a kill script by running it against the real running server kills the user's working instance and
causes confusion. Build synthetic netstat output files and exercise the `for /f` + `if` matching logic against
those instead; reserve any live run for a final, explicit, single pass.
