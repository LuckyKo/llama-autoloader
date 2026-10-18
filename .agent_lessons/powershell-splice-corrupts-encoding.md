---
name: powershell-splice-corrupts-encoding
description: Get-Content | Set-Content in Windows PowerShell 5.1 silently mangles UTF-8 in static/app.js — double-encodes every non-ASCII char, adds a BOM and flips LF to CRLF.
metadata:
  type: project
  verified: 2026-10-18
---

I replaced a ~144-line block of `static/app.js` by splicing with PowerShell:

```powershell
$c   = Get-Content static/app.js          # <-- reads UTF-8 bytes as cp1252
$out = $c[0..263] + $new + $c[409..($c.Count-1)]
Set-Content static/app.js -Value $out -Encoding UTF8
```

`node --check` still passed and the whole pytest suite was green — the file is *valid*
UTF-8, it is the **mojibake** that is valid UTF-8. Damage:

| Property | HEAD | After the splice |
|---|---|---|
| BOM | none | added |
| Line endings | LF | CRLF |
| Emoji (👁 x4) | `U+1F441` | `U+00F0 U+009F U+2018 U+0081` |
| Em-dash / ellipsis / middle dot | `U+2014 U+2026 U+00B7` | `U+00E2 U+0080 U+0094 ...` |
| Everything else | ASCII | ASCII |

**Why no test caught it:** every check available at the time (syntax check, unit tests,
live `/health` + `/v1/models` + WebUI 200) passes on a mojibake'd file. Only reading the
rendered UI or diffing codepoints shows it.

**Detect it** by comparing codepoint sets, not by eyeballing the console:

```python
import subprocess
base = subprocess.run(["git","cat-file","blob","HEAD:static/app.js"], capture_output=True).stdout.decode("utf-8")
cur  = open("static/app.js", "rb").read()
print("BOM:", cur[:3] == b"\xef\xbb\xbf", "CRLF:", cur.count(b"\r\n"))
```

Watch out: `git show HEAD:f > tmp` is **also** broken — PowerShell's `>` re-encodes as
UTF-16, so the baseline comes back as `0xFF 0xFE` and every comparison is garbage. Use
`git cat-file blob` with `capture_output=True` (binary-safe).

**Recovery** is the inverse transform: re-encode the mis-decoded text back to cp1252
bytes, then decode those as UTF-8. Python's `cp1252` codec leaves five slots UNDEFINED
(`0x81 0x8D 0x8F 0x90 0x9D`) which Windows maps to the matching control codepoint, so
build the inverse map byte-by-byte instead of calling `str.encode("cp1252")`:

```python
BACK = {}
for b in range(256):
    try:    BACK[bytes([b]).decode("cp1252")] = b
    except UnicodeDecodeError: BACK[chr(b)] = b
repaired = bytearray(BACK[ch] for ch in text).decode("utf-8")
```

Always **verify the recovered non-ASCII set against HEAD and abort on any loss** before
writing — the one legitimate difference is a character inside code you deliberately
replaced (here `U+2192` lived only in the old `renderModels()`).

**Rule: never edit a UTF-8 source file with PowerShell `Get-Content`/`Set-Content` on this
box.** Use the editor tool, or Python with an explicit `encoding="utf-8"` and
`newline="\n"`.

Related: [[webui-appjs-render-drift]], [[webui-appjs-robustness-fixes]]