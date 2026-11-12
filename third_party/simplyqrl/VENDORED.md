# Vendored SimplyQRL

**Source:** https://github.com/javier-lazaro/SimplyQRL
**Revision:** `b534cc9` (the lock of June 2025 that produced the published results of the reference study)
**Modified:** no. The Python files are byte-identical to upstream (checked by `tests/test_vendored_integrity.py`, which holds the checksum of every file).
This note and `LICENSE` are the only files added to the directory.

**LICENSE.** `LICENSE` is the MIT licence text of the upstream project (Copyright (c) 2025 Javier Lazaro Gonzalez), kept here because the licence requires the notice to
accompany copies.

Verify at any time:

```bash
git clone https://github.com/javier-lazaro/SimplyQRL.git /tmp/sq
git -C /tmp/sq checkout b534cc9
diff -r third_party/simplyqrl /tmp/sq/src/simplyqrl      # differs only in VENDORED.md and LICENSE
```

## Why it is vendored

The repository of the reference study vendors the library for the same reason: a dependency on a git revision that resolves differently over time
cannot be audited. Vendoring one revision makes the claim "this is the library that produced the results" checkable.

## What this does not mean

This is not a fork. Nothing in this directory is edited. The adjustments that this work needs are applied when `quantumrl_dissection` is imported, in
`src/quantumrl_dissection/training/adjustments.py`, with guards that fail if the library changes shape; they are described in `docs/adjustments.md`. If you need
to change a file here, do not: add a guarded adjustment there and describe it.
