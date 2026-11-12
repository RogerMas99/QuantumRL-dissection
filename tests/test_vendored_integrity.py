"""The vendored SimplyQRL must stay byte-identical to upstream b534cc9.

This test exists because the rule was broken within minutes of being written. A
bulk rename of `core/envs.py` -> `core/obs_adapters.py` matched
`from .envs import make_vec_env` inside `third_party/simplyqrl/dqn.py` and `ppo.py` and
rewrote the vendored library. Nothing failed loudly: the installed copy shadowed
the vendored one, so the whole suite still passed.

That is precisely the failure mode this repository exists to prevent. The value
of vendoring is the claim "this is upstream, unmodified, and every correction is
visible separately in `quantumrl_dissection/`". An unnoticed edit turns that claim into
a lie, and every correction documented in `docs/adjustments.md` becomes
unverifiable, because the reader can no longer tell what upstream did from what
we did to it.

The checksums below are of the pristine `b534cc9` tree. If one fails, either the
vendored copy was edited - revert it - or the pin moved, in which case every
correction needs re-verifying against the new revision and these hashes updated
deliberately, in their own commit.
"""
import hashlib
import pathlib

import pytest

VENDOR = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "simplyqrl"

# sha256 of each file in upstream SimplyQRL at revision b534cc9.
EXPECTED = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "agents.py": "45317ae219cde6c6d3ed6f16de84914248cc603e75bf5d0494b50513f742c1d7",
    "buffers.py": "f416b9cbbf1cfa178e8a6bb9caf9e3ffcb230270328c4872a70fa5853501b715",
    "dqn.py": "70c9aab739574ef2ac57ac9ee7bbcaa630a388f44fcc0b4dffd90648577118a8",
    "embeddings.py": "43a420b73aa40ed8bd7abb0f147cfbecb54da341d68aecf7cc68eca9bb0c68fa",
    "envs.py": "5e6d3a053ce06f23be8b9e783537b18727aec13bd54d68f0dc6a73ab95afb64c",
    "ppo.py": "bd452c1413b3b5249b9f6d327c939c5d058ce681b53da5bdcf2231368f48f0ae",
    "qlayers.py": "8ed6434799dd30dd0cd23c50e551515d9382c097ea6e6af8dda3633cd4a78de7",
    "transformations.py": "ca77e06f30186432ea5b01f0a67f3792a20374cc1838df9224e9b4db14d754f8",
}


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendor_directory_exists():
    assert VENDOR.is_dir(), f"vendored SimplyQRL missing at {VENDOR}"
    assert (VENDOR / "VENDORED.md").exists(), "provenance note missing"


def test_no_vendored_module_imports_from_quantumrl_dissection():
    """The dependency arrow points one way only.

    `quantumrl_dissection` may import `simplyqrl`. The reverse would mean the vendored
    copy is no longer upstream, and would make the correction registry
    meaningless.
    """
    offenders = []
    for py in sorted(VENDOR.glob("*.py")):
        text = py.read_text()
        if "quantumrl_dissection" in text:
            offenders.append(py.name)
    assert not offenders, (
        f"vendored files reference quantumrl_dissection: {offenders}. "
        "The vendored tree must remain unmodified upstream code."
    )


def test_vendored_modules_import_their_own_siblings():
    """Catches exactly the bulk-rename accident described in this file's docstring.

    `dqn.py` and `ppo.py` import `make_vec_env` from SimplyQRL's own `envs`
    module. Our observation adapters live in `quantumrl_dissection.core.obs_adapters`
    and are deliberately named differently - if these imports ever point at
    `obs_adapters`, a rename has leaked into vendored code.
    """
    for name in ("dqn.py", "ppo.py"):
        text = (VENDOR / name).read_text()
        assert "from .envs import make_vec_env" in text, (
            f"{name} no longer imports make_vec_env from simplyqrl.envs - "
            "the vendored copy has been edited"
        )
        assert "obs_adapters" not in text, (
            f"{name} references obs_adapters - a rename leaked into vendored code"
        )


def test_expected_module_set():
    """The file list itself is part of the pin: an added file is a modification."""
    found = {p.name for p in VENDOR.glob("*.py")}
    expected = {
        "__init__.py", "agents.py", "buffers.py", "dqn.py", "embeddings.py",
        "envs.py", "ppo.py", "qlayers.py", "transformations.py",
    }
    assert found == expected, (
        f"vendored module set changed. missing={expected - found} "
        f"unexpected={found - expected}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_checksums(name):
    path = VENDOR / name
    if not path.exists():
        pytest.skip(f"{name} absent")
    assert _sha(path) == EXPECTED[name], (
        f"{name} differs from upstream b534cc9. Revert it, or - if the pin moved "
        "deliberately - re-verify every entry in docs/adjustments.md against the "
        "new revision and update these hashes in their own commit."
    )
