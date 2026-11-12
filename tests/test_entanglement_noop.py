"""the final-ring property: the `ent` flag cannot affect a `skolik` circuit at n_layers_q = 1.

Why this needs a test rather than a paragraph
---------------------------------------------
`build_skolik_qlayer` closes each layer with a circular ring of CZ gates, and the
circuit is measured in the PauliZ basis. CZ is diagonal, and diagonal unitaries
commute with Z, so the entangling block of the FINAL layer can never change
`<Z_i>`. At depth 1 that is the only entangling block, and the flag is a complete
no-op; at depth L only L-1 blocks do anything.

Two consequences the repo depends on:

  * The paper's `Skolik_DR_L1_Entangled` vs `Skolik_DR_L1_Unentangled` contrast
    compares a circuit against itself, which is why their logged returns are
    identical on all 10 seeds. That is an experimental-design artefact, NOT
    duplicated data - and this test is the evidence for the distinction.

  * the depth sweep covers depth on the Skolik template with `ent=True`, so its depth
    axis is confounded with effective entanglement depth (0, 1 and 4 blocks at
    L = 1, 2, 5). See docs/reproducibility.md, Experiment 03.

The Hsiao template is unaffected - its ablation is valid at every depth - so the
test also pins that contrast, to keep the claim precise rather than sweeping.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from simplyqrl.qlayers import (  # noqa: E402
    build_dr_qlayer,
    build_hsiao_qlayer,
    build_skolik_qlayer,
)


def _max_abs_diff(builder, n_qubits, n_layers, **kwargs) -> float:
    """Same weights, entanglement on vs off, largest output difference."""
    torch.manual_seed(0)
    x = torch.rand(8, 4) * 2 - 1
    ent_on = builder(n_qubits, n_layers, ent=True, **kwargs)
    ent_off = builder(n_qubits, n_layers, ent=False, **kwargs)
    with torch.no_grad():
        for p_on, p_off in zip(ent_on.parameters(), ent_off.parameters()):
            w = torch.rand_like(p_on) * 2 * np.pi
            p_on.copy_(w)
            p_off.copy_(w)
        return float((ent_on(x) - ent_off(x)).abs().max())


def test_skolik_entanglement_is_a_noop_at_depth_one():
    assert _max_abs_diff(build_skolik_qlayer, 4, 1) == 0.0, (
        "entanglement changed the output at depth 1 - upstream may have moved "
        "the CZ ring, in which case the final-ring property needs revisiting"
    )


@pytest.mark.parametrize("n_layers", [2, 5])
def test_skolik_entanglement_matters_beyond_depth_one(n_layers):
    """The claim is narrow on purpose: only the LAST block is wasted."""
    assert _max_abs_diff(build_skolik_qlayer, 4, n_layers) > 1e-6


@pytest.mark.parametrize("n_layers", [1, 2])
def test_hsiao_entanglement_is_never_a_noop(n_layers):
    """Contrast arm: the paper's Hsiao ablation is valid at every depth."""
    assert _max_abs_diff(build_hsiao_qlayer, 4, n_layers,
                         emb_indices=[1, 2, 3]) > 1e-6


@pytest.mark.parametrize("n_layers", [1, 2])
def test_single_qubit_dr_cannot_entangle(n_layers):
    """Trivially true, and the paper says so in its own comments - pinned so the
    Salinas 1Q rows are never read as an entanglement contrast."""
    assert _max_abs_diff(build_dr_qlayer, 1, n_layers,
                         emb_indices=[1, 2, 3]) == 0.0


def test_paper_l1_entanglement_rows_are_identical_as_predicted():
    """Ties the circuit fact to the published data.

    If this fails, either the extraction changed or the two rows stopped
    matching - both worth knowing, because the whole the final-ring property argument rests on
    the coincidence being explained rather than suspicious.
    """
    pd = pytest.importorskip("pandas")
    from quantumrl_dissection.analysis.baselines import DATA_DIR

    csv = DATA_DIR / "paper_ppo_baselines.csv"
    if not csv.exists():
        pytest.skip("paper baselines not extracted")
    df = pd.read_csv(csv)
    ent = df[df.config == "Skolik_DR_L1_Entangled"].set_index("seed").best_ma50
    unent = df[df.config == "Skolik_DR_L1_Unentangled"].set_index("seed").best_ma50
    assert len(ent) == len(unent) == 10
    assert (ent == unent).all(), "L1 ent/unent returns diverged - re-check the final-ring property"
