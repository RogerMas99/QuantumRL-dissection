"""
Provenance of a result: was it produced by the REAL circuit or by a simulator?

Anything produced by ``SU2SkolikEmulator`` (agent_type ``su2``) or by the batched state-vector simulator (agent_type ``sv``) is a
result computed with a SIMULATOR of the circuit. It is reported apart from the results of the real PennyLane circuit (agent_type
``hybrid``) and of the classical arms, which are not affected, and it is never presented as a result of the real circuit.

This module is the single place that knows which agent types are simulators. It is used three ways:

* ``run_arm`` stamps a ``provenance`` block into every manifest produced by a simulator;
* ``analysis.loaders.summarise_run`` / ``to_dataframe`` carry a ``simulated`` column (derived from the agent type) and warn when a
  table contains simulated rows;
* ``exclude_simulated`` is the guard to put in front of any table that is meant for the real circuit: it drops simulated rows
  unless ``approved=True`` is passed on purpose.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional

SIMULATED_AGENT_TYPES = frozenset({"su2", "sv"})

STATUS = "SIMULATED - computed with a simulator of the circuit, reported apart from the results of the real circuit"
POLICY_DOC = "docs/reproducibility.md"
NOTICE = (
    "SIMULATOR RESULTS PRESENT (agent_type su2/sv). " + STATUS + ". "
    "See " + POLICY_DOC + "."
)


def is_simulated_agent_type(agent_type: Optional[str]) -> bool:
    return str(agent_type).lower() in SIMULATED_AGENT_TYPES


def is_simulated_manifest(manifest: Dict[str, Any]) -> bool:
    """True for a manifest produced by a simulator, stamped or not."""
    if (manifest.get("provenance") or {}).get("simulated"):
        return True
    return is_simulated_agent_type(manifest.get("agent_type"))


def is_simulated_arm(arm: str) -> bool:
    """True if the registered arm runs on a simulator (looked up in `ARMS`)."""
    from quantumrl_dissection.arms import ARMS

    entry = ARMS.get(arm)
    return bool(entry) and is_simulated_agent_type(entry[0])


def provenance_block(agent_type: str) -> Optional[Dict[str, Any]]:
    """The block `run_arm` writes into a manifest; None for non-simulator agents."""
    if not is_simulated_agent_type(agent_type):
        return None
    return {
        "simulated": True,
        "simulator": str(agent_type).lower(),
        "status": STATUS,
        "policy": POLICY_DOC,
    }


def warn_if_simulated(df) -> None:
    """Emit one loud warning if a DataFrame carries simulated rows."""
    if "simulated" in getattr(df, "columns", []) and bool(df["simulated"].any()):
        warnings.warn(NOTICE, stacklevel=3)


def exclude_simulated(df, approved: bool = False):
    """Guard for tables meant for the final work.

    Drops the rows produced by a simulator unless `approved=True`. Passing `approved=True`
    states that the simulated rows are meant to be there; it is never a default.
    """
    if approved or "simulated" not in getattr(df, "columns", []):
        return df
    return df[~df["simulated"]].copy()


def simulated_arms(arms: Iterable[str]):
    return [a for a in arms if is_simulated_arm(a)]
