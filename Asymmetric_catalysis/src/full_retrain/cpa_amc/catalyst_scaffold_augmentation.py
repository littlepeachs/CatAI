"""Train-only CPA catalyst scaffold perturbations.

The CPA Catalyst test scaffold replaces the two outer aromatic rings of the
training phosphoric-acid scaffold by partially hydrogenated rings.  This
module exposes a deterministic, default-off string transformation that makes
the corresponding H8 view while preserving every catalyst substituent and
stereochemical marker.

It is deliberately independent of ORRA and does not mutate source data.
"""

from __future__ import annotations

import re


_FIRST_OUTER_RING = re.compile(
    r"C=C(?P<ring>%?\d+)C\(C=CC=C(?P=ring)\)"
)
_SECOND_OUTER_RING = re.compile(
    r"=CC(?P<outer>%?\d+)=C(?P<inner>%?\d+)C=CC=C(?P=outer)"
)
_FUSED_SECOND_OUTER_RING = re.compile(
    r"=CC(?P<outer>\d+)=CC=CC=C(?P=outer)(?P<inner>\d+)"
)


def catalyst_h8_scaffold_view(smiles: object) -> tuple[str, int]:
    """Return a partially hydrogenated catalyst view and changed-ring count.

    The transformation is intentionally narrow.  Non-matching strings are
    returned unchanged, so enabling the augmentation cannot alter Imine or
    Thiol inputs or unrelated catalyst families.
    """

    text = "" if smiles is None else str(smiles)
    transformed, first_count = _FIRST_OUTER_RING.subn(
        lambda match: (
            f"C=C{match.group('ring')}C(CCCC{match.group('ring')})"
        ),
        text,
        count=1,
    )
    if first_count == 0:
        return text, 0

    matches = list(_SECOND_OUTER_RING.finditer(transformed))
    if matches:
        match = matches[-1]
        replacement = (
            f"=CC{match.group('outer')}=C{match.group('inner')}"
            f"CCCC{match.group('outer')}"
        )
        transformed = (
            transformed[: match.start()]
            + replacement
            + transformed[match.end() :]
        )
        return transformed, 2

    matches = list(_FUSED_SECOND_OUTER_RING.finditer(transformed))
    if matches:
        match = matches[-1]
        replacement = (
            f"=CC{match.group('outer')}=C{match.group('inner')}"
            f"CCCC{match.group('outer')}"
        )
        transformed = (
            transformed[: match.start()]
            + replacement
            + transformed[match.end() :]
        )
        return transformed, 2

    return transformed, 1
