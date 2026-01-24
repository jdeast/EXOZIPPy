"""
model_types.py

Definitions of model-type enums, ModelKey, and label<->ModelKey conversion.

Label grammar (current conventions)
-----------------------------------
<base> ::= "PSPL" | "FSPL" | "2L1S" | ...

<modifier sequence> (separated by spaces) encodes parallax and (for binaries) orbital motion:

    "<base> static"
        parallax_branch = none, lens_orb_motion = none

    "<base> par <branch>"
        parallax_branch = <branch>, lens_orb_motion = none

    "<base> 2Dorb par <branch>"
        parallax_branch = <branch>, lens_orb_motion = ORB_2D

    "<base> kep par <branch>"
        parallax_branch = <branch>, lens_orb_motion = KEP

Also, as a convenience, "<base>" alone is treated as "<base> static".

Examples
--------
    "PSPL static"
    "FSPL static"
    "PSPL par u0+"
    "FSPL par u0-"
    "2L1S static"
    "2L1S par u0+"
    "2L1S 2Dorb par u0+"
    "2L1S kep par u0--"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


# ============================================================================
# Enums
# ============================================================================


class LensType(Enum):
    POINT = "point"
    BINARY = "binary"   # e.g. 2L1S


class SourceType(Enum):
    POINT = "point"
    FINITE = "finite"   # for finite-source PSPL/FSPL, etc.


class ParallaxBranch(Enum):
    NONE = "none"
    U0_PLUS = "u0+"
    U0_MINUS = "u0-"
    U0_PP = "u0++"
    U0_MM = "u0--"
    U0_PM = "u0+-"
    U0_MP = "u0-+"


class LensOrbMotion(Enum):
    NONE = "none"
    ORB_2D = "2Dorb"    # 2L1S 2-parameter orbital motion
    KEP = "kep"         # 2L1S full Keplerian orbital motion
    # extendable


# ============================================================================
# ModelKey
# ============================================================================


@dataclass(frozen=True)
class ModelKey:
    lens_type: LensType
    source_type: SourceType
    parallax_branch: ParallaxBranch
    lens_orb_motion: LensOrbMotion

    def __post_init__(self) -> None:
        # Enforce: point lens cannot have orbital motion.
        if self.lens_type == LensType.POINT and self.lens_orb_motion is not LensOrbMotion.NONE:
            raise ValueError("Point lenses must have lens_orb_motion == NONE")


# ============================================================================
# Tag mapping tables
# ============================================================================

# Base model tag → lens_type / source_type
# Extend these dictionaries if you introduce new base model types.
SOURCE_TAGS: Dict[str, SourceType] = {
    "PSPL": SourceType.POINT,
    "FSPL": SourceType.FINITE,
    "2L1S": SourceType.POINT,   # binary lens with single source
}

LENS_TAGS: Dict[str, LensType] = {
    "PSPL": LensType.POINT,
    "FSPL": LensType.POINT,
    "2L1S": LensType.BINARY,
}

PARALLAX_BRANCH_TAGS: Dict[str, ParallaxBranch] = {
    "none": ParallaxBranch.NONE,
    "u0+": ParallaxBranch.U0_PLUS,
    "u0-": ParallaxBranch.U0_MINUS,
    "u0++": ParallaxBranch.U0_PP,
    "u0--": ParallaxBranch.U0_MM,
    "u0+-": ParallaxBranch.U0_PM,
    "u0-+": ParallaxBranch.U0_MP,
}

LENS_MOTION_TAGS: Dict[str, LensOrbMotion] = {
    "none": LensOrbMotion.NONE,
    "2Dorb": LensOrbMotion.ORB_2D,
    "kep": LensOrbMotion.KEP,
}


# ============================================================================
# Label <-> ModelKey conversion
# ============================================================================


def label_to_model_key(label: str) -> ModelKey:
    """
    Parse a human-readable label into a ModelKey.

    Expected forms (tokens separated by spaces):

        "<base> static"
        "<base> par <branch>"
        "<base> 2Dorb par <branch>"
        "<base> kep par <branch>"

    As a convenience, "<base>" alone is treated as "<base> static".

    Parameters
    ----------
    label : str

    Returns
    -------
    ModelKey

    Raises
    ------
    ValueError
        If the label cannot be parsed or refers to unknown tags.
    """
    tokens = label.strip().split()
    if not tokens:
        raise ValueError("Empty model label")

    base = tokens[0]

    # Defaults
    parallax_branch = ParallaxBranch.NONE
    lens_orb_motion = LensOrbMotion.NONE

    # 'PSPL static' or '2L1S static'
    if len(tokens) == 2 and tokens[1] == "static":
        # parallax_branch and lens_orb_motion remain NONE
        pass

    # Bare 'PSPL' etc → treat as static
    elif len(tokens) == 1:
        # parallax_branch and lens_orb_motion remain NONE
        pass

    else:
        # More complex forms:
        #   "<base> par u0+"
        #   "<base> 2Dorb par u0+"
        #   "<base> kep par u0--"
        tail = tokens[1:]

        # Case 1: 'par <branch>'
        if tail[0] == "par":
            lens_motion_token = "none"
            branch_token = tail[1] if len(tail) > 1 else "none"

        # Case 2: '<motion> par <branch>'
        else:
            lens_motion_token = tail[0]                # '2Dorb' or 'kep'
            if len(tail) < 2 or tail[1] != "par":
                raise ValueError(
                    f"Cannot parse label {label!r}: "
                    f"expected 'par' after lens motion token {lens_motion_token!r}"
                )
            branch_token = tail[2] if len(tail) > 2 else "none"

        lens_orb_motion = LENS_MOTION_TAGS.get(lens_motion_token)
        if lens_orb_motion is None:
            raise ValueError(
                f"Unknown lens orbital motion token in label {label!r}: {lens_motion_token!r}"
            )

        parallax_branch = PARALLAX_BRANCH_TAGS.get(branch_token)
        if parallax_branch is None:
            raise ValueError(
                f"Unknown parallax branch token in label {label!r}: {branch_token!r}"
            )

    # Now decode lens_type and source_type from the base tag
    lens_type = LENS_TAGS.get(base)
    source_type = SOURCE_TAGS.get(base)

    if lens_type is None or source_type is None:
        raise ValueError(
            f"Unknown base model tag in label {label!r}: {base!r}"
        )

    return ModelKey(
        lens_type=lens_type,
        source_type=source_type,
        parallax_branch=parallax_branch,
        lens_orb_motion=lens_orb_motion,
    )


def model_key_to_label(key: ModelKey) -> str:
    """
    Map a ModelKey back to a human-readable label following the same
    conventions as label_to_model_key().

    Rules
    -----
    - Choose a base tag that matches (lens_type, source_type).
    - If parallax_branch == NONE and lens_orb_motion == NONE -> "<base> static"
    - If lens_orb_motion == NONE and parallax_branch != NONE -> "<base> par <branch>"
    - If lens_orb_motion != NONE and parallax_branch != NONE ->
          "<base> <motion> par <branch>"

    Parameters
    ----------
    key : ModelKey

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If no suitable base tag or motion/branch tag is found.
    """
    # Find a base tag that matches lens_type+source_type
    base = None
    for candidate, lt in LENS_TAGS.items():
        if lt == key.lens_type and SOURCE_TAGS[candidate] == key.source_type:
            base = candidate
            break

    if base is None:
        raise ValueError(f"No base label mapping for ModelKey {key!r}")

    # Static
    if (
        key.parallax_branch == ParallaxBranch.NONE
        and key.lens_orb_motion == LensOrbMotion.NONE
    ):
        return f"{base} static"

    # Find motion label
    lens_motion_label = None
    for lbl, motion in LENS_MOTION_TAGS.items():
        if motion == key.lens_orb_motion:
            lens_motion_label = lbl
            break
    if lens_motion_label is None:
        raise ValueError(
            f"No lens motion label mapping for LensOrbMotion {key.lens_orb_motion!r}"
        )

    # Find branch label
    branch_label = None
    for lbl, br in PARALLAX_BRANCH_TAGS.items():
        if br == key.parallax_branch:
            branch_label = lbl
            break
    if branch_label is None:
        raise ValueError(
            f"No parallax branch label mapping for {key.parallax_branch!r}"
        )

    if lens_motion_label == "none":
        # Pure parallax, no orbital motion: "<base> par u0+"
        return f"{base} par {branch_label}"

    # Orbital motion + parallax: "<base> 2Dorb par u0+"
    return f"{base} {lens_motion_label} par {branch_label}"

'''

# ============================================================================
# Enums and ModelKey
# ============================================================================

class LensType(Enum):
    POINT = "point"
    BINARY = "binary"   # for 2L1S, etc.


class SourceType(Enum):
    POINT = "point"
    FINITE = "finite"


class ParallaxBranch(Enum):
    NONE = "none"
    U0_PLUS = "u0+"
    U0_MINUS = "u0-"
    U0_PP = "u0++"
    U0_MM = "u0--"
    U0_PM = "u0+-"
    U0_MP = "u0-+"


class LensOrbMotion(Enum):
    NONE = "none"
    ORB_2D = "2Dorb"  # 2L1S 2-parameter orbital motion
    KEP = "kep"       # 2L1S Full Keplerian orbital motion
    # extendable


@dataclass(frozen=True)
class ModelKey:
    lens_type: LensType
    source_type: SourceType
    parallax_branch: ParallaxBranch
    lens_orb_motion: LensOrbMotion

    def __post_init__(self):
        # Enforce: point lens cannot have orbital motion.
        if (
            self.lens_type == LensType.POINT
            and self.lens_orb_motion is not LensOrbMotion.NONE
        ):
            raise ValueError("Point lenses must have lens_orb_motion == NONE")


# ============================================================================
# Conversion Functions
# ============================================================================

def label_to_model_key(label: str) -> ModelKey:
    """
            Map user-facing string labels (e.g. 'static PSPL', 'PL par u0+')
            to structured ModelKey.

            TODO: Implement mapping rules that correspond to your current labels.
            """
    if label == "static PSPL":
        return ModelKey(
            lens_type=LensType.POINT,
            source_type=SourceType.POINT,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )
    if label == "static FSPL":
        return ModelKey(
            lens_type=LensType.POINT,
            source_type=SourceType.FINITE,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )

    if label == "PL par u0+":
        return ModelKey(
            lens_type=LensType.POINT,
            source_type=(
                SourceType.FINITE if self.finite_source else SourceType.POINT
            ),
            parallax_branch=ParallaxBranch.U0_PLUS,
            lens_orb_motion=LensOrbMotion.NONE,
        )

    if label == "PL par u0-":
        return ModelKey(
            lens_type=LensType.POINT,
            source_type=(
                SourceType.FINITE if self.finite_source else SourceType.POINT
            ),
            parallax_branch=ParallaxBranch.U0_MINUS,
            lens_orb_motion=LensOrbMotion.NONE,
        )

    # TODO: add mappings for other parallax branches and binary models.
    raise ValueError(f"Unknown model label: {label}")


def model_key_to_label(key: ModelKey) -> str:
    if key.lens_type == LensType.POINT and key.parallax_branch == ParallaxBranch.NONE:
        if key.source_type == SourceType.POINT:
            return "static PSPL"
        elif key.source_type == SourceType.FINITE:
            return "static FSPL"

    if key.lens_type == LensType.POINT and key.parallax_branch in (
            ParallaxBranch.U0_PLUS,
            ParallaxBranch.U0_MINUS,
            ParallaxBranch.U0_PP,
            ParallaxBranch.U0_MM,
            ParallaxBranch.U0_PM,
            ParallaxBranch.U0_MP,
    ):
        return f"PL par {key.parallax_branch.value}"

    # TODO: extend for binary models, motion models, etc.
    return f"{key.lens_type.value} / {key.source_type.value} / " \
           f"{key.parallax_branch.value} / {key.lens_orb_motion.value}"


'''
