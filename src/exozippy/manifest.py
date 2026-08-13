"""The one interpreter for the manifest vocabulary.

A component declares ``self.manifest`` at stage 2: a dict mapping parameter
name -> entry.  An entry may be

* ``None`` (or any falsy value) -- a free parameter with no options,
* a string -- the key of an ``expressions:`` block in the component's
  defaults.yaml (almost always ``"default"``),
* a dict -- options, optionally including ``"expr_key"``.

The vocabulary has three consumers, at three different lifecycle stages:

* ``graph.determine_pymc_build_order`` (stage 4) -- needs the expression key
  and its dependency list to place the parameter in the build order;
* ``Component.add_parameter`` (stage 5) -- needs the expression key, its
  dependency list, and every remaining option;
* ``System.derived_params`` -- needs only "does this entry name an
  expression at all".

Each of them used to answer those questions itself, and they DISAGREED: a
dict without ``expr_key`` (``{"overrides": ...}``, ``{"lower": ...}``) is a
free parameter to ``add_parameter``, but ``graph.py`` fell back to the
``"default"`` expression for *any* dict.  That was inert only while no
pinned free parameter had an unused ``expressions:`` block in its
defaults.yaml, and became a hard "Dependency Error" the moment one did --
Band's linear-law ``u1``, whose Kipping expression the manifest deliberately
ignores, and planet's ``beam``, whose "off" entry made every orbit-less
config demand an RV semi-amplitude.  Both were fixed in graph.py by hand;
this module exists so the vocabulary cannot re-diverge.

Nothing here imports from the rest of the package, so every consumer can
import it without a cycle.  It contains no component-specific knowledge --
it is a parser for a small data format, nothing more.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ManifestEntry:
    """A manifest entry, split into its expression key and its options.

    ``expr_key`` is the name of the ``expressions:`` block the entry selects,
    or ``None`` for a free parameter.  ``options`` holds everything else the
    entry carried (``shape``, ``names``, ``overrides``, ``deps``,
    ``table_note``, ``force_node``, direct field overrides such as ``lower``
    ...) -- never ``expr_key`` itself.
    """

    expr_key: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def names_expression(self) -> bool:
        """Does this entry name an expression (i.e. is the parameter derived)?

        This is the structural question, answered from the manifest alone:
        it does not check that the named block exists in the component's
        defaults.yaml.  ``System.derived_params`` asks exactly this, because
        it runs without a resolved config in hand.
        """
        return self.expr_key is not None

    @property
    def deps(self) -> Optional[List[str]]:
        """The entry's own dependency override, or ``None`` if it has none."""
        return self.options.get("deps")

    @property
    def overrides(self) -> Optional[Dict[str, Any]]:
        """Component-computed per-element defaults, layered UNDER user params."""
        return self.options.get("overrides")

    @property
    def shape(self) -> Optional[tuple]:
        """A shape override for parameters that are not one-per-element."""
        return self.options.get("shape")

    @property
    def names(self) -> Optional[List[str]]:
        """A per-element name override for resolution and display labels."""
        return self.options.get("names")

    def expression_config(
        self, expressions: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """The ``expressions:`` block this entry selects, or ``None``.

        ``None`` means the parameter is free *as built*: either the entry
        names no expression, or it names one the resolved config does not
        define.  This is the question ``add_parameter`` and ``graph.py`` both
        ask, and they must get the same answer -- an expression graph.py
        wires an edge for but add_parameter never builds is a spurious edge
        (or a cycle); one add_parameter builds but graph.py missed is a node
        built out of order.
        """
        if self.expr_key is None:
            return None
        return (expressions or {}).get(self.expr_key)

    def dep_names(
        self, expr_cfg: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """The dependency names for this entry's expression.

        A ``deps`` list on the manifest entry wins outright (components use
        it for per-instance dependencies -- an orbit's body groups); with no
        such list, the expression block's own ``deps`` apply.
        """
        manifest_deps = self.deps
        if manifest_deps is not None:
            return list(manifest_deps)
        return list((expr_cfg or {}).get("deps", []))


def interpret_manifest_entry(raw: Any) -> ManifestEntry:
    """Parse one raw manifest value into a :class:`ManifestEntry`.

    An empty-string ``expr_key`` is normalized to ``None``: it names no
    block, and treating it as a name is the kind of near-miss the three
    hand-written readers used to disagree about.
    """
    if not raw:
        # None, {}, "", 0, False: a free parameter carrying no options.
        return ManifestEntry()
    if isinstance(raw, str):
        return ManifestEntry(expr_key=raw)
    if isinstance(raw, dict):
        options = dict(raw)  # never hand back a view of the live manifest
        expr_key = options.pop("expr_key", None) or None
        return ManifestEntry(expr_key=expr_key, options=options)
    raise TypeError(
        f"Manifest entries must be None, a string naming an expressions: "
        f"block, or a dict of options; got {type(raw).__name__} ({raw!r})."
    )
