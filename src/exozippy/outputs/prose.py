"""The prose collector behind ``<prefix>_paper.tex``.

Components (and run.py) declare one crude sentence per modeling decision --
"We imposed the \\citet{Chen:2017} mass--radius relation on planet b." --
and the writer (``outputs/modeling.py``) concatenates them, grouped and
ordered, into a compilable draft modeling section.  No LLM is involved
anywhere: the value of the file is that every citation relevant to how the
fit was actually done lands in one place, with enough connective prose to
seed a paper's modeling section.

Two rules keep the prose honest (see notes/modeling_prose.txt):

- **Declare at the implementation site.**  A sentence is emitted by the
  code path that implements the feature it describes -- the same rule as
  ``Parameter.add_prior_contribution`` ("declare it in the same place you
  call ``pm.Potential``") -- so the prose can never drift from the
  behavior.  There is deliberately no ``describe_model()`` hook that
  re-derives which features were active.

- **The sentence text is the single source of citations.**  Bibtex keys
  are regex-extracted from ``\\cite*{...}`` in the text itself; there is
  no separate ``citations=`` argument to drift from the text.  Every key
  must exist in the shipped universal ``references.bib``
  (``tests/test_prose.py`` cross-references them statically, in the mold
  of ``tests/test_latex_macro_xref.py``).

The collector lives on ``System`` (``system.prose``) and is REGENERATED
into the output file at each checkpoint (after ``build_model()``, again at
wrap-up) -- never appended to on disk.  ``add()`` is therefore idempotent
by ``key``: a second ``build_model()`` on one System (the GUI does this)
replaces rather than accumulates, and a wrap-up call may overwrite a
sentence declared earlier with updated facts.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------
# Section vocabulary
# ----------------------------------------------------------------------
#
# Fixed and owned here: a sentence names one of these sections and the
# writer emits the sections in exactly this order.  An unknown section
# RAISES at add() time -- a silently ignored section name would drop the
# sentence from the document, which is precisely the failure mode this
# feature exists to prevent (a modeling choice the paper never mentions).
#
# The order is the flow of a conventional modeling section: what the data
# are, then the physical model (stellar -> planetary -> orbits ->
# microlensing), then how the noise is treated, then the priors, then how
# the posterior was sampled.  The post-fit sections (results onward) are
# only populated at wrap-up; before sampling they are simply absent from
# the document.  Figures, the parameter table and the bibliography are
# structural parts of the DOCUMENT, not prose sections -- the writer
# places them itself.

SECTION_ORDER = (
    "intro",
    "data",
    "stellar",
    "planetary",
    "orbits",
    "microlensing",
    "noise",
    "priors",
    "sampling",
    "results",
    "convergence",
    "modes",
    "evidence",
)

# Sections populated only after sampling; the writer uses this to split
# the document into "Modeling" (everything before) and "Results".
POST_FIT_SECTIONS = ("results", "convergence", "modes", "evidence")

# \cite, \citet, \citep, \citealt, \citeauthor, starred forms, and the
# optional [pre][post] arguments; group 1 is the comma-separated key list.
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}")


def extract_cite_keys(text):
    """Every bibtex key cited by ``text``, in order of first appearance.

    Handles all natbib forms (``\\citet``, ``\\citep``, ``\\citealt``,
    starred variants, optional arguments) and comma-separated multi-key
    citations.  Duplicates are dropped.
    """
    seen = []
    for match in _CITE_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and key not in seen:
                seen.append(key)
    return seen


def join_names(names, conjunction="and"):
    """Oxford-comma list: 'A', 'A and B', 'A, B, and C'.

    The one aggregation helper the prose layer offers: components fold a
    per-instance fact into one readable sentence ("...for HARPS, TRES,
    and APF") instead of emitting near-identical sentences per instance.
    """
    names = [str(n) for n in names]
    if len(names) == 0:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} {conjunction} {names[1]}"
    return f"{', '.join(names[:-1])}, {conjunction} {names[-1]}"


def plural(n, singular, plural_form=None):
    """'1 dataset' / '3 datasets'; irregular plurals via ``plural_form``."""
    word = singular if n == 1 else (plural_form or singular + "s")
    return f"{n} {word}"


def get_collector(system):
    """``system``'s prose collector, attaching one if absent.

    Component code calls this instead of touching ``system.prose``
    directly.  A real ``System`` builds its collector in ``__init__``, so
    this is a plain read there -- but component-level tests exercise
    ``build_likelihood`` against MINIMAL fake systems (SimpleNamespace and
    friends), and prose declaration must never be the reason such a test
    fails.  Attach-on-first-use keeps every sentence a fake's build makes
    inspectable on the fake itself; an object that refuses the setattr
    gets a throwaway collector (the sentences are discarded, which is
    fine: nothing renders a fake).
    """
    collector = getattr(system, "prose", None)
    if isinstance(collector, ProseCollector):
        return collector
    collector = ProseCollector()
    try:
        system.prose = collector
    except (AttributeError, TypeError):
        pass
    return collector


@dataclass
class ProseSentence:
    """One declared sentence.

    ``text``    LaTeX prose (it may cite: ``\\citet{Chen:2017}``).  It is
                emitted verbatim, so anything that is not meant to be
                LaTeX must be escaped by the caller (``latex_escape``) --
                same contract as ``table_note``.
    ``section`` One of ``SECTION_ORDER``.
    ``key``     Idempotency handle: a later ``add()`` with the same key
                REPLACES this sentence.  Defaults to the text itself, so
                byte-identical duplicates collapse even without a key.
    ``rank``    Sort key within the section (lower = earlier); ties keep
                insertion order.  50 is the default middle ground --
                reserve <10 for topic sentences and >90 for trailing
                caveats.
    """

    text: str
    section: str
    key: str
    rank: float = 50.0
    _order: int = field(default=0, compare=False)

    def cite_keys(self):
        return extract_cite_keys(self.text)


class ProseCollector:
    """Ordered, idempotent bag of :class:`ProseSentence`.

    Instantiated once per ``System`` (``system.prose``); components add
    sentences during stages 1-7, run.py adds the sampling/results
    sentences, and ``outputs/modeling.py`` renders the whole thing at
    each checkpoint.
    """

    def __init__(self):
        self._sentences = {}  # key -> ProseSentence, insertion-ordered
        self._counter = 0
        self._software = []  # names for the \software{...} line

    def add_software(self, name):
        """Declare a package for the document's ``\\software{...}`` line.

        Same declare-at-site rule as ``add()``: the code path that uses
        celerite2 is the one that declares celerite2.  Idempotent; the
        writer adds the always-used core stack itself.
        """
        if name not in self._software:
            self._software.append(name)

    @property
    def software(self):
        return list(self._software)

    def add(self, text, section, key=None, rank=50.0):
        """Declare (or replace) one sentence.

        Idempotent by ``key``: calling twice with the same key keeps one
        copy, with the LAST text/rank (so a wrap-up pass may refresh a
        sentence with updated facts).  A replaced sentence keeps its
        original position in insertion order -- regeneration must not
        shuffle the paragraph.
        """
        if section not in SECTION_ORDER:
            raise ValueError(
                f"Unknown prose section '{section}'. Valid sections, in "
                f"document order: {', '.join(SECTION_ORDER)}. (Raising "
                f"rather than ignoring: a silently dropped sentence is a "
                f"modeling choice the draft never mentions.)"
            )
        if key is None:
            key = text
        existing = self._sentences.get(key)
        order = existing._order if existing is not None else self._counter
        if existing is None:
            self._counter += 1
        self._sentences[key] = ProseSentence(
            text=text, section=section, key=key, rank=rank, _order=order
        )

    def sentences(self, sections=None):
        """All sentences in document order.

        Sorted by (section order, rank, insertion order).  ``sections``
        optionally restricts to an iterable of section names.
        """
        wanted = set(SECTION_ORDER if sections is None else sections)
        chosen = [s for s in self._sentences.values() if s.section in wanted]
        return sorted(
            chosen,
            key=lambda s: (SECTION_ORDER.index(s.section), s.rank, s._order),
        )

    def paragraphs(self, sections=None):
        """``[(section, [text, ...]), ...]`` in document order.

        One paragraph per non-empty section -- the writer's unit of
        layout.
        """
        out = []
        for section in SECTION_ORDER if sections is None else sections:
            texts = [s.text for s in self.sentences(sections=[section])]
            if texts:
                out.append((section, texts))
        return out

    def cite_keys(self):
        """Union of every sentence's cited keys, first-appearance order."""
        seen = []
        for s in self.sentences():
            for key in s.cite_keys():
                if key not in seen:
                    seen.append(key)
        return seen

    def __len__(self):
        return len(self._sentences)

    def __contains__(self, key):
        return key in self._sentences
