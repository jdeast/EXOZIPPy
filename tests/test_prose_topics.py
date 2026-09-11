"""The modeling-draft prose topic band is extensible and graph-ordered.

A prose "topic" groups sentences by subject ACROSS components -- rvinstrument
writes into `orbits`, transit into `planetary`, and planet into both -- so a
topic is not a synonym for a component and the band cannot be derived from the
component list. What CAN be derived is a topic's position: the component that
declared it sits somewhere in the build graph, and dependency order is the
editorial order.

The vocabulary used to be closed, and closed to astronomy specifically, so a
component from another field had to file its "what we fitted" sentence under
`data`, where it was ordered among data-inventory sentences rather than after
them.
"""

import pytest

from exozippy.components.component import Component
from exozippy.outputs.modeling import _DOC_SECTIONS, _doc_sections
from exozippy.outputs.prose import (
    SECTION_ORDER,
    TOPIC_SECTIONS,
    ProseCollector,
)


def test_a_collector_with_no_declared_topic_is_exactly_the_historical_order():
    """Given no declared topic, Then section_order IS SECTION_ORDER.

    The regression pin for every astronomy fit: making the band extensible
    must not move a single existing paragraph.
    """
    assert ProseCollector().section_order == SECTION_ORDER


def test_a_declared_topic_lands_in_the_band_not_at_the_end():
    """Given a declared topic, Then it sits after the shipped topics.

    Position is the whole point -- appending it after `evidence` would put the
    model description after the results, which is why the band has an end
    index rather than the topics being appended to the tuple.
    """
    collector = ProseCollector()
    collector.register_topic("pharmacokinetics")
    order = collector.section_order

    index = order.index("pharmacokinetics")
    assert order[index - 1] == TOPIC_SECTIONS[-1]
    assert order[index + 1] == "noise"


def test_registering_is_idempotent_and_a_shipped_name_is_a_no_op():
    """Given repeated or shipped names, Then the order does not grow.

    A component declaring "stellar" is claiming the shipped topic, not
    creating a second section that happens to share its name.
    """
    collector = ProseCollector()
    collector.register_topic("pharmacokinetics")
    once = collector.section_order

    collector.register_topic("pharmacokinetics")
    collector.register_topic("stellar")

    assert collector.section_order == once
    assert once.count("stellar") == 1


def test_topics_keep_their_registration_order():
    """Given two topics, Then they render in the order they were registered.

    System registers in build-graph order, so this is what makes the graph the
    editorial order.
    """
    collector = ProseCollector()
    collector.register_topic("first")
    collector.register_topic("second")

    order = collector.section_order
    assert order.index("first") < order.index("second")


def test_an_undeclared_section_still_raises_and_names_the_escape_hatch():
    """Given a typo, Then it raises -- and says how to add a real topic.

    Accepting any string would make the band extensible at the cost of the
    typo protection that matters more: a misspelled section would silently
    become its own one-sentence paragraph.
    """
    collector = ProseCollector()
    with pytest.raises(ValueError) as excinfo:
        collector.add("Text.", section="stellr", key="k")
    assert "prose_topic" in str(excinfo.value)


def test_a_declared_topic_accepts_sentences():
    """Given a registered topic, Then add() takes it and paragraphs() emits it."""
    collector = ProseCollector()
    collector.register_topic("pharmacokinetics")
    collector.add("A sentence.", section="pharmacokinetics", key="k")

    assert [sec for sec, _ in collector.paragraphs()] == ["pharmacokinetics"]


def test_component_declares_no_topic_by_default():
    """Given the base class, Then prose_topic is None.

    Declared on Component rather than left to getattr for the same reason
    `label` is: a generic consumer must be able to read it without a guard.
    """
    assert Component.prose_topic is None


def test_the_pharmacokinetics_components_declare_their_own_topic():
    """Given Subject, Then it owns a topic and its model sentence uses it."""
    from exozippy.components.pharmacokinetics.subject import Subject

    assert Subject.prose_topic == "pharmacokinetics"
    assert Subject.prose_topic not in SECTION_ORDER


def test_a_declared_topic_is_routed_into_the_modeling_section():
    """Given a declared topic, Then the writer renders it under Modeling.

    Without routing, a topic would be collected and then rendered into no
    document section at all -- the silent drop the import-time assert in
    modeling.py prevents for the shipped vocabulary. An unrouted topic is the
    same bug with a new name.
    """
    collector = ProseCollector()
    collector.register_topic("pharmacokinetics")

    sections = _doc_sections(collector)
    modeling = next(secs for head, secs in sections if head == "Modeling")

    assert "pharmacokinetics" in modeling
    # ...and every shipped section keeps its existing home.
    for heading, original in _DOC_SECTIONS:
        current = next(s for h, s in sections if h == heading)
        assert set(original) <= set(current)


def test_doc_sections_is_untouched_without_a_declared_topic():
    """Given no topics, Then the routing table is the static one, identically."""
    assert _doc_sections(ProseCollector()) == _DOC_SECTIONS
