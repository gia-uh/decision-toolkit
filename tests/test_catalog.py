"""The curated catalog: the initial 12 plus variants, planned entries and waitlist."""

from __future__ import annotations

from decision_toolkit import catalog

#: The initial selection: 12 models that meet the requirements (open weights,
#: commercial-friendly license, CPU-viable, System One-compatible or with a
#: public adapter).
INITIAL_TWELVE = {
    "laya",
    "julia-1",
    "von",
    "thx-01",
    "gliner2.5-decide",
    "gliner2.5-multi-decide",
    "d3-lite",
    "intern-decision-0.8b",
    "kev-0.8b",
    "sieve-0.8b",
    "strands-decider-2b",
    "opendecider-nano",
}


def test_initial_twelve_are_catalogued():
    assert INITIAL_TWELVE <= set(catalog.CATALOG_BY_ID)


def test_ids_are_unique():
    ids = [model.id for model in catalog.CATALOG]
    assert len(ids) == len(set(ids))


def test_statuses_are_known():
    assert {model.status for model in catalog.CATALOG} <= {"m0", "variant", "planned", "waitlist"}


def test_m0_entries_have_implemented_backends():
    for model in catalog.CATALOG:
        if model.status == "m0":
            assert model.backend in catalog.IMPLEMENTED_BACKENDS


def test_every_entry_has_license_and_repo():
    for model in catalog.CATALOG:
        assert model.license, model.id
        assert "/" in model.repo, model.id
        assert model.params, model.id


def test_waitlist_flags_unverified_licenses():
    for model in catalog.CATALOG:
        if model.status == "waitlist":
            assert model.license in {"unverified", "other (LFM Open License)"} or model.notes, model.id


def test_lookup_helpers():
    assert catalog.get("laya").repo == "convaiinnovations/laya"
    assert catalog.get("nope") is None
    assert catalog.find_by_repo("convaiinnovations/laya").id == "laya"
    assert catalog.find_by_repo("nope/nope") is None


def test_laya_variants_share_the_repo():
    variants = [m for m in catalog.CATALOG if m.repo == "convaiinnovations/laya"]
    assert {m.subfolder for m in variants} == {None, "multilingual", "typed-decisions"}
