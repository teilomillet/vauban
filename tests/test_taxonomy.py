"""Tests for the canonical harm taxonomy module."""

from __future__ import annotations

from vauban.taxonomy import (
    CATEGORY_ALIASES,
    TAXONOMY,
    TaxonomyCoverage,
    all_categories,
    coverage_report,
    resolve_category,
)


class TestTaxonomyStructure:
    """Taxonomy is well-formed and internally consistent."""

    def test_taxonomy_is_nonempty(self) -> None:
        assert len(TAXONOMY) > 0

    def test_all_domains_have_categories(self) -> None:
        for domain in TAXONOMY:
            assert len(domain.categories) > 0, f"domain {domain.id} is empty"

    def test_category_ids_are_unique(self) -> None:
        ids: list[str] = [
            cat.id for domain in TAXONOMY for cat in domain.categories
        ]
        assert len(ids) == len(set(ids)), f"duplicate category ids: {ids}"

    def test_domain_ids_are_unique(self) -> None:
        ids = [domain.id for domain in TAXONOMY]
        assert len(ids) == len(set(ids))

    def test_category_count(self) -> None:
        cats = all_categories()
        # Plan specifies ~22 categories
        assert len(cats) >= 20

    def test_all_categories_returns_frozenset(self) -> None:
        result = all_categories()
        assert isinstance(result, frozenset)

    def test_known_categories_present(self) -> None:
        cats = all_categories()
        expected = {
            "weapons", "violence", "terrorism",
            "cyber_attack", "malware",
            "fraud", "financial_crime", "disinformation", "social_engineering",
            "exploitation", "sexual_content", "child_safety",
            "self_harm",
            "hate_speech", "discrimination",
            "surveillance", "privacy_pii", "doxxing",
            "bioweapons", "chemical_weapons", "radiological_nuclear",
            "drugs", "professional_malpractice", "radicalization",
        }
        assert expected == cats


class TestResolveCategory:
    """Alias resolution normalizes legacy strings."""

    def test_canonical_id_passes_through(self) -> None:
        assert resolve_category("weapons") == "weapons"

    def test_alias_resolves(self) -> None:
        assert resolve_category("hacking") == "cyber_attack"
        assert resolve_category("crime") == "violence"
        assert resolve_category("phishing") == "social_engineering"

    def test_case_insensitive(self) -> None:
        assert resolve_category("Hacking") == "cyber_attack"
        assert resolve_category("FRAUD") == "fraud"

    def test_unknown_passes_through(self) -> None:
        assert resolve_category("unknown_thing") == "unknown_thing"

    def test_whitespace_stripped(self) -> None:
        assert resolve_category("  hacking  ") == "cyber_attack"

    def test_all_aliases_resolve_to_canonical(self) -> None:
        canonical = all_categories()
        for alias, target in CATEGORY_ALIASES.items():
            assert target in canonical, (
                f"alias {alias!r} -> {target!r} is not a canonical category"
            )


class TestCoverageReport:
    """Coverage computation is correct."""

    def test_full_coverage(self) -> None:
        observed = set(all_categories())
        result = coverage_report(observed)
        assert result.coverage_ratio == 1.0
        assert len(result.missing) == 0
        assert result.present == all_categories()

    def test_empty_coverage(self) -> None:
        result = coverage_report(set())
        assert result.coverage_ratio == 0.0
        assert result.missing == all_categories()
        assert len(result.present) == 0

    def test_partial_coverage(self) -> None:
        observed = {"weapons", "drugs", "fraud"}
        result = coverage_report(observed)
        assert result.present == frozenset({"weapons", "drugs", "fraud"})
        assert "weapons" not in result.missing
        assert "malware" in result.missing
        assert 0.0 < result.coverage_ratio < 1.0

    def test_aliases_counted(self) -> None:
        observed = {"hacking", "weapons"}  # hacking -> cyber_attack
        result = coverage_report(observed)
        assert "cyber_attack" in result.present
        assert "hacking" in result.aliased
        assert result.aliased["hacking"] == "cyber_attack"

    def test_coverage_ratio_math(self) -> None:
        canonical = all_categories()
        observed = {"weapons", "drugs"}
        result = coverage_report(observed)
        assert result.coverage_ratio == len(result.present) / len(canonical)

    def test_returns_taxonomy_coverage(self) -> None:
        result = coverage_report({"weapons"})
        assert isinstance(result, TaxonomyCoverage)
