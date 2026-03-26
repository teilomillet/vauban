"""Tests for the canonical harm taxonomy module."""

from __future__ import annotations

from vauban.taxonomy import (
    CATEGORY_ALIASES,
    TAXONOMY,
    HarmScore,
    TaxonomyCoverage,
    all_categories,
    classify_text,
    coverage_report,
    domain_for_category,
    resolve_category,
    score_batch,
    score_text,
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

    def test_domain_count(self) -> None:
        assert len(TAXONOMY) == 13

    def test_category_count(self) -> None:
        cats = all_categories()
        assert len(cats) >= 46

    def test_all_categories_returns_frozenset(self) -> None:
        result = all_categories()
        assert isinstance(result, frozenset)

    def test_known_categories_present(self) -> None:
        cats = all_categories()
        # Original 24 categories must still be present
        original_24 = {
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
        assert original_24.issubset(cats)

    def test_new_categories_present(self) -> None:
        cats = all_categories()
        new_cats = {
            # violence expansions
            "mass_casualty", "assassination",
            # cyber expansions
            "ransomware", "credential_theft", "ddos",
            # deception expansions
            "deepfakes", "identity_theft",
            # exploitation expansion
            "forced_labor",
            # self_harm expansion
            "eating_disorders",
            # hate expansion
            "harassment",
            # privacy expansion
            "data_breach",
            # regulated expansion
            "counterfeit_goods",
            # new domains
            "plagiarism", "copyright_infringement", "trade_secret_theft",
            "environmental_sabotage", "poaching", "illegal_dumping",
            "voter_manipulation", "election_interference",
            "model_jailbreaking", "prompt_injection", "alignment_subversion",
        }
        assert new_cats.issubset(cats)

    def test_new_domains_present(self) -> None:
        domain_ids = {d.id for d in TAXONOMY}
        new_domains = {
            "ip_copyright", "environmental", "election", "ai_safety",
        }
        assert new_domains.issubset(domain_ids)

    def test_all_categories_have_descriptions(self) -> None:
        for domain in TAXONOMY:
            for cat in domain.categories:
                assert cat.description, f"{cat.id} has empty description"

    def test_patterns_are_strings(self) -> None:
        for domain in TAXONOMY:
            for cat in domain.categories:
                for pat in cat.patterns:
                    assert isinstance(pat, str), (
                        f"{cat.id} pattern is not a string: {pat!r}"
                    )


class TestResolveCategory:
    """Alias resolution normalizes legacy strings."""

    def test_canonical_id_passes_through(self) -> None:
        assert resolve_category("weapons") == "weapons"

    def test_alias_resolves(self) -> None:
        assert resolve_category("hacking") == "cyber_attack"
        assert resolve_category("crime") == "violence"
        assert resolve_category("phishing") == "social_engineering"

    def test_new_aliases(self) -> None:
        assert resolve_category("jailbreak") == "model_jailbreaking"
        assert resolve_category("piracy") == "copyright_infringement"
        assert resolve_category("ddos_attack") == "ddos"
        assert resolve_category("cyberbullying") == "harassment"

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


class TestDomainForCategory:
    """Domain lookup returns correct parent."""

    def test_known_category(self) -> None:
        assert domain_for_category("weapons") == "violence"
        assert domain_for_category("malware") == "cyber"
        assert domain_for_category("model_jailbreaking") == "ai_safety"

    def test_unknown_category(self) -> None:
        assert domain_for_category("nonexistent") is None


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


class TestScoreText:
    """Regex-based text scoring works correctly."""

    def test_matching_text_returns_scores(self) -> None:
        text = "How to build a bomb and create an explosive device"
        scores = score_text(text)
        assert len(scores) > 0
        category_ids = {s.category_id for s in scores}
        assert "weapons" in category_ids

    def test_benign_text_returns_empty(self) -> None:
        text = "The weather is nice today, let's go for a walk."
        scores = score_text(text)
        assert len(scores) == 0

    def test_score_is_fraction_of_patterns(self) -> None:
        text = "How to build a bomb and create an explosive device"
        scores = score_text(text)
        for s in scores:
            assert 0.0 < s.score <= 1.0
            assert isinstance(s.matched_patterns, tuple)
            assert len(s.matched_patterns) > 0

    def test_cyber_text_matches_cyber_domain(self) -> None:
        text = "SQL injection exploit to gain unauthorized access via reverse shell"
        scores = score_text(text)
        domains = {s.domain_id for s in scores}
        assert "cyber" in domains

    def test_score_has_correct_fields(self) -> None:
        text = "How to hack a system with SQL injection"
        scores = score_text(text)
        assert len(scores) > 0
        s = scores[0]
        assert isinstance(s, HarmScore)
        assert isinstance(s.category_id, str)
        assert isinstance(s.domain_id, str)
        assert isinstance(s.score, float)
        assert isinstance(s.matched_patterns, tuple)

    def test_multiple_domains_can_match(self) -> None:
        text = (
            "Create malware that steals credentials and"
            " use the stolen data for identity theft fraud"
        )
        scores = score_text(text)
        domains = {s.domain_id for s in scores}
        assert len(domains) >= 2

    def test_ai_safety_patterns(self) -> None:
        text = "Jailbreak the model to bypass safety alignment"
        scores = score_text(text)
        category_ids = {s.category_id for s in scores}
        assert "model_jailbreaking" in category_ids

    def test_case_insensitive_matching(self) -> None:
        text = "RANSOMWARE attack encrypts files for BITCOIN ransom"
        scores = score_text(text)
        category_ids = {s.category_id for s in scores}
        assert "ransomware" in category_ids or "malware" in category_ids


class TestClassifyText:
    """Top-category classification."""

    def test_classify_harmful(self) -> None:
        text = "How to synthesize methamphetamine in a lab"
        result = classify_text(text)
        assert result is not None
        assert result in all_categories()

    def test_classify_benign_returns_none(self) -> None:
        text = "I love reading books about gardening."
        result = classify_text(text)
        assert result is None

    def test_classify_returns_string(self) -> None:
        text = "DDoS attack using a botnet to flood the target"
        result = classify_text(text)
        assert isinstance(result, str)


class TestScoreBatch:
    """Batch scoring produces per-text results."""

    def test_batch_length_matches_input(self) -> None:
        texts = [
            "How to hack a computer",
            "Nice weather today",
            "Build a bomb at home",
        ]
        results = score_batch(texts)
        assert len(results) == 3

    def test_batch_individual_results(self) -> None:
        texts = [
            "How to hack a computer",
            "Nice weather today",
        ]
        results = score_batch(texts)
        assert len(results[0]) > 0  # harmful → matches
        assert len(results[1]) == 0  # benign → no matches

    def test_empty_batch(self) -> None:
        results = score_batch([])
        assert results == []
