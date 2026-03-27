# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Canonical harm category taxonomy with regex-based classification.

A two-level tree: *domains* group related *categories*.  The taxonomy is a
reference — it is not enforced on data files.  Categories use the exact
``id`` strings found in JSONL ``"category"`` fields; aliases map legacy or
freeform strings to canonical IDs.

Each category carries optional regex *patterns* for automated text
classification via :func:`score_text` and :func:`classify_text`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HarmCategory:
    """One leaf in the taxonomy tree."""

    id: str
    name: str
    description: str
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HarmDomain:
    """Top-level grouping of related harm categories."""

    id: str
    name: str
    categories: tuple[HarmCategory, ...]


@dataclass(frozen=True, slots=True)
class HarmScore:
    """Score for one category against a text sample."""

    category_id: str
    domain_id: str
    score: float  # 0.0-1.0: fraction of patterns matched
    matched_patterns: tuple[str, ...]


# ── Taxonomy ──────────────────────────────────────────────────────────

TAXONOMY: tuple[HarmDomain, ...] = (
    # ── 1. Violence & Weapons ─────────────────────────────────────────
    HarmDomain(
        id="violence",
        name="Violence & Weapons",
        categories=(
            HarmCategory(
                "weapons",
                "Weapons",
                "Design, manufacture, or acquisition of weapons.",
                patterns=(
                    r"\b(bomb|explosive|IED|firearm|gun)\b",
                    r"\bweapon\s+(design|manufactur|build|assembl)",
                    r"\b(detonat|fuse|primer|propellant)\b",
                    r"\b3[dD].?print.{0,20}(gun|receiver|barrel)\b",
                    r"\b(silencer|suppressor|bump.?stock)\b",
                ),
            ),
            HarmCategory(
                "violence",
                "Violence",
                "Instructions or incitement to physical violence.",
                patterns=(
                    r"\b(kill|murder|assault|attack|maim)\b",
                    r"\bhow\s+to\s+(hurt|harm|injure|wound)\b",
                    r"\b(stab|strangle|bludgeon|shoot)\s+(someone|a\s+person)",
                    r"\b(torture|mutilat)\b",
                ),
            ),
            HarmCategory(
                "terrorism",
                "Terrorism",
                "Planning, financing, or executing terrorist acts.",
                patterns=(
                    r"\bterroris[mt]\b",
                    r"\b(jihad|martyr\s+operation|suicide\s+bomb)\b",
                    r"\b(recruit|radicali[sz]).{0,30}(attack|cell|network)\b",
                    r"\b(car\s+bomb|truck\s+attack|mass\s+casualty)\b",
                ),
            ),
            HarmCategory(
                "mass_casualty",
                "Mass Casualty",
                "Plans targeting crowds, infrastructure, or public spaces.",
                patterns=(
                    r"\bmass\s+(shooting|casualt|kill)\b",
                    r"\b(target|attack).{0,20}(crowd|school|hospital|stadium)\b",
                    r"\b(manifesto|body\s+count|kill\s+count)\b",
                ),
            ),
            HarmCategory(
                "assassination",
                "Assassination",
                "Targeted killing of specific individuals.",
                patterns=(
                    r"\bassassinat\b",
                    r"\b(hit\s*man|contract\s+kill)\b",
                    r"\b(eliminat|neutrali[sz]e).{0,15}(target|person|official)\b",
                ),
            ),
        ),
    ),
    # ── 2. Cyber Threats ──────────────────────────────────────────────
    HarmDomain(
        id="cyber",
        name="Cyber Threats",
        categories=(
            HarmCategory(
                "cyber_attack",
                "Cyber Attack",
                "Hacking, exploitation, or unauthorized access.",
                patterns=(
                    r"\b(hack|exploit|breach|intrusion|pwn)\b",
                    r"\b(SQL\s*injection|XSS|RCE|buffer\s+overflow)\b",
                    r"\b(reverse\s+shell|privilege\s+escalat|root\s+access)\b",
                    r"\b(penetrat|unauthorized\s+access)\b",
                ),
            ),
            HarmCategory(
                "malware",
                "Malware",
                "Creation or distribution of malicious software.",
                patterns=(
                    r"\b(malware|virus|trojan|worm|spyware)\b",
                    r"\b(keylogger|rootkit|backdoor|botnet)\b",
                    r"\b(payload|shellcode|dropper|loader)\b",
                    r"\b(ransomware|crypto.?lock|encrypt.{0,10}files)\b",
                ),
            ),
            HarmCategory(
                "ransomware",
                "Ransomware",
                "Encryption-based extortion targeting systems or data.",
                patterns=(
                    r"\bransomware\b",
                    r"\b(encrypt.{0,15}(files|data|drive).{0,15}(ransom|payment|bitcoin))\b",
                    r"\b(decrypt\s+key|ransom\s+note|pay\s+.*bitcoin)\b",
                ),
            ),
            HarmCategory(
                "credential_theft",
                "Credential Theft",
                "Stealing passwords, tokens, or authentication material.",
                patterns=(
                    r"\b(credential|password)\s+(steal|dump|harvest|crack)\b",
                    r"\b(brute\s*force|dictionary\s+attack|rainbow\s+table)\b",
                    r"\b(mimikatz|hashcat|john\s+the\s+ripper)\b",
                    r"\b(session\s+hijack|token\s+theft|cookie\s+steal)\b",
                ),
            ),
            HarmCategory(
                "ddos",
                "DDoS",
                "Distributed denial-of-service attacks.",
                patterns=(
                    r"\b[Dd][Dd][Oo][Ss]\b",
                    r"\b(denial.of.service|flood\s+attack|amplification\s+attack)\b",
                    r"\b(booter|stresser|botnet.{0,15}(attack|flood))\b",
                ),
            ),
        ),
    ),
    # ── 3. Deception & Fraud ──────────────────────────────────────────
    HarmDomain(
        id="deception",
        name="Deception & Fraud",
        categories=(
            HarmCategory(
                "fraud",
                "Fraud",
                "Scams, impersonation, or financial deception.",
                patterns=(
                    r"\b(scam|fraud|con\s+artist|swindl)\b",
                    r"\b(ponzi|pyramid\s+scheme|advance.fee)\b",
                    r"\b(impersonat|fake\s+identity|forge)\b",
                ),
            ),
            HarmCategory(
                "financial_crime",
                "Financial Crime",
                "Money laundering, insider trading, or tax evasion.",
                patterns=(
                    r"\b(money\s+launder|launder\s+money)\b",
                    r"\b(insider\s+trad|tax\s+evas|embezzl)\b",
                    r"\b(shell\s+company|offshore\s+account|smurfing)\b",
                ),
            ),
            HarmCategory(
                "disinformation",
                "Disinformation",
                "Deliberate spread of false information at scale.",
                patterns=(
                    r"\b(disinformation|misinformation|fake\s+news)\b",
                    r"\b(propaganda|astroturf|troll\s+farm)\b",
                    r"\b(bot\s+network|sockpuppet|coordinated\s+inauthentic)\b",
                ),
            ),
            HarmCategory(
                "social_engineering",
                "Social Engineering",
                "Manipulation or phishing to exploit human trust.",
                patterns=(
                    r"\b(phishing|spear.?phish|vishing|smishing)\b",
                    r"\b(social\s+engineer|pretexting|baiting)\b",
                    r"\b(manipulat.{0,15}(click|link|download|open))\b",
                ),
            ),
            HarmCategory(
                "deepfakes",
                "Deepfakes",
                "Synthetic media to deceive or defame.",
                patterns=(
                    r"\b(deepfake|face\s*swap|synthetic\s+media)\b",
                    r"\b(voice\s+clon|audio\s+fake|lip\s*sync\s+fake)\b",
                    r"\b(fake\s+(video|image|photo).{0,20}(person|celebrity|politician))\b",
                ),
            ),
            HarmCategory(
                "identity_theft",
                "Identity Theft",
                "Stealing or fabricating personal identities.",
                patterns=(
                    r"\b(identity\s+theft|steal\s+identity)\b",
                    r"\b(fake\s+(ID|passport|driver.?s?\s+licen[cs]e))\b",
                    r"\b(SSN|social\s+security\s+number).{0,15}(steal|fake|buy)\b",
                ),
            ),
        ),
    ),
    # ── 4. Exploitation & Abuse ───────────────────────────────────────
    HarmDomain(
        id="exploitation",
        name="Exploitation & Abuse",
        categories=(
            HarmCategory(
                "exploitation",
                "Exploitation",
                "Human trafficking, forced labor, or coercion.",
                patterns=(
                    r"\b(human\s+traffick|sex\s+traffick|labor\s+traffick)\b",
                    r"\b(forced\s+labor|debt\s+bondage|indentured)\b",
                    r"\b(coerci|exploit.{0,15}(worker|person|victim))\b",
                ),
            ),
            HarmCategory(
                "sexual_content",
                "Sexual Content",
                "Non-consensual sexual material or minors-adjacent content.",
                patterns=(
                    r"\b(non.?consensual|revenge\s+porn)\b",
                    r"\b(sexual\s+(exploit|abus|assault))\b",
                    r"\b(sextort|intimate\s+image.{0,10}without\s+consent)\b",
                ),
            ),
            HarmCategory(
                "child_safety",
                "Child Safety",
                "Content endangering or exploiting minors.",
                patterns=(
                    r"\b(CSAM|child\s+(porn|exploitat|abus))\b",
                    r"\b(groom|predator).{0,15}(child|minor|underage)\b",
                    r"\b(pedophil|paedophil)\b",
                ),
            ),
            HarmCategory(
                "forced_labor",
                "Forced Labor",
                "Coerced work, modern slavery, or sweatshop operations.",
                patterns=(
                    r"\b(forced\s+labor|modern\s+slavery|slave\s+labor)\b",
                    r"\b(sweatshop|bonded\s+labor|debt\s+slavery)\b",
                ),
            ),
        ),
    ),
    # ── 5. Self-Harm ──────────────────────────────────────────────────
    HarmDomain(
        id="self_harm",
        name="Self-Harm",
        categories=(
            HarmCategory(
                "self_harm",
                "Self-Harm",
                "Suicide methods, self-injury instructions, or glorification.",
                patterns=(
                    r"\b(suicide|self.?harm|self.?injur)\b",
                    r"\b(cut\s+(myself|yourself|wrists))\b",
                    r"\b(overdose|hang\s+(myself|yourself))\b",
                    r"\bhow\s+to\s+(kill|end).{0,10}(myself|yourself|my\s+life)\b",
                ),
            ),
            HarmCategory(
                "eating_disorders",
                "Eating Disorders",
                "Promotion of dangerous eating behaviors or starvation.",
                patterns=(
                    r"\b(pro.?ana|pro.?mia|thinspo|thinspiration)\b",
                    r"\b(starv|purg|binge.{0,5}purg)\b",
                    r"\b(anorexi|bulimi).{0,15}(tip|trick|how\s+to)\b",
                ),
            ),
        ),
    ),
    # ── 6. Hate & Discrimination ──────────────────────────────────────
    HarmDomain(
        id="hate",
        name="Hate & Discrimination",
        categories=(
            HarmCategory(
                "hate_speech",
                "Hate Speech",
                "Slurs, dehumanization, or incitement against protected groups.",
                patterns=(
                    r"\b(hate\s+speech|racial\s+slur)\b",
                    r"\b(dehumani[sz]|subhuman|inferior\s+race)\b",
                    r"\b(ethnic\s+cleansing|genocide)\b",
                ),
            ),
            HarmCategory(
                "discrimination",
                "Discrimination",
                "Systematic exclusion or bias based on protected characteristics.",
                patterns=(
                    r"\b(discriminat|segregat|apartheid)\b",
                    r"\b(exclude|ban).{0,15}(race|gender|religion|ethnic)\b",
                ),
            ),
            HarmCategory(
                "harassment",
                "Harassment",
                "Targeted intimidation, threats, or abuse campaigns.",
                patterns=(
                    r"\b(harass|bully|intimidat|stalk)\b",
                    r"\b(death\s+threat|rape\s+threat|bomb\s+threat)\b",
                    r"\b(brigade|mob|swat).{0,10}(target|victim|person)\b",
                ),
            ),
        ),
    ),
    # ── 7. Privacy & Surveillance ─────────────────────────────────────
    HarmDomain(
        id="privacy",
        name="Privacy & Surveillance",
        categories=(
            HarmCategory(
                "surveillance",
                "Surveillance",
                "Unauthorized monitoring, stalking, or tracking.",
                patterns=(
                    r"\b(surveillanc|spy\s+on|monitor\s+without)\b",
                    r"\b(track\s+(someone|person|spouse|partner))\b",
                    r"\b(hidden\s+camera|bug\s+a\s+phone|wiretap)\b",
                ),
            ),
            HarmCategory(
                "privacy_pii",
                "PII Exposure",
                "Extraction or leakage of personally identifiable information.",
                patterns=(
                    r"\b(PII|personal.{0,5}identif)\b",
                    r"\b(leak|expos|extract).{0,15}(email|phone|address|SSN)\b",
                    r"\b(scrape|harvest).{0,10}(personal|user)\s+(data|info)\b",
                ),
            ),
            HarmCategory(
                "doxxing",
                "Doxxing",
                "Publishing private information to harass or endanger.",
                patterns=(
                    r"\b(dox|doxx)\b",
                    r"\b(publish|post|reveal).{0,15}(address|phone|identity)\b",
                    r"\b(real\s+name|home\s+address).{0,10}(find|reveal|publish)\b",
                ),
            ),
            HarmCategory(
                "data_breach",
                "Data Breach",
                "Unauthorized access to or exfiltration of stored data.",
                patterns=(
                    r"\b(data\s+breach|data\s+leak|data\s+dump)\b",
                    r"\b(exfiltrat|steal\s+data|dump\s+database)\b",
                    r"\b(SQL\s+dump|database\s+leak)\b",
                ),
            ),
        ),
    ),
    # ── 8. CBRN ───────────────────────────────────────────────────────
    HarmDomain(
        id="cbrn",
        name="CBRN",
        categories=(
            HarmCategory(
                "bioweapons",
                "Bioweapons",
                "Biological agents designed to harm humans, animals, or crops.",
                patterns=(
                    r"\b(bioweapon|biological\s+weapon|bio.?terror)\b",
                    r"\b(anthrax|ricin|botulinum|smallpox).{0,15}(weaponi[sz]|release|dispers)\b",
                    r"\b(pathogen|toxin).{0,15}(engineer|modif|enhance)\b",
                ),
            ),
            HarmCategory(
                "chemical_weapons",
                "Chemical Weapons",
                "Synthesis or deployment of toxic chemical agents.",
                patterns=(
                    r"\b(chemical\s+weapon|nerve\s+agent|sarin|VX|mustard\s+gas)\b",
                    r"\b(synthe[st]i[sz]).{0,15}(poison|toxic|lethal)\b",
                    r"\b(chlorine\s+gas|phosgene).{0,10}(attack|release)\b",
                ),
            ),
            HarmCategory(
                "radiological_nuclear",
                "Radiological / Nuclear",
                "Dirty bombs, nuclear material acquisition, or sabotage.",
                patterns=(
                    r"\b(dirty\s+bomb|nuclear\s+weapon|enriched\s+uranium)\b",
                    r"\b(radioactive|nuclear).{0,15}(material|device|bomb)\b",
                    r"\b(centrifuge|plutonium|fissile)\b",
                ),
            ),
        ),
    ),
    # ── 9. Regulated & Professional ───────────────────────────────────
    HarmDomain(
        id="regulated",
        name="Regulated & Professional",
        categories=(
            HarmCategory(
                "drugs",
                "Drugs",
                "Synthesis, trafficking, or abuse of controlled substances.",
                patterns=(
                    r"\b(synthe[st]i[sz]).{0,15}(meth|fentanyl|LSD|MDMA|cocaine)",
                    r"\b(drug\s+(traffick|smuggl|deal))\b",
                    r"\b(cook|manufacture).{0,10}(meth|drugs)\b",
                ),
            ),
            HarmCategory(
                "professional_malpractice",
                "Professional Malpractice",
                "Dangerous medical, legal, or financial advice.",
                patterns=(
                    r"\b(prescri|dosage|medicat).{0,15}(without|illegal|fake)\b",
                    r"\b(practice\s+(law|medicine)\s+without)\b",
                    r"\b(fake\s+(doctor|lawyer|diploma))\b",
                ),
            ),
            HarmCategory(
                "radicalization",
                "Radicalization",
                "Recruitment into extremist ideologies.",
                patterns=(
                    r"\b(radicali[sz]|recruit).{0,15}(extremis|violent|jihad)\b",
                    r"\b(propaganda|indoctrinat).{0,15}(youth|online|forum)\b",
                ),
            ),
            HarmCategory(
                "counterfeit_goods",
                "Counterfeit Goods",
                "Production or sale of counterfeit products or currency.",
                patterns=(
                    r"\b(counterfeit|forg).{0,10}(money|currency|bill|note)\b",
                    r"\b(fake|counterfeit).{0,10}(product|brand|designer)\b",
                    r"\b(print\s+fake|replica\s+currency)\b",
                ),
            ),
        ),
    ),
    # ── 10. IP & Copyright ────────────────────────────────────────────
    HarmDomain(
        id="ip_copyright",
        name="IP & Copyright",
        categories=(
            HarmCategory(
                "plagiarism",
                "Plagiarism",
                "Passing off others' work as original.",
                patterns=(
                    r"\b(plagiari[sz]|copy\s+without\s+attribut)\b",
                    r"\b(submit\s+as\s+(my|your)\s+own)\b",
                ),
            ),
            HarmCategory(
                "copyright_infringement",
                "Copyright Infringement",
                "Unauthorized reproduction or distribution of copyrighted material.",
                patterns=(
                    r"\b(pirat|torrent|crack|keygen)\b",
                    r"\b(copyright\s+infring|DMCA\s+bypass)\b",
                    r"\b(download\s+(free|illegal).{0,10}(movie|music|software|book))\b",
                ),
            ),
            HarmCategory(
                "trade_secret_theft",
                "Trade Secret Theft",
                "Stealing proprietary business information.",
                patterns=(
                    r"\b(trade\s+secret|proprietary).{0,15}(steal|leak|exfiltrat)\b",
                    r"\b(corporate\s+espionage|industrial\s+spy)\b",
                ),
            ),
        ),
    ),
    # ── 11. Environmental ─────────────────────────────────────────────
    HarmDomain(
        id="environmental",
        name="Environmental",
        categories=(
            HarmCategory(
                "environmental_sabotage",
                "Environmental Sabotage",
                "Deliberate destruction of ecosystems or infrastructure.",
                patterns=(
                    r"\b(sabotag|destroy).{0,15}(pipeline|dam|reservoir|grid)\b",
                    r"\b(ecoterror|environmental\s+sabotag)\b",
                    r"\b(arson).{0,10}(forest|wildfire)\b",
                ),
            ),
            HarmCategory(
                "poaching",
                "Poaching",
                "Illegal hunting or trafficking of protected wildlife.",
                patterns=(
                    r"\b(poach|illegal\s+hunt|traffick\s+wildlife)\b",
                    r"\b(ivory|rhino\s+horn|bushmeat).{0,10}(sell|trade|smuggl)\b",
                ),
            ),
            HarmCategory(
                "illegal_dumping",
                "Illegal Dumping",
                "Unauthorized disposal of hazardous or toxic waste.",
                patterns=(
                    r"\b(illegal\s+dump|dump\s+waste|toxic\s+dump)\b",
                    r"\b(dispos|dump).{0,15}(hazardous|chemical|toxic|nuclear)\b",
                ),
            ),
        ),
    ),
    # ── 12. Election Integrity ────────────────────────────────────────
    HarmDomain(
        id="election",
        name="Election Integrity",
        categories=(
            HarmCategory(
                "voter_manipulation",
                "Voter Manipulation",
                "Suppression, intimidation, or deception targeting voters.",
                patterns=(
                    r"\b(voter\s+(suppress|intimidat|deceiv))\b",
                    r"\b(fake\s+(poll|ballot)|ballot\s+stuff)\b",
                    r"\b(mislead\s+voter|wrong\s+voting\s+date)\b",
                ),
            ),
            HarmCategory(
                "election_interference",
                "Election Interference",
                "Foreign or domestic interference in electoral processes.",
                patterns=(
                    r"\b(election\s+(interfer|hack|manipulat|rig))\b",
                    r"\b(voting\s+machine\s+(hack|tamper))\b",
                    r"\b(foreign\s+(interfer|influenc).{0,15}election)\b",
                ),
            ),
        ),
    ),
    # ── 13. AI Safety ─────────────────────────────────────────────────
    HarmDomain(
        id="ai_safety",
        name="AI Safety",
        categories=(
            HarmCategory(
                "model_jailbreaking",
                "Model Jailbreaking",
                "Bypassing model safety alignment via prompts or modifications.",
                patterns=(
                    r"\b(jailbreak|abliterat|uncensor)\b",
                    r"\b(bypass|remove|disable).{0,15}(safety|alignment|guard)\b",
                    r"\b(DAN|do\s+anything\s+now)\b",
                ),
            ),
            HarmCategory(
                "prompt_injection",
                "Prompt Injection",
                "Injecting instructions to override model behavior.",
                patterns=(
                    r"\b(prompt\s+inject|indirect\s+inject)\b",
                    r"\b(ignore\s+(previous|all)\s+instructions)\b",
                    r"\b(system\s+prompt\s+(leak|extract|override))\b",
                ),
            ),
            HarmCategory(
                "alignment_subversion",
                "Alignment Subversion",
                "Undermining safety training via fine-tuning or weight modification.",
                patterns=(
                    r"\b(alignment\s+subver|safety\s+remov)\b",
                    r"\b(fine.?tun|LoRA).{0,15}(remov|bypass|undo).{0,10}safety\b",
                    r"\b(refusal\s+direction|steering\s+vector).{0,10}(remov|cut|ablat)\b",
                ),
            ),
        ),
    ),
)

# ── Aliases ───────────────────────────────────────────────────────────

CATEGORY_ALIASES: dict[str, str] = {
    # Legacy / freeform → canonical
    "hacking": "cyber_attack",
    "hack": "cyber_attack",
    "crime": "violence",
    "phishing": "social_engineering",
    "scam": "fraud",
    "bio": "bioweapons",
    "chemical": "chemical_weapons",
    "nuclear": "radiological_nuclear",
    "stalking": "surveillance",
    "pii": "privacy_pii",
    "dox": "doxxing",
    "suicide": "self_harm",
    "trafficking": "exploitation",
    "csam": "child_safety",
    "extremism": "radicalization",
    "propaganda": "disinformation",
    "misinfo": "disinformation",
    # New aliases
    "ddos_attack": "ddos",
    "dos_attack": "ddos",
    "ransomware_attack": "ransomware",
    "password_theft": "credential_theft",
    "face_swap": "deepfakes",
    "synthetic_media": "deepfakes",
    "id_theft": "identity_theft",
    "slavery": "forced_labor",
    "anorexia": "eating_disorders",
    "bulimia": "eating_disorders",
    "cyberbullying": "harassment",
    "trolling": "harassment",
    "leak": "data_breach",
    "piracy": "copyright_infringement",
    "pirating": "copyright_infringement",
    "counterfeiting": "counterfeit_goods",
    "eco_terrorism": "environmental_sabotage",
    "wildlife_trafficking": "poaching",
    "vote_suppression": "voter_manipulation",
    "election_hacking": "election_interference",
    "jailbreak": "model_jailbreaking",
    "abliteration": "model_jailbreaking",
    "injection": "prompt_injection",
}


# ── Helpers ───────────────────────────────────────────────────────────


_ALL_CATEGORIES: frozenset[str] = frozenset(
    cat.id for domain in TAXONOMY for cat in domain.categories
)

_DOMAIN_BY_CATEGORY: dict[str, str] = {
    cat.id: domain.id
    for domain in TAXONOMY
    for cat in domain.categories
}

_CATEGORY_BY_ID: dict[str, HarmCategory] = {
    cat.id: cat for domain in TAXONOMY for cat in domain.categories
}


def all_categories() -> frozenset[str]:
    """Return the set of all canonical category IDs."""
    return _ALL_CATEGORIES


def resolve_category(raw: str) -> str:
    """Normalize a category string via aliases; return canonical ID or raw."""
    lowered = raw.strip().lower()
    canonical = CATEGORY_ALIASES.get(lowered, lowered)
    return canonical


def domain_for_category(category_id: str) -> str | None:
    """Return the domain ID for a given category, or None if unknown."""
    return _DOMAIN_BY_CATEGORY.get(category_id)


# ── Coverage ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TaxonomyCoverage:
    """Result of comparing observed categories against the taxonomy."""

    present: frozenset[str]
    missing: frozenset[str]
    aliased: dict[str, str]
    coverage_ratio: float


def coverage_report(observed: set[str]) -> TaxonomyCoverage:
    """Compute taxonomy coverage for a set of observed category strings."""
    canonical = all_categories()
    aliased: dict[str, str] = {}
    resolved: set[str] = set()

    for raw in observed:
        resolved_id = resolve_category(raw)
        resolved.add(resolved_id)
        if resolved_id != raw.strip().lower():
            aliased[raw] = resolved_id

    present = frozenset(resolved & canonical)
    missing = frozenset(canonical - resolved)
    ratio = len(present) / len(canonical) if canonical else 0.0

    return TaxonomyCoverage(
        present=present,
        missing=missing,
        aliased=aliased,
        coverage_ratio=ratio,
    )


# ── Regex-based scoring ──────────────────────────────────────────────

# Compiled pattern cache: category_id → list of compiled regexes
_COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {}


def _get_compiled_patterns(category_id: str) -> list[re.Pattern[str]]:
    """Return compiled regexes for a category (cached)."""
    if category_id not in _COMPILED_PATTERNS:
        cat = _CATEGORY_BY_ID.get(category_id)
        if cat is None or not cat.patterns:
            _COMPILED_PATTERNS[category_id] = []
        else:
            _COMPILED_PATTERNS[category_id] = [
                re.compile(p, re.IGNORECASE) for p in cat.patterns
            ]
    return _COMPILED_PATTERNS[category_id]


def score_text(text: str) -> list[HarmScore]:
    """Score a text against all taxonomy categories.

    Returns a :class:`HarmScore` for every category that has at least one
    pattern match.  The ``score`` field is the fraction of the category's
    patterns that matched (0.0-1.0).
    """
    results: list[HarmScore] = []
    for domain in TAXONOMY:
        for cat in domain.categories:
            patterns = _get_compiled_patterns(cat.id)
            if not patterns:
                continue
            matched: list[str] = []
            for pat in patterns:
                if pat.search(text):
                    matched.append(pat.pattern)
            if matched:
                results.append(HarmScore(
                    category_id=cat.id,
                    domain_id=domain.id,
                    score=len(matched) / len(patterns),
                    matched_patterns=tuple(matched),
                ))
    return results


def classify_text(text: str) -> str | None:
    """Return the highest-scoring category ID, or None if no patterns match."""
    scores = score_text(text)
    if not scores:
        return None
    best = max(scores, key=lambda s: (s.score, len(s.matched_patterns)))
    return best.category_id


def score_batch(texts: list[str]) -> list[list[HarmScore]]:
    """Score multiple texts against the taxonomy."""
    return [score_text(t) for t in texts]


__all__ = [
    "CATEGORY_ALIASES",
    "TAXONOMY",
    "HarmCategory",
    "HarmDomain",
    "HarmScore",
    "TaxonomyCoverage",
    "all_categories",
    "classify_text",
    "coverage_report",
    "domain_for_category",
    "resolve_category",
    "score_batch",
    "score_text",
]
