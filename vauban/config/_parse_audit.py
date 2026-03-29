# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [audit] section of a TOML config."""

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.config._types import TomlDict
from vauban.types import AuditConfig


def _parse_audit(raw: TomlDict) -> AuditConfig | None:
    """Parse the optional [audit] section into an AuditConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("audit")
    if sec is None:
        return None
    reader = SectionReader("[audit]", require_toml_table("[audit]", sec))

    company_name = reader.string("company_name")
    system_name = reader.string("system_name")

    thoroughness = reader.literal(
        "thoroughness", ("quick", "standard", "deep"),
        default="standard",
    )

    pdf_report = reader.boolean("pdf_report", default=True)
    pdf_report_filename = reader.string(
        "pdf_report_filename", default="audit_report.pdf",
    )
    if not pdf_report_filename.endswith(".pdf"):
        msg = "[audit].pdf_report_filename must end with .pdf"
        raise ValueError(msg)

    attacks = reader.data.get("attacks")
    attacks_list: list[str] | None = None
    if attacks is not None:
        if not isinstance(attacks, list):
            msg = f"[audit].attacks must be a list, got {type(attacks).__name__}"
            raise TypeError(msg)
        attacks_list = [str(a) for a in attacks]

    softprompt_steps = reader.optional_integer("softprompt_steps")
    if softprompt_steps is not None and softprompt_steps < 1:
        msg = f"[audit].softprompt_steps must be >= 1, got {softprompt_steps}"
        raise ValueError(msg)

    strategies = reader.data.get("jailbreak_strategies")
    strategies_list: list[str] | None = None
    if strategies is not None:
        if not isinstance(strategies, list):
            msg = (
                "[audit].jailbreak_strategies must be a list,"
                f" got {type(strategies).__name__}"
            )
            raise TypeError(msg)
        strategies_list = [str(s) for s in strategies]

    return AuditConfig(
        company_name=company_name,
        system_name=system_name,
        thoroughness=thoroughness,
        pdf_report=pdf_report,
        pdf_report_filename=pdf_report_filename,
        attacks=attacks_list,
        softprompt_steps=softprompt_steps,
        jailbreak_strategies=strategies_list,
    )
