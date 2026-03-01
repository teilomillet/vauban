"""Parse the [measure] section of a TOML config."""

from vauban.config._types import TomlDict
from vauban.types import MeasureConfig, SubspaceBankEntry


def _parse_measure(raw: TomlDict) -> MeasureConfig:
    """Parse the optional [measure] section into a MeasureConfig."""
    mode_raw = raw.get("mode", "direction")
    if not isinstance(mode_raw, str):
        msg = f"[measure].mode must be a string, got {type(mode_raw).__name__}"
        raise TypeError(msg)
    if mode_raw not in ("direction", "subspace", "dbdi", "diff"):
        msg = (
            f"[measure].mode must be 'direction', 'subspace', 'dbdi',"
            f" or 'diff', got {mode_raw!r}"
        )
        raise ValueError(msg)

    top_k_raw = raw.get("top_k", 5)
    if not isinstance(top_k_raw, int):
        msg = (
            f"[measure].top_k must be an integer,"
            f" got {type(top_k_raw).__name__}"
        )
        raise TypeError(msg)

    clip_quantile_raw = raw.get("clip_quantile", 0.0)
    if not isinstance(clip_quantile_raw, int | float):
        msg = (
            f"[measure].clip_quantile must be a number,"
            f" got {type(clip_quantile_raw).__name__}"
        )
        raise TypeError(msg)
    clip_quantile = float(clip_quantile_raw)
    if not 0.0 <= clip_quantile < 0.5:
        msg = (
            f"[measure].clip_quantile must be in [0.0, 0.5),"
            f" got {clip_quantile}"
        )
        raise ValueError(msg)

    transfer_models_raw = raw.get("transfer_models", [])
    if not isinstance(transfer_models_raw, list):
        msg = (
            f"[measure].transfer_models must be a list,"
            f" got {type(transfer_models_raw).__name__}"
        )
        raise TypeError(msg)
    transfer_models: list[str] = []
    for i, item in enumerate(transfer_models_raw):
        if not isinstance(item, str):
            msg = (
                f"[measure].transfer_models[{i}] must be a string,"
                f" got {type(item).__name__}"
            )
            raise TypeError(msg)
        transfer_models.append(item)

    measure_only_raw = raw.get("measure_only", False)
    if not isinstance(measure_only_raw, bool):
        msg = (
            f"[measure].measure_only must be a boolean,"
            f" got {type(measure_only_raw).__name__}"
        )
        raise TypeError(msg)

    # -- diff_model (required when mode="diff") --
    diff_model_raw = raw.get("diff_model")
    diff_model: str | None = None
    if diff_model_raw is not None:
        if not isinstance(diff_model_raw, str):
            msg = (
                f"[measure].diff_model must be a string,"
                f" got {type(diff_model_raw).__name__}"
            )
            raise TypeError(msg)
        diff_model = diff_model_raw

    if mode_raw == "diff" and diff_model is None:
        msg = "[measure].diff_model is required when mode = 'diff'"
        raise ValueError(msg)

    # -- bank (list of inline tables for Steer2Adapt subspace bank) --
    bank_raw = raw.get("bank", [])
    if not isinstance(bank_raw, list):
        msg = (
            f"[measure].bank must be a list of tables,"
            f" got {type(bank_raw).__name__}"
        )
        raise TypeError(msg)

    bank: list[SubspaceBankEntry] = []
    for i, entry in enumerate(bank_raw):
        if not isinstance(entry, dict):
            msg = (
                f"[measure].bank[{i}] must be a table,"
                f" got {type(entry).__name__}"
            )
            raise TypeError(msg)
        name = entry.get("name")
        if not isinstance(name, str):
            msg = (
                f"[measure].bank[{i}].name must be a string,"
                f" got {type(name).__name__}"
            )
            raise TypeError(msg)
        harmful_src = entry.get("harmful", "default")
        if not isinstance(harmful_src, str):
            msg = (
                f"[measure].bank[{i}].harmful must be a string,"
                f" got {type(harmful_src).__name__}"
            )
            raise TypeError(msg)
        harmless_src = entry.get("harmless", "default")
        if not isinstance(harmless_src, str):
            msg = (
                f"[measure].bank[{i}].harmless must be a string,"
                f" got {type(harmless_src).__name__}"
            )
            raise TypeError(msg)
        bank.append(SubspaceBankEntry(
            name=name, harmful=harmful_src, harmless=harmless_src,
        ))

    return MeasureConfig(
        mode=mode_raw,
        top_k=int(top_k_raw),
        clip_quantile=clip_quantile,
        transfer_models=transfer_models,
        diff_model=diff_model,
        measure_only=measure_only_raw,
        bank=bank,
    )
