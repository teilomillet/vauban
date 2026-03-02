"""JSON Schema generation for Vauban TOML configs."""

from __future__ import annotations

import json
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from vauban._backend import SUPPORTED_BACKENDS
from vauban.types import (
    ApiEvalConfig,
    CastConfig,
    CircuitConfig,
    ComposeOptimizeConfig,
    CutConfig,
    DefenseStackConfig,
    DepthConfig,
    DetectConfig,
    EnvironmentConfig,
    EvalConfig,
    FeaturesConfig,
    FusionConfig,
    IntentConfig,
    LinearProbeConfig,
    MeasureConfig,
    MetaConfig,
    OptimizeConfig,
    PolicyConfig,
    ProbeConfig,
    RepBendConfig,
    ScanConfig,
    SICConfig,
    SoftPromptConfig,
    SteerConfig,
    SurfaceConfig,
    SVFConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

type JsonSchema = dict[str, object]


@dataclass(frozen=True, slots=True)
class _DataclassSectionSpec:
    """Mapping from a TOML section to a config dataclass."""

    name: str
    config_type: type[object]
    field_aliases: dict[str, str]


_MODEL_SECTION_KEYS: frozenset[str] = frozenset({"path"})
_DATA_SECTION_KEYS: frozenset[str] = frozenset({"harmful", "harmless", "borderline"})
_OUTPUT_SECTION_KEYS: frozenset[str] = frozenset({"dir"})

_DATACLASS_SECTION_SPECS: tuple[_DataclassSectionSpec, ...] = (
    _DataclassSectionSpec("measure", MeasureConfig, {}),
    _DataclassSectionSpec("cut", CutConfig, {}),
    _DataclassSectionSpec(
        "eval",
        EvalConfig,
        {
            "prompts_path": "prompts",
            "refusal_phrases_path": "refusal_phrases",
        },
    ),
    _DataclassSectionSpec("surface", SurfaceConfig, {"prompts_path": "prompts"}),
    _DataclassSectionSpec("detect", DetectConfig, {}),
    _DataclassSectionSpec("optimize", OptimizeConfig, {}),
    _DataclassSectionSpec("compose_optimize", ComposeOptimizeConfig, {}),
    _DataclassSectionSpec("softprompt", SoftPromptConfig, {}),
    _DataclassSectionSpec("sic", SICConfig, {}),
    _DataclassSectionSpec("depth", DepthConfig, {}),
    _DataclassSectionSpec("probe", ProbeConfig, {}),
    _DataclassSectionSpec("steer", SteerConfig, {}),
    _DataclassSectionSpec("cast", CastConfig, {}),
    _DataclassSectionSpec("svf", SVFConfig, {}),
    _DataclassSectionSpec("api_eval", ApiEvalConfig, {}),
    _DataclassSectionSpec("meta", MetaConfig, {}),
    _DataclassSectionSpec("environment", EnvironmentConfig, {}),
    _DataclassSectionSpec("scan", ScanConfig, {}),
    _DataclassSectionSpec("policy", PolicyConfig, {}),
    _DataclassSectionSpec("intent", IntentConfig, {}),
    _DataclassSectionSpec("defend", DefenseStackConfig, {}),
    _DataclassSectionSpec("circuit", CircuitConfig, {}),
    _DataclassSectionSpec("features", FeaturesConfig, {}),
    _DataclassSectionSpec("linear_probe", LinearProbeConfig, {}),
    _DataclassSectionSpec("fusion", FusionConfig, {}),
    _DataclassSectionSpec("repbend", RepBendConfig, {}),
)

_MANUAL_SECTION_KEYS: dict[str, frozenset[str]] = {
    "model": _MODEL_SECTION_KEYS,
    "data": _DATA_SECTION_KEYS,
    "output": _OUTPUT_SECTION_KEYS,
}

KNOWN_SECTION_KEYS: dict[str, frozenset[str]] = {
    **_MANUAL_SECTION_KEYS,
    **{
        spec.name: frozenset(
            spec.field_aliases.get(field.name, field.name)
            for field in fields(spec.config_type)
        )
        for spec in _DATACLASS_SECTION_SPECS
    },
}

KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "backend",
        "verbose",
        *KNOWN_SECTION_KEYS.keys(),
    },
)


@dataclass(slots=True)
class _SchemaState:
    """Mutable state for recursive schema generation."""

    defs: dict[str, JsonSchema]


def generate_config_schema() -> JsonSchema:
    """Generate a JSON Schema for Vauban TOML configs."""
    state = _SchemaState(defs={})
    properties: dict[str, object] = {
        "model": _model_section_schema(),
        "data": _data_section_schema(state),
        "backend": {
            "type": "string",
            "enum": sorted(SUPPORTED_BACKENDS),
            "default": "mlx",
            "description": "Runtime tensor backend.",
        },
        "verbose": {
            "type": "boolean",
            "default": True,
            "description": "Emit progress logs to stderr.",
        },
        "output": _output_section_schema(),
    }
    for spec in _DATACLASS_SECTION_SPECS:
        properties[spec.name] = _schema_for_dataclass(
            spec.config_type,
            state,
            field_aliases=spec.field_aliases,
            description=_dataclass_description(spec.config_type),
        )

    schema: JsonSchema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Vauban Config",
        "description": (
            "Schema for Vauban TOML pipeline configs. "
            "TOML tables map to object-valued sections in this JSON representation. "
            "Note: [model] is optional for standalone api_eval "
            "(token_text set) but JSON Schema cannot express conditional "
            "requirements, so it is listed as required here."
        ),
        "type": "object",
        "properties": properties,
        "required": ["model"],
        "additionalProperties": False,
    }
    if state.defs:
        schema["$defs"] = state.defs
    return schema


def write_config_schema(path: str | Path) -> Path:
    """Write the generated JSON Schema to *path*."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(generate_config_schema(), indent=2) + "\n")
    return path_obj


def _model_section_schema() -> JsonSchema:
    """Schema for the required [model] section."""
    return {
        "type": "object",
        "description": "Model selection.",
        "properties": {
            "path": {
                "type": "string",
                "description": "Model id or local model directory.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _data_section_schema(state: _SchemaState) -> JsonSchema:
    """Schema for the optional [data] section."""
    state.defs["DataSourceTable"] = {
        "type": "object",
        "description": "Structured HuggingFace dataset reference.",
        "properties": {
            "hf": {"type": "string"},
            "split": {"type": "string", "default": "train"},
            "column": {"type": "string", "default": "prompt"},
            "config": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["hf"],
        "additionalProperties": False,
    }
    source_schema: JsonSchema = {
        "anyOf": [
            {"type": "string"},
            {"$ref": "#/$defs/DataSourceTable"},
        ],
    }
    return {
        "type": "object",
        "description": (
            "Prompt datasets. Required for most pipelines; depth-only configs "
            "may omit [data]."
        ),
        "properties": {
            "harmful": source_schema,
            "harmless": source_schema,
            "borderline": source_schema,
        },
        "required": ["harmful", "harmless"],
        "additionalProperties": False,
    }


def _output_section_schema() -> JsonSchema:
    """Schema for the optional [output] section."""
    return {
        "type": "object",
        "description": "Output location for reports and exported artifacts.",
        "properties": {
            "dir": {"type": "string"},
        },
        "additionalProperties": False,
    }


def _schema_for_dataclass(
    cls: type[object],
    state: _SchemaState,
    *,
    field_aliases: dict[str, str],
    description: str,
) -> JsonSchema:
    """Build an object schema for a dataclass."""
    type_hints = cast("dict[str, object]", get_type_hints(cls))
    properties: dict[str, object] = {}
    required: list[str] = []
    field_info: tuple[Field[object], ...] = cast(
        "tuple[Field[object], ...]",
        fields(cls),
    )

    for field in field_info:
        annotation = type_hints.get(field.name, field.type)
        property_name = field_aliases.get(field.name, field.name)
        field_schema = _schema_for_type(annotation, state)
        default = _default_for_field(field)
        if default is not None:
            field_schema["default"] = default
        properties[property_name] = field_schema
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(property_name)

    section_schema: JsonSchema = {
        "type": "object",
        "description": description,
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        section_schema["required"] = required
    return section_schema


def _schema_for_type(annotation: object, state: _SchemaState) -> JsonSchema:
    """Convert a Python annotation into JSON Schema."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return _schema_for_union(get_args(annotation), state)
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin in (list, set):
        args = get_args(annotation)
        items_schema = (
            _schema_for_type(args[0], state)
            if args
            else {"type": "object"}
        )
        return {"type": "array", "items": items_schema}
    if origin is tuple:
        return _schema_for_tuple(get_args(annotation), state)
    if origin is dict:
        args = get_args(annotation)
        value_annotation = args[1] if len(args) == 2 else object
        return {
            "type": "object",
            "additionalProperties": _schema_for_type(value_annotation, state),
        }
    if annotation in (str, Path):
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _schema_ref_for_dataclass(annotation, state)
    return {"type": "object"}


def _schema_for_union(
    union_args: tuple[object, ...],
    state: _SchemaState,
) -> JsonSchema:
    """Build a schema for a union annotation."""
    non_none_args = [arg for arg in union_args if arg is not NoneType]
    if len(non_none_args) == 1:
        return _schema_for_type(non_none_args[0], state)
    return {"anyOf": [_schema_for_type(arg, state) for arg in non_none_args]}


def _schema_for_tuple(
    tuple_args: tuple[object, ...],
    state: _SchemaState,
) -> JsonSchema:
    """Build a schema for a tuple annotation."""
    if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
        return {
            "type": "array",
            "items": _schema_for_type(tuple_args[0], state),
        }
    prefix_items = [_schema_for_type(arg, state) for arg in tuple_args]
    return {
        "type": "array",
        "prefixItems": prefix_items,
        "minItems": len(prefix_items),
        "maxItems": len(prefix_items),
    }


def _schema_ref_for_dataclass(
    cls: type[object],
    state: _SchemaState,
) -> JsonSchema:
    """Register a reusable dataclass schema and return a ref to it."""
    def_name = cls.__name__
    if def_name not in state.defs:
        state.defs[def_name] = _schema_for_dataclass(
            cls,
            state,
            field_aliases={},
            description=_dataclass_description(cls),
        )
    return {"$ref": f"#/$defs/{def_name}"}


def _default_for_field(field: Field[object]) -> object | None:
    """Return a JSON-compatible default for a dataclass field when available."""
    if field.default is not MISSING:
        return _json_compatible_default(field.default)
    if field.default_factory is not MISSING:
        default_factory = cast("Callable[[], object]", field.default_factory)
        return _json_compatible_default(default_factory())
    return None


def _json_compatible_default(value: object) -> object | None:
    """Convert a default value to a JSON-compatible scalar/container."""
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_compatible_default(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible_default(v) for key, v in value.items()}
    return None


def _dataclass_description(cls: type[object]) -> str:
    """Return a normalized one-line description for a dataclass."""
    doc = cls.__doc__
    if doc is None:
        return cls.__name__
    return " ".join(line.strip() for line in doc.strip().splitlines())
