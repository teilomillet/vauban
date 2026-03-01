"""Low-overhead typed helpers for TOML section parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


@dataclass(frozen=True, slots=True)
class _Missing:
    """Sentinel for omitted defaults."""


_MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class SectionReader:
    """Typed accessors for a TOML table with consistent error formatting."""

    section: str
    data: TomlDict

    def literal(
        self,
        key: str,
        choices: tuple[str, ...],
        *,
        default: str | _Missing = _MISSING,
    ) -> str:
        """Read a required or defaulted literal string."""
        raw = self._raw(key, default)
        if not isinstance(raw, str):
            self._raise_type_error(key, "a string", raw)
        if raw not in choices:
            msg = (
                f"{self._field(key)} must be one of {choices!r},"
                f" got {raw!r}"
            )
            raise ValueError(msg)
        if isinstance(raw, str):
            return raw
        msg = f"{self._field(key)} must be a string, got {type(raw).__name__}"
        raise TypeError(msg)

    def optional_literal(
        self,
        key: str,
        choices: tuple[str, ...],
    ) -> str | None:
        """Read an optional literal string."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, str):
            self._raise_type_error(key, "a string", raw)
        if raw not in choices:
            msg = (
                f"{self._field(key)} must be one of {choices!r},"
                f" got {raw!r}"
            )
            raise ValueError(msg)
        if isinstance(raw, str):
            return raw
        msg = f"{self._field(key)} must be a string, got {type(raw).__name__}"
        raise TypeError(msg)

    def string(
        self,
        key: str,
        *,
        default: str | _Missing = _MISSING,
    ) -> str:
        """Read a required or defaulted string."""
        raw = self._raw(key, default)
        if not isinstance(raw, str):
            self._raise_type_error(key, "a string", raw)
        if isinstance(raw, str):
            return raw
        msg = f"{self._field(key)} must be a string, got {type(raw).__name__}"
        raise TypeError(msg)

    def optional_string(self, key: str) -> str | None:
        """Read an optional string."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, str):
            self._raise_type_error(key, "a string", raw)
        if isinstance(raw, str):
            return raw
        msg = f"{self._field(key)} must be a string, got {type(raw).__name__}"
        raise TypeError(msg)

    def integer(
        self,
        key: str,
        *,
        default: int | _Missing = _MISSING,
    ) -> int:
        """Read a required or defaulted integer."""
        raw = self._raw(key, default)
        if not isinstance(raw, int):
            msg = (
                f"{self._field(key)} must be an integer,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        return raw

    def optional_integer(self, key: str) -> int | None:
        """Read an optional integer."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, int):
            msg = (
                f"{self._field(key)} must be an integer,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        return raw

    def number(
        self,
        key: str,
        *,
        default: float | int | _Missing = _MISSING,
    ) -> float:
        """Read a required or defaulted numeric value."""
        raw = self._raw(key, default)
        if not isinstance(raw, int | float):
            msg = (
                f"{self._field(key)} must be a number,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        return float(raw)

    def optional_number(self, key: str) -> float | None:
        """Read an optional numeric value."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, int | float):
            msg = (
                f"{self._field(key)} must be a number,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        return float(raw)

    def boolean(
        self,
        key: str,
        *,
        default: bool | _Missing = _MISSING,
    ) -> bool:
        """Read a required or defaulted boolean."""
        raw = self._raw(key, default)
        if not isinstance(raw, bool):
            msg = (
                f"{self._field(key)} must be a boolean,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        return raw

    def string_list(
        self,
        key: str,
        *,
        default: list[str] | _Missing = _MISSING,
        coerce: bool = False,
    ) -> list[str]:
        """Read a list of strings."""
        raw = self._raw(key, default)
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        if coerce:
            return [str(item) for item in raw]
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                msg = (
                    f"{self._field(key)} elements must be strings,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            values.append(item)
        return values

    def int_list(
        self,
        key: str,
        *,
        default: list[int] | _Missing = _MISSING,
    ) -> list[int]:
        """Read a list of integers."""
        raw = self._raw(key, default)
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list of ints,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        values: list[int] = []
        for item in raw:
            if not isinstance(item, int):
                msg = (
                    f"{self._field(key)} elements must be integers,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            values.append(item)
        return values

    def optional_string_list(self, key: str) -> list[str] | None:
        """Read an optional list of strings."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list of strings,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        values: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                msg = (
                    f"{self._field(key)} elements must be strings,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            values.append(item)
        return values

    def optional_int_list(self, key: str) -> list[int] | None:
        """Read an optional list of integers."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list of ints,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        values: list[int] = []
        for item in raw:
            if not isinstance(item, int):
                msg = (
                    f"{self._field(key)} elements must be integers,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            values.append(item)
        return values

    def number_list(
        self,
        key: str,
        *,
        default: list[float] | _Missing = _MISSING,
    ) -> list[float]:
        """Read a list of numbers."""
        raw = self._raw(key, default)
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list of numbers,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        values: list[float] = []
        for item in raw:
            if not isinstance(item, int | float):
                msg = (
                    f"{self._field(key)} elements must be numbers,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            values.append(float(item))
        return values

    def optional_number_list(self, key: str) -> list[float] | None:
        """Read an optional list of numbers."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list of numbers,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        values: list[float] = []
        for item in raw:
            if not isinstance(item, int | float):
                msg = (
                    f"{self._field(key)} elements must be numbers,"
                    f" got {type(item).__name__}"
                )
                raise TypeError(msg)
            values.append(float(item))
        return values

    def str_float_table(self, key: str) -> dict[str, float]:
        """Read an optional table of string keys to float values.

        Returns an empty dict if absent.
        """
        raw = self._raw_optional(key)
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            msg = (
                f"{self._field(key)} must be a table,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        result: dict[str, float] = {}
        for k, v in raw.items():
            if not isinstance(v, int | float):
                msg = (
                    f"{self._field(key)}.{k} must be a number,"
                    f" got {type(v).__name__}"
                )
                raise TypeError(msg)
            result[str(k)] = float(v)
        return result

    def number_pairs(self, key: str) -> list[tuple[float, float]] | None:
        """Read an optional list of numeric pairs."""
        raw = self._raw_optional(key)
        if raw is None:
            return None
        if not isinstance(raw, list):
            msg = (
                f"{self._field(key)} must be a list of [threshold, alpha] pairs,"
                f" got {type(raw).__name__}"
            )
            raise TypeError(msg)
        pairs: list[tuple[float, float]] = []
        for i, pair in enumerate(raw):
            if not isinstance(pair, list) or len(pair) != 2:
                msg = f"{self._field(key)}[{i}] must be a [threshold, alpha] pair"
                raise ValueError(msg)
            if not isinstance(pair[0], int | float) or not isinstance(
                pair[1], int | float,
            ):
                msg = f"{self._field(key)}[{i}] values must be numbers"
                raise TypeError(msg)
            pairs.append((float(pair[0]), float(pair[1])))
        return pairs

    def _field(self, key: str) -> str:
        """Return the fully-qualified field path."""
        return f"{self.section}.{key}"

    def _raw(
        self,
        key: str,
        default: object | _Missing,
    ) -> object:
        """Read a field, applying a default when given."""
        if key in self.data:
            return self.data[key]
        if isinstance(default, _Missing):
            msg = f"{self._field(key)} is required"
            raise ValueError(msg)
        return default

    def _raw_optional(self, key: str) -> object | None:
        """Read an optional field."""
        return self.data.get(key)

    def _raise_type_error(self, key: str, expected: str, value: object) -> None:
        """Raise a field-specific type error."""
        msg = (
            f"{self._field(key)} must be {expected},"
            f" got {type(value).__name__}"
        )
        raise TypeError(msg)
