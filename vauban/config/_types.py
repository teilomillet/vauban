# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases for the config package."""

# tomllib.load() returns dict[str, Any]. We use this alias to avoid bare Any
# while acknowledging TOML values are dynamically typed.
type TomlDict = dict[str, object]
