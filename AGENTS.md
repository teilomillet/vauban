# Agents Guidelines

## Typing Rules (STRICT)

- **All code must be fully typed.** Every function parameter, return type, and variable annotation must have explicit types.
- **`Any` is prohibited.** Never use `typing.Any` or `Any` in any context. Find or create the correct type instead.
- **`None` types must be explicit.** Use `X | None` instead of `Optional[X]`. Never leave `None` returns untyped.
- **No untyped collections.** Always use `list[str]`, `dict[str, int]`, etc. — never bare `list`, `dict`, `set`, `tuple`.
- **Use modern typing syntax.** Prefer `X | Y` over `Union[X, Y]`, `list[X]` over `List[X]` (Python 3.12+).

## Tools

- **Linter:** `ruff` — run `uv run ruff check .` before committing.
- **Type checker:** `ty` — run `uv run ty check` before committing.
- **Tests:** `pytest` — run `uv run pytest` before committing.

## Code Style

- Follow `ruff` rules configured in `pyproject.toml`.
- All public functions and classes must have docstrings.
- Imports must be sorted (enforced by ruff `I` rules).
