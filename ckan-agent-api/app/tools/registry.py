"""Tool registry: YAML catalog + Python handler (spec Increment 6a).

Mirrors ``PersonaRegistry``. Each catalog entry declares a tool's metadata and a dotted
``handler`` path (``module:function``); the handler takes a validated ``args`` dict and
returns a JSON-serializable result. The registry discovers ``catalog/*.yaml`` (a file may
hold one tool dict or a ``{tools: [...]}`` list), validates loudly, resolves handlers at
load time (fail fast), and exposes schema generation (`to_openai_tools`) and `invoke`.

Security (mirrors the persona/schema registries): ``yaml.safe_load`` only; catalog files
are path-confined to the tools directory. Only read-only tools are shipped — there is a
``read_only`` flag and a guard so a write tool cannot be silently registered (spec R4).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from app.settings import get_settings
from app.tools.results import tool_error, tool_success

REQUIRED_KEYS = {"name", "summary", "handler"}


class ToolError(RuntimeError):
    """Raised when a tool catalog entry is missing, malformed, or invalid."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    handler_path: str
    description: str = ""
    category: str = "general"
    read_only: bool = True
    args: dict[str, Any] = field(default_factory=dict)
    returns: dict[str, Any] = field(default_factory=dict)
    use_when: list[str] = field(default_factory=list)
    safety: list[str] = field(default_factory=list)
    handler: Callable[[dict[str, Any]], Any] | None = None

    def required_args(self) -> list[str]:
        return [k for k, spec in self.args.items() if isinstance(spec, dict) and spec.get("required")]

    def to_openai_tool(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for arg_name, spec in self.args.items():
            spec = spec if isinstance(spec, dict) else {}
            prop: dict[str, Any] = {"type": spec.get("type", "string")}
            if spec.get("description"):
                prop["description"] = spec["description"]
            properties[arg_name] = prop
            if spec.get("required"):
                required.append(arg_name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or self.summary,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


def _resolve_handler(path: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in path:
        raise ToolError(f"handler must be 'module:function', got {path!r}")
    module_name, func_name = path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
        handler = getattr(module, func_name)
    except (ImportError, AttributeError) as exc:
        raise ToolError(f"could not resolve handler {path!r}: {exc}") from exc
    if not callable(handler):
        raise ToolError(f"handler {path!r} is not callable")
    return handler


def _spec_from_dict(entry: dict[str, Any], source: Path) -> ToolSpec:
    missing = REQUIRED_KEYS - set(entry)
    if missing:
        raise ToolError(f"tool in {source.name} missing required key(s): {sorted(missing)}")
    read_only = bool(entry.get("read_only", True))
    if not read_only:
        # Spec R4: only read/dry-run tools are exposed; refuse to register write tools.
        raise ToolError(
            f"tool {entry['name']!r} is read_only=false; write tools are not allowed in the "
            f"tool registry (writes stay in the gated graph path)."
        )
    return ToolSpec(
        name=str(entry["name"]).strip(),
        summary=str(entry["summary"]).strip(),
        handler_path=str(entry["handler"]).strip(),
        description=str(entry.get("description") or "").strip(),
        category=str(entry.get("category") or "general").strip(),
        read_only=True,
        args=dict(entry.get("args") or {}),
        returns=dict(entry.get("returns") or {}),
        use_when=list(entry.get("use_when") or []),
        safety=list(entry.get("safety") or []),
        handler=_resolve_handler(str(entry["handler"]).strip()),
    )


class ToolRegistry:
    def __init__(self, tools_dir: Path | None = None) -> None:
        settings = get_settings()
        self.tools_dir = (tools_dir or settings.tools_dir).resolve()

    def _confined(self, path: Path) -> Path:
        resolved = path.resolve()
        if self.tools_dir != resolved and self.tools_dir not in resolved.parents:
            raise ToolError(f"tool catalog file resolves outside the tools directory: {path}")
        return resolved

    def _parse_file(self, path: Path) -> list[ToolSpec]:
        resolved = self._confined(path)
        try:
            data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ToolError(f"tool catalog is not valid YAML ({path}): {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("tools"), list):
            entries = data["tools"]
        elif isinstance(data, dict) and "name" in data:
            entries = [data]
        else:
            raise ToolError(f"tool catalog {path.name} must be a tool dict or a {{tools: [...]}} list")
        return [_spec_from_dict(entry, resolved) for entry in entries if isinstance(entry, dict)]

    def load_all(self) -> list[ToolSpec]:
        if not self.tools_dir.is_dir():
            raise ToolError(f"tools directory does not exist: {self.tools_dir}")
        specs: list[ToolSpec] = []
        for path in sorted(self.tools_dir.glob("*.yaml")):
            specs.extend(self._parse_file(path))
        names = [s.name for s in specs]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ToolError(f"duplicate tool name(s): {dupes}")
        return specs

    def get(self, name: str) -> ToolSpec:
        for spec in self.load_all():
            if spec.name == name:
                return spec
        raise ToolError(f"tool not found: {name!r}")

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        specs = self.load_all()
        if names is not None:
            allow = set(names)
            specs = [s for s in specs if s.name in allow]
        return [s.to_openai_tool() for s in specs]

    def invoke(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate args against the catalog and run the handler. Returns a result envelope."""
        args = dict(args or {})
        try:
            spec = self.get(name)
        except ToolError as exc:
            return tool_error(name, "unknown_tool", str(exc))

        missing = [a for a in spec.required_args() if a not in args or args[a] in (None, "")]
        if missing:
            return tool_error(name, "invalid_args", f"missing required arg(s): {missing}")
        for arg_name, arg_spec in spec.args.items():
            if arg_name not in args and isinstance(arg_spec, dict) and "default" in arg_spec:
                args[arg_name] = arg_spec["default"]

        try:
            result = spec.handler(args)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - surface as a structured tool error
            return tool_error(name, "handler_error", str(exc))
        return tool_success(name, result)
