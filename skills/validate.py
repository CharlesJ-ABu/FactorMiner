"""Validate all publishable Agent Skills in this repository."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^[a-z0-9-]{1,63}$")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def parse_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError("SKILL.md frontmatter is not closed") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, text


def validate_skill(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        return [f"{skill_dir.name}: missing SKILL.md"]

    try:
        metadata, text = parse_frontmatter(skill_file)
    except ValueError as exc:
        return [f"{skill_dir.name}: {exc}"]

    if set(metadata) != {"name", "description"}:
        errors.append(
            f"{skill_dir.name}: frontmatter keys must be exactly name and description"
        )
    name = metadata.get("name", "")
    if name != skill_dir.name:
        errors.append(f"{skill_dir.name}: frontmatter name must match directory")
    if not NAME_RE.fullmatch(name):
        errors.append(f"{skill_dir.name}: invalid skill name")
    if not metadata.get("description"):
        errors.append(f"{skill_dir.name}: description must not be empty")
    if "TODO" in text:
        errors.append(f"{skill_dir.name}: unresolved TODO")

    for target in LINK_RE.findall(text):
        if (
            target.startswith(("http://", "https://", "#", "mailto:"))
            or "://" in target
        ):
            continue
        resolved = (skill_dir / target.split("#", 1)[0]).resolve()
        try:
            resolved.relative_to(skill_dir.resolve())
        except ValueError:
            errors.append(f"{skill_dir.name}: link escapes skill directory: {target}")
            continue
        if not resolved.exists():
            errors.append(f"{skill_dir.name}: broken local link: {target}")

    interface_file = skill_dir / "agents" / "openai.yaml"
    if not interface_file.is_file():
        errors.append(f"{skill_dir.name}: missing agents/openai.yaml")
    else:
        interface = interface_file.read_text(encoding="utf-8")
        for field in ("display_name:", "short_description:", "default_prompt:"):
            if field not in interface:
                errors.append(f"{skill_dir.name}: openai.yaml missing {field[:-1]}")
        if f"${name}" not in interface:
            errors.append(f"{skill_dir.name}: default_prompt must mention ${name}")

    return errors


def main() -> int:
    skill_dirs = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not skill_dirs:
        print("No skills found.", file=sys.stderr)
        return 1

    errors = [
        error
        for skill_dir in skill_dirs
        for error in validate_skill(skill_dir)
    ]
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(skill_dirs)} skill(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
