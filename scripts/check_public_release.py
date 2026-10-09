"""Fail if a public GENOVA checkout contains TÜSEP test CSV files or CSV data."""

from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
blocked_suffixes = {".csv", ".zip"}
blocked_parts = {".venv", "__pycache__"}
protected = {
    f"{panel}/input/{panel}.csv"
    for panel in ("CFTR", "KANSER", "MASTER", "PAH")
}
panels = {"CFTR", "KANSER", "MASTER", "PAH"}


def audit(paths: list[str], label: str) -> None:
    bad = []
    for name in paths:
        path = Path(name)
        parts = path.parts
        panel_input = (
            len(parts) >= 3
            and parts[0] in panels
            and parts[1].lower() == "input"
            and path.name != "README_INPUT.md"
        )
        lower_parts = tuple(part.lower() for part in parts)
        result_json = (
            path.suffix.lower() == ".json"
            and (
                any(part in {"output", "results", "reports"} for part in lower_parts)
                or lower_parts[:2] == ("genova_json_merger", "input")
                or any(lower_parts[i:i + 2] == ("data", "splits") for i in range(len(lower_parts) - 1))
                or any(lower_parts[i:i + 2] == ("data", "processed") for i in range(len(lower_parts) - 1))
                or path.name.upper() == "PACKAGE_VALIDATION.JSON"
            )
        )
        if (
            name.replace("\\", "/") in protected
            or panel_input
            or result_json
            or any(parts[i].lower() == "data" and parts[i + 1].lower() == "raw" for i in range(len(parts) - 1))
            or path.suffix.lower() in blocked_suffixes
            or any(part.lower() in blocked_parts for part in path.parts)
        ):
            bad.append(name)
    if bad:
        raise SystemExit(f"BLOCKED {label}:\n" + "\n".join(sorted(bad)))
    print(f"PASS {label}: {len(paths)} files checked")


disk_files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
audit(disk_files, "working tree")

if (root / ".git").exists():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    tracked = [p for p in result.stdout.decode("utf-8").split("\0") if p]
    audit(tracked, "Git index")
