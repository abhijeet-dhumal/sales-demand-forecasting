"""Small stdlib-only .env loader and project template renderer."""

from __future__ import annotations

import os
import re
import textwrap
import json
from urllib.parse import quote
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_ENV_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_TEMPLATE_TOKEN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def find_project_root(start: Path | None = None) -> Path:
    """Find the checked-out project containing the local .env file."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".env").is_file() and (candidate / "project_config.py").is_file():
            return candidate
    raise FileNotFoundError("Project .env not found; copy .env.example to .env in the repo root")


def load_config(root: Path = ROOT) -> dict[str, str]:
    """Read .env, expand references between keys, and allow process overrides."""
    env_path = root / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(f"Missing {env_path}; copy .env.example to .env first")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at line {line_number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid .env key at line {line_number}: {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value

    derived_database_urls = {
        key for key in ("POSTGRES_URL", "POSTGRES_PY_URL")
        if "${" in values.get(key, "")
    }

    # Environment overrides are applied before expansion so dependent values
    # (for example POSTGRES_HOST) follow an overridden namespace.
    environment_overrides = {key: value for key, value in os.environ.items() if key in values}
    values.update(environment_overrides)

    for _ in range(len(values) + 1):
        changed = False
        for key, value in values.items():
            expanded = _ENV_REFERENCE.sub(lambda match: values.get(match.group(1), match.group(0)), value)
            if expanded != value:
                values[key] = expanded
                changed = True
        if not changed:
            break
    unresolved = sorted({name for value in values.values() for name in _ENV_REFERENCE.findall(value)})
    if unresolved:
        raise ValueError(f"Unresolved or cyclic .env references: {', '.join(unresolved)}")

    # Passwords commonly contain URL-reserved characters. Quote only values
    # used to derive these URLs; explicit full-URL overrides remain untouched.
    username = quote(values.get("POSTGRES_USERNAME", ""), safe="")
    password = quote(values.get("POSTGRES_PASSWORD", ""), safe="")
    host = values.get("POSTGRES_HOST", "")
    port = values.get("POSTGRES_PORT", "")
    database = quote(values.get("POSTGRES_DATABASE", ""), safe="")
    if "POSTGRES_URL" in derived_database_urls and "POSTGRES_URL" not in environment_overrides:
        values["POSTGRES_URL"] = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    if "POSTGRES_PY_URL" in derived_database_urls and "POSTGRES_PY_URL" not in environment_overrides:
        values["POSTGRES_PY_URL"] = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

    return values


def render_text(text: str, values: dict[str, str]) -> str:
    """Replace {{KEY}} tokens and fail if a template key is missing."""
    missing = sorted({key for key in _TEMPLATE_TOKEN.findall(text) if key not in values})
    if missing:
        raise ValueError(f"Missing .env values: {', '.join(missing)}")

    def replace(match: re.Match[str]) -> str:
        value = values[match.group(1)]
        before = text[match.start() - 1 : match.start()]
        after = text[match.end() : match.end() + 1]
        if before == after == '"':
            return json.dumps(value, ensure_ascii=False)[1:-1]
        if before == after == "'":
            return value.replace("'", "''")
        return value

    return _TEMPLATE_TOKEN.sub(replace, text)


def _inline_forecast_contract(root: Path, text: str) -> str:
    marker = "@@FORECAST_CONTRACT_SOURCE@@"
    if marker not in text:
        return text
    contract_source = root / "feature_repo" / "forecast_contract.py"
    if not contract_source.is_file():
        raise FileNotFoundError(f"Missing shared forecast contract: {contract_source}")
    return text.replace(marker, textwrap.indent(contract_source.read_text().rstrip(), "    "))


def _inline_kaggle_importer(root: Path, text: str) -> str:
    marker = "@@KAGGLE_IMPORT_SCRIPT@@"
    if marker not in text:
        return text
    script_source = root / "scripts" / "download_kaggle_to_minio.py"
    if not script_source.is_file():
        raise FileNotFoundError(f"Missing Kaggle importer script: {script_source}")
    return text.replace(marker, textwrap.indent(script_source.read_text().rstrip(), "      "))


def _render_feature_repo(root: Path, values: dict[str, str], output: Path) -> None:
    source_dir = root / "feature_repo"
    target_dir = output / "feature_repo"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.yaml"):
        rendered = render_text(source.read_text(), values)
        rendered = _inline_forecast_contract(root, rendered)
        target = target_dir / source.name
        target.write_text(rendered)
        try:
            target.chmod(0o600)
        except OSError:
            pass

    contract_source = source_dir / "forecast_contract.py"
    if contract_source.is_file():
        contract_target = target_dir / contract_source.name
        contract_target.write_text(contract_source.read_text())
        try:
            contract_target.chmod(0o600)
        except OSError:
            pass


def _validate_database_config(values: dict[str, str], *, require_admin: bool = False) -> None:
    placeholders = {"", "replace-me", "changeme", "change-me", "password", "example"}
    keys = ["POSTGRES_PASSWORD"]
    if require_admin:
        keys.append("POSTGRES_ADMIN_PASSWORD")
    for key in keys:
        value = values.get(key, "")
        if value.casefold() in placeholders or len(value) < 16 or "\n" in value or "\r" in value:
            raise ValueError(f"Set {key} to a unique secret of at least 16 characters before rendering project files")

    if require_admin and values["POSTGRES_PASSWORD"] == values["POSTGRES_ADMIN_PASSWORD"]:
        raise ValueError("POSTGRES_PASSWORD and POSTGRES_ADMIN_PASSWORD must be different")


def _validate_manifest_config(values: dict[str, str]) -> None:
    _validate_database_config(values, require_admin=True)

    for key in ("POSTGRES_IMAGE", "RAY_IMAGE", "PREDICTOR_IMAGE"):
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", values.get(key, "")):
            raise ValueError(f"{key} must use an immutable image digest (image@sha256:<64 hex characters>)")

    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", values.get("KAGGLE_IMPORT_IMAGE", "")):
        raise ValueError("KAGGLE_IMPORT_IMAGE must use an immutable image digest (image@sha256:<64 hex characters>)")

    if not values.get("MLFLOW_TRACKING_URI", "").startswith("https://"):
        raise ValueError("MLFLOW_TRACKING_URI must use HTTPS")
    if values.get("MLFLOW_INSECURE_TLS", "false").casefold() not in {"false", "0", "no"}:
        raise ValueError("MLFLOW_INSECURE_TLS must remain false")


def render_project(root: Path = ROOT) -> Path:
    """Render Kubernetes and Feast YAML into an ignored, private output folder."""
    values = load_config(root)
    _validate_manifest_config(values)
    output = root / ".generated"
    output.mkdir(mode=0o700, exist_ok=True)
    try:
        output.chmod(0o700)
    except OSError:
        pass

    _render_feature_repo(root, values, output)

    source_dir = root / "manifests"
    target_dir = output / "manifests"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.yaml"):
        target = target_dir / source.name
        rendered = render_text(source.read_text(), values)
        rendered = _inline_forecast_contract(root, rendered)
        rendered = _inline_kaggle_importer(root, rendered)
        target.write_text(rendered)
        try:
            target.chmod(0o600)
        except OSError:
            pass

    return output


def render_feature_repo(root: Path = ROOT) -> Path:
    """Render only Feast client configuration for notebooks, without Kubernetes Secrets."""
    values = load_config(root)
    _validate_database_config(values)
    output = root / ".generated"
    output.mkdir(mode=0o700, exist_ok=True)
    try:
        output.chmod(0o700)
    except OSError:
        pass
    _render_feature_repo(root, values, output)
    return output / "feature_repo"
