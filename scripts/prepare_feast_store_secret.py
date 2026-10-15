#!/usr/bin/env python3
"""Create the Feast Operator's expected SQL/Postgres Secret from postgres-secret.

The input Secret's credential values are read in memory and never printed. The
output is a separate Secret with `sql` and `postgres` YAML entries, as required
by the Feast Operator's database persistence configuration.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys


def yaml_string(value: str) -> str:
    # JSON string quoting is also valid YAML double-quoted scalar syntax.
    return json.dumps(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--source-secret", default="postgres-secret")
    parser.add_argument("--output-secret", default="feast-data-stores")
    args = parser.parse_args()

    result = subprocess.run(
        [
            "oc",
            "get",
            "secret",
            args.source_secret,
            "-n",
            args.namespace,
            "-o",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source = json.loads(result.stdout)
    data = {
        key: base64.b64decode(value).decode("utf-8")
        for key, value in source.get("data", {}).items()
    }
    required = ("url", "host", "port", "database", "username", "password")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(
            f"Secret {args.source_secret!r} is missing required keys: {', '.join(missing)}"
        )

    sql_config = (
        f"path: {yaml_string(data['url'])}\n"
        "cache_ttl_seconds: 60\n"
    )
    postgres_config = (
        f"host: {yaml_string(data['host'])}\n"
        f"port: {int(data['port'])}\n"
        f"database: {yaml_string(data['database'])}\n"
        "db_schema: public\n"
        f"user: {yaml_string(data['username'])}\n"
        f"password: {yaml_string(data['password'])}\n"
    )
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": args.output_secret,
            "namespace": args.namespace,
            "labels": {
                "app.kubernetes.io/name": "sales-demand-forecasting",
                "app.kubernetes.io/component": "feast-datastores",
                "app.kubernetes.io/part-of": "sales-demand-forecasting-demo",
            },
        },
        "type": "Opaque",
        "stringData": {"sql": sql_config, "postgres": postgres_config},
    }
    apply_result = subprocess.run(
        ["oc", "apply", "-f", "-"],
        input=json.dumps(secret),
        check=True,
        capture_output=True,
        text=True,
    )
    print(apply_result.stdout.strip())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
        # Never dump subprocess inputs or Secret contents on failure.
        detail = getattr(exc, "stderr", "")
        print(f"Failed to prepare Feast database Secret: {detail or exc}", file=sys.stderr)
        raise SystemExit(1)
