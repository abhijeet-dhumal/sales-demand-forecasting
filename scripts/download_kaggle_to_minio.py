"""Download the Walmart Kaggle dataset, package notebook inputs, and upload to MinIO."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import boto3
import kagglehub
from botocore.config import Config

REQUIRED_FILES = ("features.csv", "stores.csv", "test.csv", "train.csv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_dataset_files(dataset_dir: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for filename in REQUIRED_FILES:
        matches = [path for path in dataset_dir.rglob(filename) if path.is_file()]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {filename} in the downloaded dataset; found {len(matches)}"
            )
        selected[filename] = matches[0]
    return selected


def make_notebook_archive(files: dict[str, Path], archive_path: Path) -> dict[str, dict[str, object]]:
    input_metadata: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for filename in REQUIRED_FILES:
            source = files[filename]
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with source.open("rb") as source_stream, archive.open(info, "w") as archive_stream:
                shutil.copyfileobj(source_stream, archive_stream)
            input_metadata[filename] = {
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
    return input_metadata


def main() -> None:
    dataset_handle = os.environ["KAGGLE_DATASET_HANDLE"]
    endpoint_url = os.environ["MINIO_ENDPOINT"]
    bucket = os.environ["MINIO_BUCKET"]
    object_key = os.environ["MINIO_OBJECT_KEY"]
    manifest_key = os.environ["MINIO_MANIFEST_OBJECT_KEY"]
    cache_dir = Path(os.environ.get("KAGGLEHUB_CACHE", "/tmp/kagglehub-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset {dataset_handle}")
    downloaded_dir = Path(kagglehub.dataset_download(dataset_handle, output_dir=str(cache_dir)))
    files = find_dataset_files(downloaded_dir)

    with tempfile.TemporaryDirectory(prefix="walmart-import-", dir="/tmp") as temp_dir:
        archive_path = Path(temp_dir) / "walmart-sales-forecast.zip"
        input_metadata = make_notebook_archive(files, archive_path)
        archive_sha256 = sha256_file(archive_path)
        archive_size = archive_path.stat().st_size
        provenance = {
            "dataset_handle": dataset_handle,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_object": f"s3://{bucket}/{object_key}",
            "archive_size_bytes": archive_size,
            "archive_sha256": archive_sha256,
            "source_files": input_metadata,
            "packages": {"kagglehub": version("kagglehub"), "boto3": version("boto3")},
        }

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name="us-east-1",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        client.head_bucket(Bucket=bucket)
        client.upload_file(
            str(archive_path),
            bucket,
            object_key,
            ExtraArgs={
                "ContentType": "application/zip",
                "Metadata": {"sha256": archive_sha256, "dataset-handle": dataset_handle},
            },
        )
        client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )

        uploaded = client.head_object(Bucket=bucket, Key=object_key)
        if uploaded["ContentLength"] != archive_size:
            raise RuntimeError("Uploaded ZIP size does not match the local archive")
        if uploaded.get("Metadata", {}).get("sha256") != archive_sha256:
            raise RuntimeError("Uploaded ZIP checksum metadata does not match the local archive")
        remote_digest = hashlib.sha256()
        response = client.get_object(Bucket=bucket, Key=object_key)
        for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
            remote_digest.update(chunk)
        if remote_digest.hexdigest() != archive_sha256:
            raise RuntimeError("Uploaded ZIP content checksum does not match the local archive")
        client.head_object(Bucket=bucket, Key=manifest_key)

    print(f"Uploaded s3://{bucket}/{object_key} ({archive_size} bytes, sha256={archive_sha256})")
    print(f"Uploaded provenance s3://{bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
