"""Fetch declared datasets into data/raw/. Nothing here is committed.

    python -m data.download --all
    python -m data.download --dataset uci-cbm

Refuses to fetch anything not declared in `data/registry.py`, so every byte in
`data/raw/` is traceable to a licence and a citation.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import httpx

from data.registry import REGISTRY, Dataset, get

RAW = Path(__file__).parent / "raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(dataset: Dataset, *, force: bool = False) -> Path:
    target = RAW / dataset.key
    marker = target / ".fetched"

    if marker.exists() and not force:
        print(f"  {dataset.key}: already present ({target}). Use --force to refetch.")
        return target

    target.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if dataset.archive else Path(dataset.url).suffix or ".dat"
    archive_path = target / f"{dataset.key}{suffix}"

    print(f"  {dataset.key}: fetching {dataset.url}")
    with httpx.stream("GET", dataset.url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with archive_path.open("wb") as handle:
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)

    size_mb = archive_path.stat().st_size / 1e6
    checksum = _sha256(archive_path)
    print(f"  {dataset.key}: {size_mb:.1f} MB  sha256={checksum[:16]}...")

    if dataset.archive:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target)
            print(f"  {dataset.key}: extracted {len(zipfile.ZipFile(archive_path).namelist())} entries")
        except zipfile.BadZipFile:
            print(
                f"  {dataset.key}: NOT a zip archive. The source may have moved or now "
                f"require a login. Verify {dataset.url} by hand before citing it.",
                file=sys.stderr,
            )
            raise

    marker.write_text(f"{dataset.url}\nsha256={checksum}\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch declared public datasets.")
    parser.add_argument("--dataset", action="append", choices=sorted(REGISTRY), default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="Refetch even if present.")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Pass --all or --dataset KEY.")

    keys = sorted(REGISTRY) if args.all else args.dataset
    failures: list[str] = []

    for key in keys:
        dataset = get(key)
        print(f"\n{dataset.name}")
        print(f"  licence: {dataset.licence}")
        try:
            fetch(dataset, force=args.force)
        except Exception as exc:  # noqa: BLE001 - report and continue; one dead link
            failures.append(key)  # should not block the other downloads
            print(f"  {key}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} dataset(s) failed: {', '.join(failures)}. "
            "Do not cite a source in the deck that did not download.",
            file=sys.stderr,
        )
        return 1

    print(f"\n{len(keys)} dataset(s) ready in {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
