#!/usr/bin/env python3
"""The engine snapshot a `.devN` release embeds, pinned once for BOTH packages.

A `.devN` tag is built from the upstream SNAPSHOT engine. The JVM wheels take the jars from the Docker image
`arcadedata/arcadedb`, the native wheels resolve `arcadedb-*` from the Sonatype snapshot repository; both channels follow upstream
and move by the day. `scripts/dev_engine_pin.json` names one snapshot: the image by index digest and the sha256 of every
`arcadedb-*.jar` in it. This script proves that BOTH channels still hold exactly those jars, byte for byte, and fails closed
(exit 1 on a mismatch, exit 2 on a source it cannot read).

    dev_engine_pin.py check [--tag TAG] [--pin FILE]   # image (amd64 and arm64) and Sonatype jars against the pin
    dev_engine_pin.py capture IMAGE_DIGEST             # print a pin for an image digest (the Sonatype jars must agree)
    dev_engine_pin.py verify-jars DIR                  # the jars in DIR (a download of the image) are exactly the pinned ones
    dev_engine_pin.py shell-env TAG                    # print JVM image reference / snapshot facts a workflow needs (dev tags only)

`check --tag` applies the pin ONLY to a `.devN` tag: a final or `.postN` tag exits 1, because such a release links exactly
its release engine and never a snapshot. Tests: tests/test_dev_engine_pin.py. Output is ASCII only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET  # nosec B405 - parses the maven-metadata.xml of a fixed https repository
from pathlib import Path

PIN_FILE = Path(__file__).resolve().parent / "dev_engine_pin.json"
REGISTRY = "https://registry-1.docker.io/v2/arcadedata/arcadedb"
AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:arcadedata/arcadedb:pull"
SNAPSHOTS = "https://central.sonatype.com/repository/maven-snapshots/com/arcadedb"
INDEX_TYPES = "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json"
MANIFEST_TYPES = "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"
LIB = re.compile(
    r"^(?:\./)?home/arcadedb/lib/(arcadedb-[^/]+)-(\d+\.\d+\.\d+)-SNAPSHOT\.jar$"
)
DEV_TAG = re.compile(r"^\d+\.\d+\.\d+\.dev\d+$")
ARCHES = ("amd64", "arm64")
TIMEOUT = 60


class Unreadable(Exception):
    """A source (registry, Sonatype) could not be read: exit 2, never a pass."""


class _NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """The registry redirects a blob to a CDN that refuses a request carrying the registry token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.headers.pop("Authorization", None)
        return new


_OPENER = urllib.request.build_opener(_NoAuthRedirect)


def _open(url: str, headers: dict[str, str] | None = None):
    req = urllib.request.Request(
        url, headers=headers or {}
    )  # noqa: S310 - fixed https hosts
    try:
        return _OPENER.open(req, timeout=TIMEOUT)  # nosec B310
    except (urllib.error.URLError, TimeoutError) as e:
        raise Unreadable(f"{url}: {e}") from e


def _token() -> str:
    with _open(AUTH) as r:
        return json.load(r)["token"]


def _registry_json(ref: str, accept: str, token: str) -> dict:
    with _open(
        f"{REGISTRY}/manifests/{ref}",
        {"Authorization": f"Bearer {token}", "Accept": accept},
    ) as r:
        return json.load(r)


def image_jars(index_digest: str, arch: str, token: str) -> dict[str, str]:
    """{module: sha256} of the arcadedb-*-SNAPSHOT.jar files in /home/arcadedb/lib of one platform of the image."""
    index = _registry_json(index_digest, INDEX_TYPES, token)
    entries = [
        m
        for m in index.get("manifests", [])
        if m.get("platform", {}).get("architecture") == arch
    ]
    if len(entries) != 1:
        raise Unreadable(f"image {index_digest}: {len(entries)} manifests for {arch}")
    manifest = _registry_json(entries[0]["digest"], MANIFEST_TYPES, token)
    found: dict[str, str] = {}
    for layer in reversed(manifest["layers"]):  # a later layer wins over an earlier one
        with _open(
            f"{REGISTRY}/blobs/{layer['digest']}", {"Authorization": f"Bearer {token}"}
        ) as blob:
            with tarfile.open(fileobj=blob, mode="r|gz") as tar:
                for member in tar:
                    m = LIB.match(member.name)
                    if m and m.group(1) not in found and member.isfile():
                        h = hashlib.sha256()
                        f = tar.extractfile(member)
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                        found[m.group(1)] = h.hexdigest()
    if "arcadedb-engine" not in found:
        raise Unreadable(
            f"image {index_digest} ({arch}): no arcadedb-engine jar in /home/arcadedb/lib"
        )
    return found


def snapshot_jar(
    module: str, version: str, stamp: str | None = None
) -> tuple[str, str | None]:
    """(timestamped version, sha256) of a snapshot jar of a module: the newest one, or the one STAMP names; sha256 None when the
    repository has no such module (or no such stamp)."""
    if stamp:
        try:
            h = hashlib.sha256()
            with _open(f"{SNAPSHOTS}/{module}/{version}/{module}-{stamp}.jar") as r:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    h.update(chunk)
            return stamp, h.hexdigest()
        except Unreadable as e:
            if "404" in str(e):
                return stamp, None
            raise
    try:
        with _open(f"{SNAPSHOTS}/{module}/{version}/maven-metadata.xml") as r:
            root = ET.fromstring(r.read())  # nosec B314
    except Unreadable as e:
        if "404" in str(e):
            return "", None
        raise
    stamp = None
    for sv in root.iter("snapshotVersion"):
        if sv.findtext("extension") == "jar" and sv.findtext("classifier") is None:
            stamp = sv.findtext("value")
    if not stamp:
        raise Unreadable(f"{module}: no jar in the snapshot metadata")
    h = hashlib.sha256()
    with _open(f"{SNAPSHOTS}/{module}/{version}/{module}-{stamp}.jar") as r:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
    return stamp, h.hexdigest()


def jar_hashes(directory: Path) -> dict[str, str]:
    """{module: sha256} of the arcadedb-*-SNAPSHOT.jar files in a directory."""
    found = {}
    for jar in sorted(Path(directory).glob("arcadedb-*-SNAPSHOT.jar")):
        m = re.fullmatch(r"(arcadedb-.+)-\d+\.\d+\.\d+-SNAPSHOT\.jar", jar.name)
        if m:
            found[m.group(1)] = hashlib.sha256(jar.read_bytes()).hexdigest()
    return found


def jars_are_pinned(hashes: dict[str, str], pin: dict, strict: bool) -> bool:
    """The engine jar is there and every jar present is the pinned one; strict: the set of jars is the pinned set."""
    if hashes.get("arcadedb-engine") != pin["jars"]["arcadedb-engine"]:
        return False
    if any(pin["jars"].get(mod) != sha for mod, sha in hashes.items()):
        return False
    return not strict or set(hashes) == set(pin["jars"])


def pinned_snapshot(directory: Path, pin_path: Path = PIN_FILE) -> dict | None:
    """The pin when the jars of DIRECTORY are exactly the pinned snapshot (some pinned jars may be left out), else None."""
    if not Path(pin_path).is_file():
        return None
    pin = load_pin(Path(pin_path))
    return pin if jars_are_pinned(jar_hashes(directory), pin, strict=False) else None


def verify_jars(directory: Path) -> int:
    pin = load_pin()
    if not jars_are_pinned(jar_hashes(directory), pin, strict=True):
        print(
            f"FAIL the jars in {directory} are not the pinned snapshot {pin['snapshot']} (image {pin['image_digest']})"
        )
        return 1
    print(
        f"PASS the jars in {directory} are the pinned snapshot {pin['snapshot']}, same sha256"
    )
    return 0


def load_pin(path: Path = PIN_FILE) -> dict:
    pin = json.loads(path.read_text(encoding="utf-8"))
    for key in ("engine", "snapshot", "image_digest", "jars"):
        if key not in pin:
            raise ValueError(f"{path}: missing {key}")
    if not re.fullmatch(r"\d+\.\d+\.\d+-SNAPSHOT", pin["engine"]):
        raise ValueError(f"{path}: engine {pin['engine']!r} is not A.B.C-SNAPSHOT")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", pin["image_digest"]):
        raise ValueError(f"{path}: image_digest is not a sha256 digest")
    return pin


def tag_problem(tag: str, pin: dict) -> str | None:
    """Why a snapshot pin may not be applied to TAG, or None. Only `A.B.C.devN` of the pinned base may link a snapshot."""
    if not DEV_TAG.fullmatch(tag):
        return f"tag {tag} is not A.B.C.devN: a final or .postN release links exactly its release engine, never a snapshot"
    if tag.split(".dev")[0] != pin["engine"].removesuffix("-SNAPSHOT"):
        return f"tag {tag} has another base than the pinned engine {pin['engine']}"
    return None


def compare(
    pin: dict, image: dict[str, dict[str, str]], maven: dict[str, str | None]
) -> list[str]:
    """Every difference between the pin and the two channels; an empty list is a pass."""
    bad = []
    for arch, jars in image.items():
        if jars != pin["jars"]:
            for mod in sorted(set(jars) | set(pin["jars"])):
                if jars.get(mod) != pin["jars"].get(mod):
                    bad.append(
                        f"image {arch}: {mod} is {jars.get(mod)}, the pin says {pin['jars'].get(mod)}"
                    )
    for mod, sha in maven.items():
        if sha is not None and sha != pin["jars"].get(mod):
            bad.append(f"Sonatype: {mod} is {sha}, the pin says {pin['jars'].get(mod)}")
    for mod in ("arcadedb-engine", "arcadedb-server"):
        if maven.get(mod) is None:
            bad.append(f"Sonatype has no {mod} {pin['engine']}")
    return bad


def check(tag: str | None, pin_path: Path) -> int:
    pin = load_pin(pin_path)
    if tag:
        problem = tag_problem(tag, pin)
        if problem:
            print(f"FAIL {problem}")
            return 1
    token = _token()
    image = {arch: image_jars(pin["image_digest"], arch, token) for arch in ARCHES}
    # the pinned TIMESTAMPED jars, not the newest snapshot: the native build resolves exactly these, so a later upstream deploy changes nothing
    maven = {
        mod: snapshot_jar(mod, pin["engine"], pin["snapshot"])[1] for mod in pin["jars"]
    }
    bad = compare(pin, image, maven)
    for line in bad:
        print(f"FAIL {line}")
    if bad:
        print(
            f"FAIL the JVM image and the Sonatype jars are not the pinned snapshot {pin['snapshot']} (a purged or changed jar?); re-pin (scripts/dev_engine_pin.py capture)"
        )
        return 1
    print(
        f"PASS image {pin['image_digest']} (amd64, arm64) and Sonatype hold the pinned snapshot {pin['snapshot']}: {len(pin['jars'])} jars, same sha256"
    )
    return 0


def capture(digest: str) -> int:
    token = _token()
    jars = image_jars(digest, "amd64", token)
    stamps = {mod: snapshot_jar(mod, "26.12.1-SNAPSHOT") for mod in jars}
    engine_stamp = stamps["arcadedb-engine"][0]
    base = re.fullmatch(r"(\d+\.\d+\.\d+)-\d{8}\.\d{6}-\d+", engine_stamp)
    if not base:
        raise Unreadable(f"unexpected snapshot version {engine_stamp!r}")
    print(
        json.dumps(
            {
                "engine": f"{base.group(1)}-SNAPSHOT",
                "snapshot": engine_stamp,
                "image_digest": digest,
                "jars": dict(sorted(jars.items())),
            },
            indent=2,
        )
    )
    return 0


def shell_env(tag: str) -> int:
    pin = load_pin()
    problem = tag_problem(tag, pin)
    if problem:
        print(f"::error::{problem}", file=sys.stderr)
        return 1
    print(f"ENGINE={pin['engine']}")
    print(f"SNAPSHOT={pin['snapshot']}")  # the timestamped version Maven resolves
    print(f"IMAGE_REF=arcadedata/arcadedb:{pin['engine']}@{pin['image_digest']}")
    print(f"IMAGE_DIGEST={pin['image_digest']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tag")
    c.add_argument("--pin", type=Path, default=PIN_FILE)
    sub.add_parser("capture").add_argument("digest")
    sub.add_parser("shell-env").add_argument("tag")
    sub.add_parser("verify-jars").add_argument("directory", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "check":
            return check(a.tag, a.pin)
        if a.cmd == "capture":
            return capture(a.digest)
        if a.cmd == "verify-jars":
            return verify_jars(a.directory)
        return shell_env(a.tag)
    except (Unreadable, ValueError, OSError, tarfile.TarError, KeyError) as e:
        print(f"FAIL cannot read a source: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
