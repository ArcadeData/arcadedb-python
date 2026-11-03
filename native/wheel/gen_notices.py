#!/usr/bin/env python3
"""Generate THIRD_PARTY_NOTICES.md for the native `arcadedb` wheel from what is compiled into libarcadedb.so.

  python gen_notices.py --build-log native/target/lib/build.log --engine 26.10.2 \
      --graalvm-home <GraalVM CE home> --out THIRD_PARTY_NOTICES.md
  python gen_notices.py ... --check THIRD_PARTY_NOTICES.md      # exit 1 on drift (build_wheel.py does this)
  python gen_notices.py ... --notice NOTICE [--notice-out NOTICE]   # the source pointers in NOTICE: check, or rewrite

Inputs, all read from disk, nothing typed by hand except the policy tables below:
  1. the `native-image -cp ...` line that native/scripts/build-image.sh logs in build.log (the exact jar set of the image);
  2. each jar: licence names from its Maven pom (parent poms followed), verbatim LICENSE/NOTICE files from the jar itself;
  3. the GraalVM CE home: release file (the GraalVM and JDK source pointers are generated from its SOURCE and JAVA_* fields),
     LICENSE.txt (GPLv2 + Classpath Exception + UPL), legal/java.base notices;
  4. the ArcadeDB NOTICE file: META-INF/NOTICE* of the arcadedb-engine jar on the classpath if it has one (what ships); otherwise the
     NOTICE at the engine release tag (`git show`, needs the tag fetched); otherwise the NOTICE of the checked-out repository.
     The heading in the output does not name the source (the three are the same text for a release); stderr does.
  5. used_classes.txt next to build.log (build-image.sh copies it from the native-image analysis report that -H:+PrintAnalysisCallTree
     writes; GraalVM CE has no `--emit build-report`, that is Oracle GraalVM only): the classes whose methods the image compiles. Each is
     mapped to its JDK module with `jimage list` of the GraalVM home, and a module without notice coverage fails the run (see
     check_module_coverage). The measured module list goes to --modules-out (the CI build record), not into the notices: it can differ
     per platform and the committed THIRD_PARTY_NOTICES.md must be the same on all of them.
Paths: the classpath of a container build names jars as /root/.m2/repository/...; they are read from --m2-repo (default
~/.m2/repository) by their path below `.m2/repository/`, and no absolute path reaches the output.
notice-texts/ holds licence texts that are in no jar: the ANTLR and JTS BSD texts (their poms only link them) and, for libjvector,
the Highway LICENSE and jvector NOTICE.txt at jvector tag 4.0.2.

NO STUDIO: arcadedb-studio is not on the image classpath (its web assets bundle ApexCharts >= 5.1.0, a Community/Commercial licence
that is not compatible with Apache-2.0). This script refuses a classpath that contains it.

Licence policy: the wheel ships under Apache-2.0. ALLOWED are permissive licences; anything else is listed under "Licence review"
and the script exits 2 with --strict. It never decides such a case; the maintainer does.
"""

import argparse
import difflib
import hashlib
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
M2 = Path.home() / ".m2" / "repository"  # replaced by --m2-repo in main()
M2_MARK = ".m2/repository/"
TEXTS = HERE / "notice-texts"

ALLOWED = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "0BSD",
    "Zlib",
    "UPL-1.0",
    "CC0-1.0",
    "BlueOak-1.0.0",
}
# pom <license><name> -> SPDX. Several <license> entries in one pom are alternatives (Maven convention), not a conjunction.
POM_NAMES = [
    (r"apache", "Apache-2.0"),
    (r"^mit$", "MIT"),
    (r"bsd-?2", "BSD-2-Clause"),
    (r"bsd-?3", "BSD-3-Clause"),
    (r"universal permissive", "UPL-1.0"),
    (r"eclipse distribution|^edl", "BSD-3-Clause"),  # EDL-1.0 is the BSD-3-Clause text
    (r"eclipse public license.*2", "EPL-2.0"),
    (r"cc0|public domain", "CC0-1.0"),
]

HIGHWAY_COMMIT = "16f5f5c768064d0bb52952888a14236bec75c104"  # jvector 4.0.2: jvector-native/src/main/native/third_party/highway


def sha(b):
    return hashlib.sha256(b).hexdigest()[:12]


def text(b):
    # trailing whitespace is stripped per line (the repository's pre-commit hook does the same to the committed copy)
    s = b.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(l.rstrip() for l in s.split("\n")).rstrip() + "\n"


def parse_classpath(build_log):
    for line in (
        Path(build_log).read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        m = re.search(r"native-image -cp (\S+) --shared", line)
        if m:
            # ':' on Linux/macOS, ';' with C:\... paths on Windows (the repository part is matched below as ".m2/repository/")
            cp = m.group(1).strip('"')  # an @args file quotes the classpath
            return [j.replace("\\", "/") for j in cp.split(";" if ";" in cp else ":")]
    sys.exit(f"gen_notices: no 'native-image -cp ... --shared' line in {build_log}")


def pom_licenses(pom, depth=0):
    if not pom.exists():
        return None
    root = ET.parse(pom).getroot()
    ns = {"m": root.tag.split("}")[0][1:]}
    names = [
        l.findtext("m:name", namespaces=ns)
        for l in root.findall("m:licenses/m:license", ns)
    ]
    if names:
        return names
    p = root.find("m:parent", ns)
    if p is not None and depth < 8:
        g, a, v = (
            p.findtext(f"m:{x}", namespaces=ns)
            for x in ("groupId", "artifactId", "version")
        )
        return pom_licenses(
            M2 / g.replace(".", "/") / a / v / f"{a}-{v}.pom", depth + 1
        )
    return None


def spdx_from_pom(name):
    for pat, spdx in POM_NAMES:
        if re.search(pat, name.strip().lower()):
            return spdx
    return None


def choose(spdx_options):
    """Alternatives: prefer an ALLOWED one (first in the order the project lists them is not meaningful, so pick by policy order)."""
    for o in spdx_options:
        if o in ALLOWED:
            return o
    return None


def jar_info(jar):
    # The classpath may name the repository as /root/.m2/repository (container build) while it is read from M2 here: map by the part
    # below ".m2/repository/", never by the absolute prefix.
    i = jar.find(M2_MARK)
    if i < 0:
        return None  # the glue jar of this repository (Apache-2.0, native/)
    rel = Path(jar[i + len(M2_MARK) :])
    p = M2 / rel
    if not p.exists():
        sys.exit(
            f"gen_notices: {jar} is on the classpath but {p} does not exist (--m2-repo?)"
        )
    ver = rel.parts[-2]
    art = rel.parts[-3]
    grp = ".".join(rel.parts[:-3])
    names = pom_licenses(p.with_suffix(".pom"))
    opts = [spdx_from_pom(n) or f"UNKNOWN({n})" for n in (names or [])]
    chosen = choose(opts)
    files = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if re.search(
                r"(^|/)META-INF/(LICEN[CS]E|NOTICE|COPYING)[^/]*$", n, re.I
            ) and not n.endswith("/"):
                files[n] = z.read(n)
    return dict(
        gav=f"{grp}:{art}:{ver}",
        grp=grp,
        art=art,
        ver=ver,
        pom_names=names,
        opts=opts,
        chosen=chosen,
        files=files,
    )


def engine_notice(engine, jars):
    """(label, bytes) of the ArcadeDB NOTICE: engine jar first, then the release tag, then this checkout's NOTICE."""
    for j in jars:
        if j["grp"] == "com.arcadedb" and j["art"] == "arcadedb-engine":
            for n, b in sorted(j["files"].items()):
                if re.search(r"(^|/)META-INF/NOTICE[^/]*$", n, re.I):
                    return f"{j['gav']}!/{n}", b
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{engine}:NOTICE"], capture_output=True
    )
    if r.returncode == 0:
        return f"tag {engine}", r.stdout
    # This repository has no ArcadeDB tag and no NOTICE of its own: a copy of the engine's NOTICE (from the tag of the release the
    # image links) is kept next to this script. Refresh it with `git show <engine tag>:NOTICE` when the engine release changes.
    kept = HERE / "ARCADEDB_ENGINE_NOTICE"
    if kept.exists():
        return "copy kept in native/wheel", kept.read_bytes()
    if (REPO / "NOTICE").exists():
        return "repository checkout (tag not fetched)", (REPO / "NOTICE").read_bytes()
    sys.exit(
        f"gen_notices: no NOTICE in the engine jar, at tag {engine} or in the checkout"
    )


# ---- JDK modules linked into the image ----
GPL_CPE_FILES = (
    "ADDITIONAL_LICENSE_INFO",
    "ASSEMBLY_EXCEPTION",
    "LICENSE",
)  # in every legal/<module>; byte-identical to java.base's
# Further notice files in legal/<module>, each with the review decision. "reproduced" = verbatim in the notices below (always, on every
# platform: the committed notices must not depend on which build measured what); "not linked" = covers code or data that the image does
# not contain. A file that is not listed here fails the run (unless it is byte-identical to a reproduced one): someone must read it and
# decide. Evidence for the decisions of 2026-11-03 (the used_classes report of the x86-64, aarch64 and Windows builds of the image, and
# strings of the shipped libarcadedb.so, positive controls included):
#   java.xml: 444 classes of com.sun.org.apache.xerces.internal.*, com.sun.xml.internal.stream, javax.xml.catalog are compiled, so
#     xerces.md is reproduced; org.w3c.dom.{Node,Element,Document,Attr,TypeInfo,...} are types in the image (dom.md); the whole
#     jdk/xml/internal/jdkcatalog resource tree (93 DTD, XSD, ENT, MOD files: XMLSchema, datatypes, xhtml1.0, xhtml1.1, xmlspec, xml.xsd)
#     is registered (listed as /java.xml/jdk/xml/internal/jdkcatalog/... in the library), so the eight W3C data notices are reproduced.
#     No class of com.sun.org.apache.bcel, com.sun.org.apache.xalan, com.sun.org.apache.xpath, com.sun.org.apache.xml.internal or
#     com.sun.java_cup is in the report or named in the library, and no Xalan resource (HTMLEntities.properties) is registered: GUARDS
#     below fails the run if that changes.
#   java.desktop: the report has one class, java.beans.Introspector$1 (JDK code, no third-party notice), and no java.awt, javax.imageio,
#     javax.swing, sun.awt, sun.java2d or sun.font class; the library has no Java_sun_awt/Java_java_awt/Java_sun_java2d/Java_sun_font/
#     Java_javax_imageio JNI name and does not link libawt, liblcms, libfontmanager, libjavajpeg or libsplashscreen (NEEDED: libz, libdl,
#     libpthread, libm, librt, libc). Its nine notices cover the C code of those libraries and their headers. GUARDS fails the run when
#     another java.desktop package is reached.
#   jdk.localedata: cldr.md is byte-identical to java.base's (covered by hash); the report has only the two provider classes
#     (sun.util.resources.cldr.provider.CLDRLocaleDataMetaInfo, sun.util.resources.provider.LocaleDataProvider), not the locale data classes,
#     and sun/text/resources/ext/thai_dict (thaidict.md) is not a registered resource of the library.
#   jdk.incubator.vector: sleef.md covers the SLEEF C library; the image has the Java classes (VectorMathLibrary$SLEEF looks symbols up by
#     name) but links no SLEEF code and ships no libsleef.
#   java.base: the five "not linked" files were reviewed earlier (hotspot, AES-NI, gcc, siphash C code; public-suffix data not in the library).
REVIEWED_LEGAL = {
    "java.base": {
        "icu.md": "reproduced",
        "unicode.md": "reproduced",
        "cldr.md": "reproduced",
        # zlib.md (Windows and macOS homes) and wepoll.md (Windows home) are not in every home: the committed notices must not depend on
        # the platform, so their texts are stored in notice-texts/jdk/ (checked equal to the home's file where there is one). The Linux
        # image links the system libz.so.1 and the Windows and macOS images link libzip with its bundled zlib; wepoll is the epoll
        # emulation of libnio on Windows. Both are reproduced without proof of absence.
        "zlib.md": "reproduced",
        "wepoll.md": "reproduced",
        "aes.md": "not linked",
        "c-libutl.md": "not linked",
        "gcc.md": "not linked",
        "siphash.md": "not linked",
        "public_suffix.md": "not linked",
    },
    "java.xml": {
        "xerces.md": "reproduced",
        "dom.md": "reproduced",
        "schema10part1.md": "reproduced",
        "schema10part2.md": "reproduced",
        "xhtml10.md": "reproduced",
        "xhtml10schema.md": "reproduced",
        "xhtml11.md": "reproduced",
        "xhtml11schema.md": "reproduced",
        "xmlspec.md": "reproduced",
        "xmlxsd.md": "reproduced",
        "bcel.md": "not linked",
        "xalan.md": "not linked",
        "jcup.md": "not linked",
    },
    "java.desktop": {
        f: "not linked"
        for f in (
            "colorimaging.md",
            "freetype.md",  # Windows and macOS homes; Linux uses the system FreeType
            "giflib.md",
            "harfbuzz.md",
            "jpeg.md",
            "lcms.md",
            "libpng.md",
            "mesa3d.md",
            "pipewire.md",
            "xwd.md",
        )
    },
    "jdk.localedata": {"thaidict.md": "not linked"},
    "jdk.incubator.vector": {"sleef.md": "not linked"},
}
# Windows has no symlinks: its GraalVM home writes `Please see ..\java.base\<same file name>` (CRLF, nothing else; read from the
# graalvm-community-jdk-25i4-25.0.4.1.1_windows-x64_bin.zip release asset, e.g. legal/jdk.net/LICENSE) where the Linux and macOS homes
# have a copy of java.base's file. Only exactly that text, naming the same file, is taken as java.base's file.
WINDOWS_POINTER = re.compile(
    rb"Please see \.\.[\\/]java\.base[\\/](?P<n>[^\\/\r\n]+)\r?\n?"
)
REPRODUCED = sorted(
    (m, f) for m, d in REVIEWED_LEGAL.items() for f, v in d.items() if v == "reproduced"
)
# A "not linked" decision that rests on which classes are in the image: ("never", prefixes) = no used class of the module may be in
# these packages; ("only", prefixes) = every used class of the module must be in these packages. Otherwise the run fails.
GUARDS = {
    ("java.xml", "bcel.md"): ("never", ("com.sun.org.apache.bcel",)),
    ("java.xml", "xalan.md"): (
        "never",
        (
            "com.sun.org.apache.xalan",
            "com.sun.org.apache.xpath",
            "com.sun.org.apache.xml.internal",
        ),
    ),
    ("java.xml", "jcup.md"): ("never", ("com.sun.java_cup",)),
    **{
        ("java.desktop", f): ("only", ("java.beans",))
        for f in REVIEWED_LEGAL["java.desktop"]
    },
    ("jdk.localedata", "thaidict.md"): (
        "only",
        ("sun.util.resources.cldr.provider", "sun.util.resources.provider"),
    ),
}
# Modules without a legal/<module> directory whose code is GraalVM's own, under the GraalVM licence the wheel ships (LICENSE.txt of the
# home: GPLv2 + CPE, UPL-1.0). org.graalvm.* by prefix; jdk.graal.compiler.options by name: it is compiler/src/jdk.graal.compiler.options
# of oracle/graal at the commit the release file names, and its sources carry the Oracle GPLv2 + Classpath Exception header.
GRAALVM_OWN_PREFIXES = ("org.graalvm.",)
GRAALVM_OWN_MODULES = ("jdk.graal.compiler.options",)


def legal_bytes(legal, m, f):
    """Content of legal/<m>/<f>; a Windows pointer file to the same-named java.base file reads as that file."""
    b = (legal / m / f).read_bytes()
    if m != "java.base":
        ptr = WINDOWS_POINTER.fullmatch(b)
        target = legal / "java.base" / f
        if ptr and ptr.group("n").decode("utf-8", "replace") == f and target.is_file():
            return target.read_bytes()
    return b


def reproduced_text(G, m, f):
    """Text of a reproduced notice: the GraalVM home's file, else the stored copy (platform-specific files), which must equal the home's
    file where the home has it."""
    home = G / "legal" / m / f
    stored = TEXTS / "jdk" / f"{m}.{f}"
    if home.is_file():
        t = text(home.read_bytes())
        if stored.exists() and text(stored.read_bytes()) != t:
            sys.exit(
                f"gen_notices: legal/{m}/{f} of the GraalVM home differs from {stored}; read it and update the stored copy"
            )
        return t
    if stored.exists():
        return text(stored.read_bytes())
    sys.exit(
        f"gen_notices: legal/{m}/{f} is not in the GraalVM home and there is no stored copy {stored}"
    )


def jimage_packages(G):
    """package name -> set of JDK modules, from `jimage list` of the GraalVM home (a JDK tool, on every platform)."""
    exe = G / "bin" / ("jimage.exe" if os.name == "nt" else "jimage")
    r = subprocess.run(
        [str(exe), "list", str(G / "lib" / "modules")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if r.returncode:
        sys.exit(
            f"gen_notices: `{exe.name} list` failed ({r.returncode}): {r.stderr[:300]}"
        )
    pkgs, mod = {}, None
    for line in r.stdout.splitlines():
        if line.startswith("Module:"):
            mod = line.split(":", 1)[1].strip()
            continue
        e = line.strip()
        if mod and e.endswith(".class") and "/" in e:
            pkgs.setdefault(e.rsplit("/", 1)[0].replace("/", "."), set()).add(mod)
    if "java.lang" not in pkgs:
        sys.exit(
            "gen_notices: `jimage list` output has no java.lang package (format change?)"
        )
    return pkgs


def used_modules(used_classes, G):
    """{JDK module: packages} for the classes of the native-image used_classes_*.txt report: one `<loader>:<java name>` per line, as CE 25.4
    writes it (e.g. `null:java.lang.String`, `com.oracle.svm.hosted.NativeImageClassLoader:com.arcadedb.Foo`); loader names hold no colon,
    so the class is the part after the last one (a bare `<java name>` line also parses).
    """
    pkgs = jimage_packages(G)
    names = set()
    for line in (
        Path(used_classes).read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        n = line.strip().rsplit(":", 1)[-1]
        n = re.sub(r"(\[\])+$", "", n).split("/", 1)[
            0
        ]  # arrays; hidden classes (Foo$$Lambda/0x...)
        if n:
            names.add(n)
    mods = {}  # module -> packages with a used class
    for n in names:
        if "." in n:
            pkg = n.rsplit(".", 1)[0]
            for m in pkgs.get(pkg, ()):
                mods.setdefault(m, set()).add(pkg)
    # A report that did not parse would map to no module and pass silently: refuse an implausible one.
    if len(names) < 1000 or "java.base" not in mods:
        sys.exit(
            f"gen_notices: {used_classes} lists {len(names)} classes and {len(mods)} JDK modules: not a used_classes report of this image"
        )
    return mods


def check_module_coverage(G, modules):
    """Problems (strings) for modules whose notices are not covered. `modules` is {module: used packages}. Covered = legal/<module> exists
    in the GraalVM home, its GPLv2 + Classpath Exception files equal java.base's (reproduced once, via LICENSE.txt and legal/java.base),
    and every further file has a decision in REVIEWED_LEGAL (or is byte-identical to a reproduced file), with its GUARDS holding.
    """
    legal = G / "legal"
    base = {f: sha((legal / "java.base" / f).read_bytes()) for f in GPL_CPE_FILES}
    reproduced = {sha(reproduced_text(G, m, f).encode()) for m, f in REPRODUCED}
    problems = []
    for m, pkgs in sorted(modules.items()):
        d = legal / m
        if not d.is_dir():
            if not (m in GRAALVM_OWN_MODULES or m.startswith(GRAALVM_OWN_PREFIXES)):
                problems.append(
                    f"{m}: no legal/{m} in the GraalVM home and not a GraalVM-own module"
                )
            continue
        have = {p.name for p in d.iterdir() if p.is_file()}
        for f in GPL_CPE_FILES:
            if f not in have:
                problems.append(f"{m}: legal/{m}/{f} is missing")
            elif sha(legal_bytes(legal, m, f)) != base[f]:
                problems.append(
                    f"{m}: legal/{m}/{f} differs from legal/java.base/{f}; read it and extend the policy"
                )
        for f in sorted(have - set(GPL_CPE_FILES)):
            decision = REVIEWED_LEGAL.get(m, {}).get(f)
            if decision is None:
                if sha(text(legal_bytes(legal, m, f)).encode()) in reproduced:
                    continue  # the same text is reproduced for another module
                problems.append(
                    f"{m}: legal/{m}/{f} has no review decision; check whether the code or data it covers is in the image, "
                    "then add it to REVIEWED_LEGAL (reproduced or not linked)"
                )
            elif decision == "not linked" and (m, f) in GUARDS:
                kind, prefixes = GUARDS[(m, f)]
                hit = sorted(
                    p
                    for p in pkgs
                    if any(p == x or p.startswith(x + ".") for x in prefixes)
                )
                if kind == "never" and hit:
                    problems.append(
                        f"{m}: legal/{m}/{f} is marked not linked but the image compiles classes of {', '.join(hit)}"
                    )
                if kind == "only" and set(pkgs) - set(hit):
                    problems.append(
                        f"{m}: legal/{m}/{f} is marked not linked but the image compiles classes of "
                        f"{', '.join(sorted(set(pkgs) - set(hit)))} (decided for {', '.join(prefixes)} only)"
                    )
    return problems


def source_pointers(rel):
    """Where the GPLv2 + CPE sources of this GraalVM build are, from the `release` file of the home: SOURCE (`<suite>:<rev>`, the JDK
    is the `.` entry `git:<rev>`), JAVA_RUNTIME_VERSION, JAVA_VERSION, GRAALVM_VERSION.
    """
    for k in ("SOURCE", "JAVA_RUNTIME_VERSION", "JAVA_VERSION", "GRAALVM_VERSION"):
        if not rel.get(k):
            sys.exit(f"gen_notices: the release file of the GraalVM home has no {k}")
    src = dict(t.split(":", 1) for t in rel["SOURCE"].split() if ":" in t)
    m = re.fullmatch(r"git:([0-9a-f]{7,40})\+?", src.get(".", ""))
    if not m:
        sys.exit(
            f"gen_notices: no `.:git:<rev>` (the JDK) in SOURCE of the release file: {rel['SOURCE'][:200]}"
        )
    jdk_rev = m.group(1)
    revs = {
        src[k] for k in ("substratevm", "compiler", "sdk", "truffle", "vm") if k in src
    }
    if not revs or not all(re.fullmatch(r"[0-9a-f]{40}", r) for r in revs):
        sys.exit(
            f"gen_notices: no graal commit in SOURCE of the release file: {rel['SOURCE'][:200]}"
        )
    graal = next(iter(revs)) if len(revs) == 1 else ", ".join(sorted(revs))
    v = re.match(r"(\d+)\.(\d+)\.(\d+)", rel["JAVA_VERSION"])
    if not v:
        sys.exit(f"gen_notices: cannot read JAVA_VERSION {rel['JAVA_VERSION']}")
    up_repo = f"https://github.com/openjdk/jdk{v.group(1)}u"
    up_tag = f"jdk-{v.group(1)}.{v.group(2)}.{v.group(3)}-ga"
    gv = rel["GRAALVM_VERSION"]
    return dict(
        graal_url="https://github.com/oracle/graal",
        graal_text=f"tag vm-{gv}; the release file of the GraalVM CE home names commit {graal} of substratevm",
        jdk_url="https://github.com/graalvm/labs-openjdk",
        jdk_text=(
            f"the labsjdk that GraalVM CE {gv} is built on, JAVA_RUNTIME_VERSION {rel['JAVA_RUNTIME_VERSION']}; "
            f"the release file names commit {jdk_rev}; the OpenJDK update it is based on: {up_repo} tag {up_tag}"
        ),
        graal_rev=graal,
        jdk_rev=jdk_rev,
        up_repo=up_repo,
        up_tag=up_tag,
    )


NOTICE_BLOCK = re.compile(r"^    GraalVM:.*?(?=^  Other GraalVM modules)", re.M | re.S)


def render_notice(have, ptr):
    """NOTICE (hand-written) with its two source-pointer lines replaced by the generated ones; every other line is kept."""
    block = (
        f"    GraalVM:  {ptr['graal_url']}  ({ptr['graal_text']})\n"
        f"    OpenJDK:  {ptr['jdk_url']}  ({ptr['jdk_text']})\n"
    )
    if len(NOTICE_BLOCK.findall(have)) != 1:
        sys.exit(
            "gen_notices: NOTICE has no single `    GraalVM:` ... `  Other GraalVM modules` block to regenerate"
        )
    return NOTICE_BLOCK.sub(lambda _m: block, have)


def fence(body):
    n = max([3] + [len(m) + 1 for m in re.findall(r"`{3,}", body)])
    return "`" * n + "\n" + body + "`" * n + "\n"


def main():
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    global M2
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--build-log",
        help="build.log of build-image.sh (holds the native-image classpath)",
    )
    g.add_argument("--lib-dir", help="shortcut: <lib-dir>/build.log")
    ap.add_argument("--engine", required=True, help="engine release tag, e.g. 26.10.2")
    ap.add_argument("--graalvm-home", required=True)
    ap.add_argument(
        "--m2-repo",
        default=str(M2),
        help="Maven repository the classpath jars are read from (a container build logs /root/.m2/repository)",
    )
    ap.add_argument("--out")
    ap.add_argument(
        "--used-classes",
        help="used_classes report of the image build (default: used_classes.txt next to build.log; build-image.sh writes it)",
    )
    ap.add_argument(
        "--modules-out",
        help="write the JDK modules linked into the image here (one per line; for the build record)",
    )
    ap.add_argument(
        "--notice",
        help="the NOTICE file of the wheel: exit 1 when its GraalVM and OpenJDK source-pointer lines differ from the generated ones",
    )
    ap.add_argument(
        "--notice-out",
        help="with --notice: write the NOTICE with the regenerated source-pointer lines here",
    )
    ap.add_argument("--check", help="compare with this file; exit 1 when it differs")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 when the licence review list is not empty",
    )
    a = ap.parse_args()
    M2 = Path(a.m2_repo)
    log = a.build_log or str(Path(a.lib_dir) / "build.log")
    cp = parse_classpath(log)
    if any("arcadedb-studio" in c for c in cp):
        sys.exit(
            "gen_notices: arcadedb-studio is on the classpath; its ApexCharts assets are not Apache-2.0 compatible (see module docstring)"
        )
    G = Path(a.graalvm_home)
    rel = dict(
        re.findall(
            r'^(\w+)="?(.*?)"?$', (G / "release").read_text(encoding="utf-8"), re.M
        )
    )

    ptr = source_pointers(rel)
    uc = (
        Path(a.used_classes)
        if a.used_classes
        else Path(log).with_name("used_classes.txt")
    )
    if not uc.exists():
        sys.exit(
            f"gen_notices: {uc} does not exist: build-image.sh writes it from the native-image analysis report (-H:+PrintAnalysisCallTree); "
            "rebuild the image with the current script (or pass --used-classes)"
        )
    used = used_modules(uc, G)
    modules = sorted(used)
    problems = check_module_coverage(G, used)
    if a.modules_out:
        Path(a.modules_out).write_text(
            "\n".join(modules) + "\n", encoding="utf-8", newline="\n"
        )
    print(
        f"gen_notices: {len(modules)} JDK modules in the image: {' '.join(modules)}",
        file=sys.stderr,
    )
    if problems:
        for pr in problems:
            print(f"gen_notices: NO NOTICE COVERAGE: {pr}", file=sys.stderr)
        sys.exit(1)

    jars = [j for j in (jar_info(c) for c in cp) if j]
    review = []  # (what, why)
    out = []
    w = out.append

    w("# Third-party notices (native image, preview)\n")
    w(
        f"Generated by `native-wheel/gen_notices.py` from the classpath that built `libarcadedb.so` (engine {a.engine}, "
        f"GraalVM CE {rel.get('GRAALVM_VERSION')}). Do not edit by hand: `build_wheel.py` fails when this file differs from the "
        "generator output. Licence names come from each artifact's own Maven pom; notice texts are copied "
        "verbatim from the jars, the engine tag and the GraalVM home. A text of the Apache License 2.0 is in "
        "`licenses/Apache-2.0.txt`.\n"
    )

    # ---- Java ----
    w("## Java libraries compiled into libarcadedb.so\n")
    w(
        f"{len(jars)} Maven artifacts on the native-image classpath (plus the glue jar of this repository, Apache-2.0).\n"
    )
    w("| Artifact | Declared in pom | Applied | Verbatim texts |")
    w("|---|---|---|---|")
    appendix_java = {}  # sha -> (name, text, [gavs])
    for j in jars:
        decl = " / ".join(j["pom_names"] or ["none"])
        if j["chosen"] is None:
            review.append((j["gav"], f"licence not in the allowed set: {decl}"))
        refs = []
        for n, b in sorted(j["files"].items()):
            k = sha(b)
            appendix_java.setdefault(k, [n.split("/")[-1], text(b), []])[2].append(
                j["gav"]
            )
            refs.append(f"`{n.split('/')[-1]}` [{k}]")
        extra = {
            "org.antlr:antlr4-runtime": "antlr4-runtime-4.13.2.LICENSE.txt",
            "org.locationtech.jts:jts-core": "jts-core-1.20.0.LICENSE_EDLv1.txt",
        }.get(f"{j['grp']}:{j['art']}")
        if extra:
            b = (TEXTS / "java" / extra).read_bytes()
            k = sha(b)
            appendix_java.setdefault(k, [extra, text(b), []])[2].append(j["gav"])
            refs.append(f"`{extra}` [{k}]")
        w(
            f"| {j['gav']} | {decl} | {j['chosen'] or 'REVIEW'} | {', '.join(refs) or 'none in jar'} |"
        )
    w("")
    w(
        "Where a pom lists several licences they are alternatives (JTS: EPL-2.0 or EDL-1.0; HdrHistogram: CC0-1.0 or BSD-2-Clause); "
        "the permissive one is applied. EDL-1.0 is the BSD-3-Clause text.\n"
    )

    # ---- native code ----
    w("## Native code shipped next to the library\n")
    w(
        "`libjvector-x86_64.so` (Linux x86-64) and `libjvector-aarch64.so` (Linux aarch64) are taken unmodified from `io.github.jbellis:jvector:4.0.2` (Apache-2.0, pom). Their sources are in the "
        "jvector repository at tag `4.0.2` (`jvector-native/src/main/native`): jvector's own C++ kernels (Apache-2.0) and the Google Highway "
        f"library, pinned there as a git submodule at commit `{HIGHWAY_COMMIT}`. Highway's `LICENSE` at that commit says: "
        "dual-licensed under the Apache License 2.0 or the BSD 3-Clause License, except `hwy/contrib/random/random-inl.h` (CC0 1.0). "
        "Applied here: Apache-2.0. The `LICENSE` of Highway and the `NOTICE.txt` of jvector are reproduced verbatim below. "
        "The libraries link `libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1` and `libc.so.6` dynamically (not shipped).\n"
    )
    for f, who in (
        (
            f"highway-{HIGHWAY_COMMIT}.LICENSE.txt",
            "libjvector-x86_64.so and libjvector-aarch64.so (google/highway)",
        ),
        ("jvector-4.0.2.NOTICE.txt", "io.github.jbellis:jvector:4.0.2"),
    ):
        b_ = (TEXTS / "java" / f).read_bytes()
        appendix_java.setdefault(sha(b_), [f, text(b_), []])[2].append(who)

    # ---- GraalVM / JDK ----
    w("## GraalVM Community Edition runtime and JDK classes in libarcadedb.so\n")
    w(
        f"`libarcadedb.so` contains the Substrate VM runtime and the parts of the OpenJDK class library the engine reaches, built with "
        f"GraalVM CE {rel.get('GRAALVM_VERSION')} (Java {rel.get('JAVA_VERSION')}, implementor {rel.get('IMPLEMENTOR')}). Licence: GNU General Public License "
        "version 2 with the Classpath Exception (verbatim below, from `LICENSE.txt` of the GraalVM CE home; also in `licenses/`). The exception "
        "permits linking and distributing these parts with code under other licences. `graal-sdk`, `polyglot`, `collections` (25.0.2, Maven Central) "
        f"and `word` ({rel.get('GRAALVM_VERSION')}) are UPL-1.0. Oracle GraalVM (GFTC) is not used. Sources: {ptr['graal_url']} ({ptr['graal_text']}); "
        f"{ptr['jdk_url']} ({ptr['jdk_text']}).\n"
    )
    w(
        "Third-party code and data inside the JDK classes that the image contains (texts from `legal/<module>` of the GraalVM home, verbatim below): "
        "ICU normalisation data (`jdk/internal/icu/impl/data/icudt76b/nfc.nrm`, registered resource), Unicode data files, and CLDR locale data "
        "(`java.base`), zlib (the bundled copy that the Windows and macOS builds of libzip link; the Linux image links the system `libz.so.1`) and wepoll (Windows `libnio`); the Apache Xerces classes of `java.xml` (`com.sun.org.apache.xerces.internal.*`, compiled into the image), the W3C DOM "
        "interfaces (`org.w3c.dom`), and the W3C XML Schema, XHTML 1.0, XHTML 1.1, XML specification and `xml.xsd` DTD and schema files "
        "(`jdk/xml/internal/jdkcatalog/`, registered resources). "
        "Notices that cover code or data the image does not contain are not reproduced: in `java.base` aes.md, c-libutl.md, gcc.md and "
        "siphash.md (HotSpot and C code) and public_suffix.md (checked: the library contains no public-suffix data); in `java.xml` bcel.md, "
        "xalan.md and jcup.md (no class of `com.sun.org.apache.bcel`, `com.sun.org.apache.xalan`, `com.sun.org.apache.xpath`, "
        "`com.sun.org.apache.xml.internal` or `com.sun.java_cup` is compiled, and the Xalan resources are not registered); in `java.desktop` "
        "the notices of the C libraries behind AWT, Java 2D and ImageIO (giflib, libpng, jpeg, lcms, harfbuzz, mesa3d, pipewire, xwd, "
        "colorimaging: the image reaches only `java.beans.Introspector`, has no AWT or imaging JNI entry point and links none of those "
        "libraries); in `jdk.localedata` thaidict.md (only the two locale provider classes are compiled, the Thai dictionary is not a "
        "registered resource; its cldr.md is identical to the `java.base` one); in `jdk.incubator.vector` sleef.md (the SLEEF C library is "
        "not linked; the module is on the module path but its vector intrinsics are disabled in the build, `-H:-VectorAPISupport`). "
        "`gen_notices.py` fails the build when the image reaches code that one of the `java.xml`, `java.desktop` or `jdk.localedata` decisions above rests on.\n"
    )
    w(
        "The JDK modules in the image are not typed here but measured on every build: `build-image.sh` builds with `-H:+PrintAnalysisCallTree` "
        "(GraalVM CE has no `--emit build-report`), which makes native-image write `used_classes_*.txt`, the classes whose methods the "
        "image compiles; `gen_notices.py` maps each class to its JDK module with `jimage list` of the GraalVM home. The run fails when a "
        "module has no `legal/<module>` directory in the home (GraalVM's own `org.graalvm.*` modules are covered by `LICENSE.txt`), when its "
        "GPLv2 + Classpath Exception files differ from the `java.base` ones reproduced below, or when its `legal` directory has a further "
        "notice file that no one has reviewed (or that is not-linked by decision while the image reaches its code). The measured list is kept in the build record of each platform (`jdk-modules.txt`).\n"
    )
    w(
        "System libraries linked dynamically and not shipped: `libz.so.1`, `libc.so.6`, `ld-linux-x86-64.so.2` (checked with `ldd`).\n"
    )

    # ---- review ----
    w("## Licence review\n")
    w(
        "Dependencies whose licence is not in the allowed permissive set or whose provenance is not recorded in the build inputs. "
        "The generator lists them; a maintainer decides.\n"
    )
    for what, why in review:
        w(f"- **{what}**: {why}")
    if not review:
        w("None open.")
    w("")

    # ---- appendix ----
    w("## Verbatim texts\n")
    src, nb = engine_notice(a.engine, jars)
    print(f"gen_notices: ArcadeDB NOTICE read from {src}", file=sys.stderr)
    w("### ArcadeDB NOTICE (engine release %s)\n" % a.engine)
    w(fence(text(nb)))
    w(
        "### GraalVM Community Edition LICENSE.txt (GPLv2, Classpath Exception, UPL-1.0, BSD-3-Clause)\n"
    )
    w(fence(text((G / "LICENSE.txt").read_bytes())))
    for m, f in (
        ("java.base", "ADDITIONAL_LICENSE_INFO"),
        ("java.base", "ASSEMBLY_EXCEPTION"),
        *REPRODUCED,
    ):
        w(f"### OpenJDK {m} legal notice: {f}\n")
        w(fence(reproduced_text(G, m, f)))
    w(
        "### Texts found inside the jars (de-duplicated by content; bracket = sha256 prefix)\n"
    )
    for k, (n, t, who) in sorted(
        appendix_java.items(), key=lambda kv: (kv[1][0].lower(), kv[1][2])
    ):
        w(f"#### {n} [{k}]\n")
        w("Applies to: " + ", ".join(sorted(who)) + "\n")
        w(fence(t))
    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body).rstrip() + "\n"
    if a.out:
        Path(a.out).write_text(body, encoding="utf-8", newline="\n")
    status = 0
    if a.check:
        have = Path(a.check).read_text(encoding="utf-8")
        if have != body:
            print(
                f"gen_notices: {a.check} differs from the generated notices; run gen_notices.py --out {a.check}",
                file=sys.stderr,
            )
            diff = list(
                difflib.unified_diff(
                    have.splitlines(),
                    body.splitlines(),
                    "committed",
                    "generated",
                    lineterm="",
                )
            )
            for l in diff[:80]:
                print(l[:300], file=sys.stderr)
            if len(diff) > 80:
                print(f"... {len(diff) - 80} more diff lines", file=sys.stderr)
            status = 1
    if a.notice:
        nhave = Path(a.notice).read_text(encoding="utf-8")
        nnew = render_notice(nhave, ptr)
        if a.notice_out:
            Path(a.notice_out).write_text(nnew, encoding="utf-8", newline="\n")
        if nhave != nnew:
            print(
                f"gen_notices: the GraalVM/OpenJDK source lines of {a.notice} differ from the release file of the GraalVM home; "
                f"run gen_notices.py ... --notice {a.notice} --notice-out {a.notice}",
                file=sys.stderr,
            )
            for l in difflib.unified_diff(
                nhave.splitlines(),
                nnew.splitlines(),
                "committed",
                "generated",
                lineterm="",
            ):
                print(l[:400], file=sys.stderr)
            status = 1
    for what, why in review:
        print(f"licence review: {what}: {why}", file=sys.stderr)
    if a.strict and review and status == 0:
        status = 2
    return status


if __name__ == "__main__":
    sys.exit(main())
