#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2021-present Arcade Data Ltd (info@arcadedata.com) and HumemAI (info@humem.ai)
# SPDX-License-Identifier: Apache-2.0
#
# Dump the default class-data-sharing (CDS) archive of the bundled JRE, so the JVM maps the JDK's classes at start instead of
# loading and verifying them (#313). A JRE made by jlink has no archive; the JVM then starts without one.
#
# Usage: dump_cds_archive.sh JRE_DIR
#
# The archive is the JVM's default one (lib/server, or bin/server on Windows): the JVM looks for it by itself at every start and
# ignores it, without an error, when it was dumped for other settings. So it must be dumped with the settings the wheel starts
# with: -XX:+UseCompactObjectHeaders, which src/arcadedb/jvm.py sets (an archive dumped without it is ignored by a JVM that has it,
# and the JVM names the default archive after it). -Xmx1g keeps compressed oops on whatever the RAM of this build host is: the
# archive is the one a heap below 32 GB uses. A start with a larger heap (jvm.py gives the JVM 50 percent of the RAM) takes no
# archive, as before. The archive holds the JDK's classes only: the engine's jars are named by absolute path and checked by
# mtime in an archive, which an installed wheel cannot match.
#
# Run it last, on the finished JRE: the archive records the size of lib/modules. It checks its own result with -Xshare:on, which
# fails the JVM start when the archive cannot be used.
set -euo pipefail

JRE_DIR="${1:?usage: dump_cds_archive.sh JRE_DIR}"
JAVA="${JRE_DIR}/bin/java"
CDS_FLAGS=(-XX:+UseCompactObjectHeaders -Xmx1g)

if [[ ! -x "$JAVA" && ! -x "${JAVA}.exe" ]]; then
    echo "dump_cds_archive.sh: no java in ${JRE_DIR}/bin" >&2
    exit 1
fi

echo "Dumping the default CDS archive of ${JRE_DIR} with: ${CDS_FLAGS[*]}"
"$JAVA" "${CDS_FLAGS[@]}" -Xshare:dump

ARCHIVES="$(find "$JRE_DIR" -name '*.jsa' -type f)"
if [[ -z "$ARCHIVES" ]]; then
    echo "dump_cds_archive.sh: the dump wrote no .jsa file under ${JRE_DIR}" >&2
    exit 1
fi
echo "CDS archive:"
while IFS= read -r archive; do
    echo "   ${archive} ($(du -k "$archive" | cut -f1) KiB)"
done <<< "$ARCHIVES"

# -Xshare:on: no archive, or one for other settings, is an error here instead of a quiet start without it
"$JAVA" "${CDS_FLAGS[@]}" -Xshare:on -version
echo "The JVM starts with the archive (-Xshare:on)"
