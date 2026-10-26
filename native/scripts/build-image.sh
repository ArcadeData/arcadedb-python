#!/usr/bin/env bash
# Build the native image (the shared library the `arcadedb` wheel ships): GraalVM CE 25.4.4.1.1, -O2, FFM unbound Highway
# handle (fastops.mode 1), -H:-VectorAPISupport (CE crash oracle/graal#14751), engine jars of the pinned RELEASE (arcadedb.version, from
# Maven Central, never a local SNAPSHOT), local symbols deleted (the default: a smaller image than the phase 7/8 profiling builds).
# MARCH (default compatibility = baseline x86-64, what a manylinux wheel promises, PEP 600; x86-64-v3 is the GraalVM CE 25.4 default and the faster reference).
# usage: [MARCH=x86-64-v3] [XMX=12g] [NI_EXTRA="-H:NumberOfThreads=2"] build-image.sh [OUTDIR]     A laptop with other jobs runs it through its job gate; the CI calls it directly.
# Platforms: Linux (libarcadedb.so; on x86-64 also libjvector-x86_64.so, jvector ships the Highway library for x86-64 only here), macOS
# (libarcadedb.dylib) and Windows in Git Bash with the MSVC environment loaded (libarcadedb.dll). The CI (test-native.yml) caps the
# builder heap and threads on the 7 GB macOS runner (XMX=5g, NI_EXTRA=-H:NumberOfThreads=2).
#   e.g. NEED_GB=13 JOB_MEM_GB=14 quiet_build.sh bash native/scripts/build-image.sh
set -eu
WT=$(cd "$(dirname "$0")/../.." && pwd)
G=${GRAALVM_HOME_P5:-$HOME/build-scratch/graalvm-dl-25.4.4.1.1/graalvm-community-25.4.4.1.1+1.1}
OUT=${1:-$WT/native/target/lib}
mkdir -p "$OUT"
ENGINE=${ARCADEDB_ENGINE_VERSION:-26.10.2}
MARCH=${MARCH:-compatibility}
XMX=${XMX:-12g}
case "$(uname -s)" in
    Linux) EXT=so ;;
    Darwin) EXT=dylib ;;
    MINGW* | MSYS* | CYGWIN*)
        EXT=dll
        G=$(cygpath -u "$G")
        ;;
    *)
        echo "unsupported platform $(uname -s)" >&2
        exit 1
        ;;
esac
LIBFILE=libarcadedb.$EXT # not LIB: the MSVC environment on Windows exports LIB (library search path)
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
export JAVA_HOME=$G PATH=$G/bin:$PATH
export NATIVE_IMAGE_OPTIONS="-J-Dfastops.mode=1 -J-Xmx$XMX -H:-VectorAPISupport -march=$MARCH${NI_EXTRA:+ $NI_EXTRA}"
cd "$WT/native"
rm -f "target/$LIBFILE" "$OUT/$LIBFILE"
./mvnw -B --no-transfer-progress -f pom.xml -DskipTests -Darcadedb.version="$ENGINE" -Dgraalvm.version=25.4.4.1.1 -Dnative.opt=-O2 package > "$OUT/build.log" 2>&1
# Windows: the Maven plugin hands native-image its arguments in an @args file that it deletes at exit, so build.log has no
# "native-image -cp <classpath> --shared" line, which gen_notices.py reads. Append one: the glue jar plus the runtime classpath that
# dependency:build-classpath resolves with the same properties (checked on Linux: the same 59 jars as the logged classpath, plus the glue jar).
if [ "$EXT" = dll ]; then
    ./mvnw -B -q --no-transfer-progress -f pom.xml -Darcadedb.version="$ENGINE" -Dgraalvm.version=25.4.4.1.1 -DincludeScope=runtime \
        dependency:build-classpath -Dmdep.outputFile=target/runtime.cp >> "$OUT/build.log" 2>&1
    echo "native-image -cp target/arcadedb-native-lib-0.0.1.jar;$(tr -d '\r\n' < target/runtime.cp) --shared" >> "$OUT/build.log"
fi
# native-image.properties in the glue jar also carries JavaArgs = -Dsun.misc.unsafe.memory.access=allow for the builder's JVM (JEP 498: the image
# would print the sun.misc.Unsafe warning that jvector causes on every vector index reopen, into a database file on Windows). The build log does not
# show a JavaArgs line; tests/test_log_redirect.py is the gate that catches a lost one.
# the jvector force-inline filters (native-image.properties in the glue jar) must be picked up: the build output lists the option and its origin (Windows prints the path with backslashes)
grep -a -E -q 'PriorityForceInline. \(origin\(s\): .META-INF.native-image.org\.humemai.arcadedb-native-lib.native-image\.properties' "$OUT/build.log" ||
    {
        echo "-H:PriorityForceInline from native-image.properties is not in the build output: the jvector force-inline hints were not applied" >&2
        exit 1
    }
INLINE_LOG="PriorityForceInline picked up from native-image.properties ($(grep -a -c 'PriorityForceInline' "$OUT/build.log") log lines)"
cp "target/$LIBFILE" "$OUT/"
LIBS=("$OUT/$LIBFILE")
# Highway (jvector's native provider), found next to libarcadedb.so by the image (ARCADEDB_NATIVE_LIBDIR); Linux x86-64 only: the other
# platforms ship no jvector library and the image falls back to the Java providers
if [ "$EXT" = so ] && [ "$(uname -m)" = x86_64 ]; then
    J=$(unzip -l "$HOME/.m2/repository/io/github/jbellis/jvector/4.0.2/jvector-4.0.2.jar" | awk '/libjvector-x86_64.so/{print $4}' | head -1)
    if [ -n "$J" ]; then unzip -p "$HOME/.m2/repository/io/github/jbellis/jvector/4.0.2/jvector-4.0.2.jar" "$J" > "$OUT/libjvector-x86_64.so"; else
        echo "libjvector-x86_64.so is not in the jvector 4.0.2 jar" >&2
        exit 1
    fi
    LIBS+=("$OUT/libjvector-x86_64.so")
fi
sha256 "${LIBS[@]}" > "$OUT/sha256.txt"
echo "native image done: engine $ENGINE; march $MARCH; $INLINE_LOG; $(grep -a 'Finished generating' "$OUT/build.log")" | tee "$OUT/build.done"
