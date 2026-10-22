#!/usr/bin/env bash
# Stage the ArcadeDB jars a native build bundles. Sourced by build-native.sh.
#
# stage_jars JARS_DIR SOURCE_DIR ARCADEDB_TAG
#
# JARS_DIR ends up holding exactly the jars asked for, on every build, its
# earlier jars (a previous build's, its arcadedb-python-bridge.jar) removed first:
#   - SOURCE_DIR when given: build.sh passes local-jars/lib, where it copied its
#     jars directory (in CI, the downloaded artifact). The jars come as given, so
#     a pinned ARCADEDB_TAG that names another engine version only warns.
#   - otherwise arcadedata/arcadedb:ARCADEDB_TAG, pulled again: the tags follow
#     upstream snapshots, so a cached image or a previous build's jars would ship
#     an old engine under a new build (#9).
# SOURCE_DIR being JARS_DIR itself (build-native.sh run by hand on its own jars)
# leaves them as they are: there is nothing to copy.

stage_jars() {
    local jars_dir="$1"
    local source_dir="$2"
    local tag="$3"

    mkdir -p "$jars_dir"

    if [[ -n "$source_dir" ]] && [[ "$source_dir" -ef "$jars_dir" ]]; then
        echo "📦 Using the jars already in: $jars_dir"
        return 0
    fi

    if [[ -n "$source_dir" ]]; then
        if ! compgen -G "$source_dir/*.jar" > /dev/null; then
            echo "❌ No JAR files in: $source_dir" >&2
            return 1
        fi
        rm -f "$jars_dir"/*.jar
        cp "$source_dir"/*.jar "$jars_dir"/
        echo "📦 Jars staged from: $source_dir"
        local engine
        engine=$(python3 "$(dirname "${BASH_SOURCE[0]}")/extract_version.py" "$jars_dir" --format=docker 2> /dev/null || true)
        if [[ "$tag" != "latest" ]] && [[ -n "$engine" ]] && [[ "$engine" != "$tag" ]]; then
            echo "⚠️  ARCADEDB_TAG is $tag, but the jars given carry engine $engine: building $engine" >&2
        fi
        return 0
    fi

    if ! command -v docker > /dev/null 2>&1; then
        echo "❌ Docker not found: it pulls the jars from arcadedata/arcadedb:$tag" >&2
        echo "💡 Or pass a jars directory: ./scripts/build.sh <platform> <python> <jars-dir>" >&2
        return 1
    fi

    local temp_jars
    temp_jars=$(mktemp -d)
    echo "📥 Pulling the jars from arcadedata/arcadedb:$tag..."
    if ! docker run --rm --pull always "arcadedata/arcadedb:$tag" tar -cf - -C /home/arcadedb lib | tar -xf - -C "$temp_jars"; then
        echo "❌ Failed to get the jars from arcadedata/arcadedb:$tag" >&2
        rm -rf "$temp_jars"
        return 1
    fi
    rm -f "$jars_dir"/*.jar
    cp "$temp_jars/lib"/*.jar "$jars_dir"/
    rm -rf "$temp_jars"
    echo "✅ Jars from arcadedata/arcadedb:$tag in: $jars_dir"
}
