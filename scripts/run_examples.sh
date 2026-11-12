#!/usr/bin/env bash
# The example runner, shared by the two example workflows:
#   .github/workflows/test-python-examples.yml   the JVM package (arcadedb-jvm)
#   .github/workflows/test-native.yml            the native package (arcadedb), job "Examples on the installed wheel"
# The per-example arguments, timeouts, JVM arguments, dataset handling and the answer check live here, once, so the two jobs cannot drift apart.
#
# usage (from anywhere; the working directory is changed to examples/):
#   scripts/run_examples.sh datasets   download the datasets the selected examples need (exit 75 of download_data.py = host down: skip, do not fail)
#   scripts/run_examples.sh run        run the selected examples, print the summary, write the GitHub outputs, exit 1 if one failed
#
# environment:
#   EX_PY                 interpreter that has the package under test installed (required)
#   EXAMPLES              space-separated globs of the examples to run, relative to examples/ (required)
#   SKIPPED_DATASETS      set by the datasets step through GITHUB_ENV: datasets whose host was unreachable
#   EXAMPLES_SKIP_FILE    optional, relative to the repository root: one line per example that cannot run on this package, "<file>.py | <reason>"
#                         (blank lines and lines that start with # are ignored). The list is printed with its reasons and each listed example is
#                         counted as skipped and named. The JVM job sets no file and skips nothing.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/examples"

# The files that the space-separated globs in EXAMPLES match, one per line (a glob without a match adds nothing).
list_examples() {
    local -a globs
    local glob
    # EXAMPLES is an input of the workflows (see the header), not a misspelling of the local $examples
    # shellcheck disable=SC2153
    read -r -a globs <<< "$EXAMPLES"
    for glob in "${globs[@]}"; do
        compgen -G "$glob" || true
    done
}

datasets() {

    # Determine which examples are selected
    examples=$(list_examples)

    # EXIT 75 MEANS "THE DATASET HOST IS DOWN", NOT "THE BINDINGS BROKE".
    #
    # On 2026-08-29 the TLS certificate for files.grouplens.org expired and
    # every job here went red on BOTH this fork and upstream's main -- 11
    # and 12 jobs, every OS, Python 3.10 through 3.14 -- at this step,
    # before a single example ran. Re-running does not help: a re-run
    # happens now, and the certificate is still expired.
    #
    # The cost is not the lost coverage, it is that upstream's main showed
    # a failing run titled with a Python bindings PR. A third party letting
    # a certificate lapse must not be indistinguishable from a regression
    # we introduced.
    #
    # So: skip the examples that need the dataset, keep running the rest,
    # and say plainly why. Any other non-zero exit is still a hard failure.
    SKIPPED_DATASETS=""
    fetch() { # fetch <label> <download_data.py args...>
        local label="$1"
        shift
        # rc MUST be captured in the else branch. After `if cmd; then ...; fi`
        # with no else, the IF STATEMENT returns 0, so a `$?` read after `fi`
        # is the if's status and not the command's -- which silently turned
        # every failure into "exited 0" and would have made this skip
        # everything, including real ones.
        local rc=0
        if "$EX_PY" download_data.py "$@"; then
            return 0
        else
            rc=$?
        fi
        if [ "$rc" -eq 75 ]; then
            echo "::warning title=Dataset unavailable::$label host unreachable; skipping examples that need it"
            SKIPPED_DATASETS="$SKIPPED_DATASETS $label"
            return 0
        fi
        echo "::error title=Dataset download failed::$label exited $rc"
        return "$rc"
    }

    if echo "$examples" | grep -qE '(^|[[:space:]])(04_csv_import_documents|05_csv_import_graph|06_vector_search_recommendations)\.py([[:space:]]|$)'; then
        echo "📥 Downloading MovieLens Small dataset..."
        fetch movielens-small movielens-small
    fi

    if echo "$examples" | grep -qE '(^|[[:space:]])(07_stackoverflow_tables_oltp|08_stackoverflow_tables_olap|09_stackoverflow_graph_oltp|10_stackoverflow_graph_olap|13_stackoverflow_hybrid_queries)\.py([[:space:]]|$)'; then
        echo "📥 Downloading Stack Overflow Tiny dataset (no vectors)..."
        fetch stackoverflow-tiny-novec stackoverflow-tiny --no-vectors
    fi

    if echo "$examples" | grep -qE '(^|[[:space:]])(11_vector_index_build|12_vector_search)\.py([[:space:]]|$)'; then
        echo "📥 Downloading Stack Overflow Tiny dataset (with vectors)..."
        fetch stackoverflow-tiny stackoverflow-tiny
    fi

    # HAND THE SKIP LIST TO THE NEXT STEP.
    #
    # Each step is its own shell, so SKIPPED_DATASETS dies here. Without
    # this the download skips politely and the example loop then runs
    # 04_csv_import_documents.py --dataset movielens-small against data
    # that was never fetched -- the job still fails, just later and with a
    # worse error. Caught in review on #6889.
    skipped_datasets="$(echo "$SKIPPED_DATASETS" | xargs)"
    echo "SKIPPED_DATASETS=$skipped_datasets" >> "$GITHUB_ENV"
    if [ -n "$skipped_datasets" ]; then
        echo "::notice title=Datasets unavailable::skipping examples that need:$SKIPPED_DATASETS"
    fi
}

run_examples() {

    echo "🚀 Running Python Examples..."
    echo ""

    # Initialize counters
    total=0
    passed=0
    failed=0
    skipped=0

    # Create results file
    results_file="example-results.txt"
    true > "$results_file"

    # Select example files by pattern(s) in EXAMPLES (space-separated globs)
    examples=$(list_examples | sort)

    if [ -z "$examples" ]; then
        echo "❌ No example files found!"
        exit 1
    fi

    # Detect timeout command (macOS uses gtimeout, Linux uses timeout)
    if command -v gtimeout &> /dev/null; then
        TIMEOUT_CMD="gtimeout"
    else
        TIMEOUT_CMD="timeout"
    fi

    # WHICH EXAMPLES NEED WHICH DATASET. Same grouping the download step
    # uses to decide what to fetch, kept in the same file so the two
    # cannot drift into disagreeing about what a dataset is for.
    needs_dataset() { # <example.py> -> prints the label, or nothing
        case "$1" in
            04_csv_import_documents.py | 05_csv_import_graph.py | 06_vector_search_recommendations.py)
                echo "movielens-small"
                ;;
            07_stackoverflow_tables_oltp.py | 08_stackoverflow_tables_olap.py | 09_stackoverflow_graph_oltp.py | 10_stackoverflow_graph_olap.py | 13_stackoverflow_hybrid_queries.py)
                echo "stackoverflow-tiny-novec"
                ;;
            11_vector_index_build.py | 12_vector_search.py)
                echo "stackoverflow-tiny"
                ;;
        esac
    }

    # EXAMPLES THAT CANNOT RUN ON THIS PACKAGE (EXAMPLES_SKIP_FILE). Nothing is skipped silently: the whole list is printed with its
    # reasons before the first example, and each skipped example is named in the results.
    skip_names=()
    skip_reasons=()
    if [ -n "${EXAMPLES_SKIP_FILE:-}" ]; then
        [ -f "$ROOT/$EXAMPLES_SKIP_FILE" ] || {
            echo "❌ skip list $EXAMPLES_SKIP_FILE not found"
            exit 1
        }
        while IFS= read -r line || [ -n "$line" ]; do
            case "$line" in
                "" | "#"*) continue ;;
            esac
            name="${line%%|*}"
            name="$(echo "$name" | xargs)"
            reason="$(echo "${line#*|}" | xargs)"
            if [ -z "$name" ] || [ "$name" = "$line" ] || [ -z "$reason" ]; then
                echo "❌ $EXAMPLES_SKIP_FILE: malformed line (want '<file>.py | <reason>'): $line"
                exit 1
            fi
            [ -f "$name" ] || {
                echo "❌ $EXAMPLES_SKIP_FILE: $name is not an example"
                exit 1
            }
            skip_names+=("$name")
            skip_reasons+=("$reason")
        done < "$ROOT/$EXAMPLES_SKIP_FILE"
        echo "Skip list ($EXAMPLES_SKIP_FILE): ${#skip_names[@]} example(s)"
        for i in "${!skip_names[@]}"; do
            echo "  ${skip_names[$i]}: ${skip_reasons[$i]}"
        done
        echo ""
    fi
    skip_reason() { # <example.py> -> prints the reason when the example is on the skip list
        local i
        for i in "${!skip_names[@]}"; do
            if [ "${skip_names[$i]}" = "$1" ]; then
                echo "${skip_reasons[$i]}"
                return 0
            fi
        done
    }

    # Run each example
    for example in $examples; do
        total=$((total + 1))

        reason=$(skip_reason "$(basename "$example")")
        if [ -n "$reason" ]; then
            echo "SKIP $example - on the skip list: $reason" | tee -a $results_file
            skipped=$((skipped + 1))
            continue
        fi

        # Skip, do not fail, when the dataset host was unreachable. Counted
        # as skipped and named, so a reader can tell "not run" from "passed".
        need=$(needs_dataset "$(basename "$example")")
        if [ -n "$need" ] && echo " ${SKIPPED_DATASETS:-} " | grep -q " $need "; then
            echo "SKIP $example - dataset '$need' unavailable (third-party host)"
            skipped=$((skipped + 1))
            continue
        fi

        # Set example-specific parameters and timeout
        case "$example" in
            "21_graph_analytical_view_sql.py")
                example_args="--base-cities 1200 --stale-growth-cities 300 --sync-growth-cities 200 --regions 12 --batch-size 300 --sample-limit 5"
                example_name="$example (reduced scale for CI)"
                timeout_duration=900 # 15 minutes
                example_jvm_args=""
                ;;
            "22_numpy_bulk_io.py")
                example_args="--rows 50000 --points 100000 --vectors 5000"
                example_name="$example (reduced scale for CI)"
                timeout_duration=600 # 10 minutes
                example_jvm_args=""
                ;;
            "04_csv_import_documents.py")
                example_args="--dataset movielens-small --export"
                example_name="$example (movielens-small dataset with export)"
                timeout_duration=900 # 15 minutes
                example_jvm_args="-Xmx8g -Xms8g"
                ;;
            "05_csv_import_graph.py")
                example_args="--dataset movielens-small --method sql --import-jsonl ./exports/movielens_small_db.jsonl.tgz --export"
                example_name="$example (movielens-small dataset, embedded sql method, import/export)"
                timeout_duration=900 # 15 minutes
                example_jvm_args=""
                ;;
            "06_vector_search_recommendations.py")
                example_args="--import-jsonl ./exports/movielens_graph_small_db.jsonl.tgz"
                example_name="$example (vector search, import from JSONL)"
                timeout_duration=900 # 15 minutes
                example_jvm_args=""
                ;;
            "07_stackoverflow_tables_oltp.py")
                example_args="--dataset stackoverflow-tiny --db arcadedb_sql --threads 1 --transactions 1000 --batch-size 500 --mem-limit 2g --run-label ci07"
                example_name="$example (stackoverflow-tiny, arcadedb_sql, minimal oltp)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "08_stackoverflow_tables_olap.py")
                example_args="--dataset stackoverflow-tiny --db arcadedb_sql --threads 1 --batch-size 500 --query-runs 1 --query-order fixed --mem-limit 2g --run-label ci08"
                example_name="$example (stackoverflow-tiny, arcadedb_sql, minimal olap)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "09_stackoverflow_graph_oltp.py")
                example_args="--dataset stackoverflow-tiny --db arcadedb_cypher --threads 1 --transactions 1000 --batch-size 500 --mem-limit 2g --run-label ci09"
                example_name="$example (stackoverflow-tiny graph oltp, arcadedb cypher, minimal)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "10_stackoverflow_graph_olap.py")
                example_args="--dataset stackoverflow-tiny --db arcadedb_cypher --threads 1 --batch-size 500 --query-runs 1 --query-order fixed --mem-limit 2g --run-label ci10"
                example_name="$example (stackoverflow-tiny graph olap, arcadedb cypher, minimal)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "11_vector_index_build.py")
                example_args="--backend arcadedb_sql --dataset stackoverflow-tiny --threads 1 --mem-limit 2g --batch-size 500 --beam-width 100 --quantization NONE --run-label ci11_arcadedb_sql"
                example_name="$example (vector build, arcadedb_sql backend, minimal)"
                timeout_duration=1200
                example_jvm_args=""
                ;;
            "12_vector_search.py")
                db_path=$(find ./my_test_databases -maxdepth 1 -type d -name 'backend=arcadedb_sql_dataset=stackoverflow-tiny_*run=ci11_arcadedb_sql' | head -n 1)
                if [ -z "$db_path" ]; then
                    echo "❌ Missing build output for 12_vector_search.py. Ensure 11_vector_index_build.py ran first." | tee -a $results_file
                    failed=$((failed + 1))
                    echo ""
                    continue
                fi
                example_args="--backend arcadedb_sql --dataset stackoverflow-tiny --db-path $db_path --ef-search-values 100 --k 10 --query-limit 100 --query-runs 1 --query-order fixed --threads 1 --mem-limit 2g --run-label ci12_arcadedb_sql"
                example_name="$example (vector search, arcadedb_sql backend, minimal)"
                timeout_duration=1200
                example_jvm_args=""
                ;;
            "13_stackoverflow_hybrid_queries.py")
                example_args="--dataset stackoverflow-tiny --batch-size 500 --encode-batch-size 64 --top-k 5 --candidate-limit 100 --min-reputation 100 --infer-sample-limit 5000 --run-label ci13"
                example_name="$example (hybrid pipeline, tiny + minimal limits)"
                timeout_duration=1800
                example_jvm_args=""
                ;;
            "14_lifecycle_timing.py")
                example_args="--runs 1 --table-records 2000 --graph-vertices 500 --vector-records 500 --vector-dimensions 32 --query-runs 10 --jvm-heap 4g"
                example_name="$example (lifecycle benchmark, minimal)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "15_import_database_vs_transactional_table_ingest.py")
                example_args="--rows-per-table 2000 --tables 2 --columns 6 --string-size 32 --batch-size 500 --async-parallel 1 --parallel 1 --heap-size 4g --work-dir ./my_test_databases/import_vs_txn_dummy_ci"
                example_name="$example (table ingest benchmark, minimal)"
                timeout_duration=1200
                example_jvm_args=""
                ;;
            "16_import_database_vs_transactional_graph_ingest.py")
                example_args="--vertices 2000 --edges 4000 --vertex-int-props 3 --vertex-str-props 2 --edge-int-props 1 --edge-str-props 1 --string-size 32 --batch-size 500 --async-parallel 1 --parallel 1 --heap-size 4g --work-dir ./my_test_databases/import_vs_txn_graph_ci"
                example_name="$example (graph ingest benchmark, minimal)"
                timeout_duration=1200
                example_jvm_args=""
                ;;
            "17_timeseries_end_to_end.py")
                example_args="--hours 2 --interval-minutes 10"
                example_name="$example (time-series end-to-end, minimal)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "18_geo_predicates_wkt.py")
                example_args=""
                example_name="$example (geo predicates with WKT points and polygons)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "19_hash_index_exact_match.py")
                example_args=""
                example_name="$example (hash index exact-match workflow)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            "20_graph_algorithms_route_planning.py")
                example_args=""
                example_name="$example (graph algorithms route planning workflow)"
                timeout_duration=900
                example_jvm_args=""
                ;;
            *)
                example_args=""
                example_name="$example"
                timeout_duration=900 # 15 minutes default
                example_jvm_args=""
                ;;
        esac

        log_file="${example%.py}.log"

        echo "----------------------------------------"
        echo "📝 Running: $example_name"
        echo "----------------------------------------"

        # The argument string of the example, split on spaces into words (no globbing)
        read -r -a example_argv <<< "$example_args"

        # Run the example with appropriate parameters
        if [ -n "$example_jvm_args" ]; then
            if ARCADEDB_JVM_ARGS="$example_jvm_args" JAVA_TOOL_OPTIONS="" _JAVA_OPTIONS="" $TIMEOUT_CMD $timeout_duration "$EX_PY" "$example" "${example_argv[@]}" > "$log_file" 2>&1; then
                exit_code=0
            else
                exit_code=$?
            fi
        else
            if ARCADEDB_JVM_ARGS="" JAVA_TOOL_OPTIONS="" _JAVA_OPTIONS="" $TIMEOUT_CMD $timeout_duration "$EX_PY" "$example" "${example_argv[@]}" > "$log_file" 2>&1; then
                exit_code=0
            else
                exit_code=$?
            fi
        fi
        # ANSWERS, NOT ONLY A CLEAN EXIT (humemai/arcadedb-embedded-python#12). Example 10 once returned 0 rows for
        # 5 of its 10 queries on every platform and passed. After a clean run, the
        # same arguments on the pure-Python reference backend must give the same
        # row count and hash for every query.
        if [ $exit_code -eq 0 ] && [ "$example" = "10_stackoverflow_graph_olap.py" ]; then
            ref_args="${example_args/--db arcadedb_cypher/--db python_memory}"
            ref_args="${ref_args/--run-label ci10/--run-label ci10ref}"
            read -r -a ref_argv <<< "$ref_args"
            ref_log="${example%.py}-reference.log"
            if ! $TIMEOUT_CMD $timeout_duration "$EX_PY" "$example" "${ref_argv[@]}" > "$ref_log" 2>&1; then
                echo "❌ reference run (python_memory) of $example failed; last 20 lines:"
                tail -n 20 "$ref_log"
                exit_code=1
            elif ! "$EX_PY" scripts/compare_query_hashes.py "$log_file" "$ref_log"; then
                echo "❌ $example: answers differ from the python_memory reference (above)"
                exit_code=1
            fi
        fi
        if [ $exit_code -eq 0 ]; then
            echo "✅ PASSED: $example_name" | tee -a $results_file
            passed=$((passed + 1))
        else
            if [ $exit_code -eq 124 ]; then
                echo "⏱️  TIMEOUT: $example_name (exceeded $((timeout_duration / 60)) minutes)" | tee -a $results_file
                failed=$((failed + 1))
            else
                echo "❌ FAILED: $example_name (exit code: $exit_code)" | tee -a $results_file
                failed=$((failed + 1))
            fi
            # Show last 20 lines of error log
            echo "Last 20 lines of output:"
            tail -n 20 "$log_file"
        fi
        echo ""
    done

    # Print summary
    echo "========================================"
    echo "📊 EXAMPLE TEST SUMMARY"
    echo "========================================"
    echo "Total:   $total"
    echo "Passed:  $passed ✅"
    echo "Failed:  $failed ❌"
    echo "Skipped: $skipped ⏭️"
    echo "========================================"
    echo ""

    echo "Examples pattern(s): $EXAMPLES"

    # Output to GitHub Actions
    {
        echo "total=$total"
        echo "passed=$passed"
        echo "failed=$failed"
        echo "skipped=$skipped"
    } >> "$GITHUB_OUTPUT"

    # Show detailed results
    echo "Detailed Results:"
    cat $results_file

    # Save list of examples for summary (before the failure exit, so
    # the summary is populated even when some examples fail)
    echo "$examples" > examples-ran.txt

    # Exit with error if any failed
    if [ $failed -gt 0 ]; then
        echo "❌ Some examples failed!"
        exit 1
    else
        echo "✅ All examples passed!"
    fi
}

case "${1:-}" in
    datasets) datasets ;;
    run) run_examples ;;
    *)
        echo "usage: $0 datasets|run" >&2
        exit 2
        ;;
esac
