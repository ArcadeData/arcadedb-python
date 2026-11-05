/*
 * Python-bindings bridge: batched row transport.
 *
 * Materializing large result sets from Python via JPype costs 2+C boundary
 * crossings per row (hasNext/next + one getProperty per column), which
 * dominates wide scans (measured 15-21x slower than Java-native iteration).
 * This helper serializes up to `max` rows into ONE JSON-array string
 * Java-side, so Python pays a single crossing per batch and parses with the
 * C-fast json module (measured ~6x faster end-to-end, ~2.7x of Java-native).
 *
 * Rows are serialized property-by-property into the engine's JSONObject,
 * which since 26.7.2 (#4967) serializes primitive arrays like float[] as
 * real JSON arrays natively.
 *
 * This class has NO temporal encoding of its own (#10522): every date and
 * date-time is written by JSONObject.put, the writer the engine uses for the
 * rows of a projection and for the members of a list or a map of any row, and
 * ColumnBatcher's epoch milliseconds are the same numbers. A DATE is the epoch
 * milliseconds of its UTC midnight, never a count of days, whatever the JVM
 * time zone. The one place the engine writes days is the top level of a
 * record's own toJSON() (Result.to_json() of a SELECT FROM T row), which is
 * the form fromJSON() reads back for a DATE column; to_json_list() is not that
 * round trip, and test_temporal_writers_agree.py pins both.
 *
 * Compiled into arcadedb-python-bridge.jar during the wheel build
 * (scripts/Dockerfile.build and scripts/build-native.sh) and consumed by
 * ResultSet.to_json_list() in the Python bindings. Values carry JSON-native
 * types (DATE and DATETIME as epoch-millisecond integers), as documented on
 * the Python side.
 */
package com.arcadedb.python;

import com.arcadedb.query.sql.executor.Result;
import com.arcadedb.query.sql.executor.ResultSet;
import com.arcadedb.serializer.json.JSONArray;
import com.arcadedb.serializer.json.JSONObject;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class RowBatcher {

  /** The key of a tagged value; Python reads the same character (the tag TypedRows uses). */
  static final String TAG = "\u0001";

  private RowBatcher() {
  }

  /**
   * JSON has no NaN or infinity, and the engine's writers turn them into null. This returns {@code value} with every
   * NaN, Infinity and -Infinity (a Double or Float, or an element of a float[], a double[], a Collection or a Map, at any
   * depth) replaced by the tag {@code {"\u0001": ["f", "NaN"]}} (the text is {@code Double.toString}), which Python turns
   * back into a float. A value with nothing non-finite is returned as it is, with no copy, so the common row costs one
   * type test per value.
   */
  static Object keepNonFinite(final Object value) {
    if (value instanceof Double || value instanceof Float) {
      final double d = ((Number) value).doubleValue();
      return Double.isFinite(d) ? value : tag(d);
    }
    if (value instanceof float[] floats) {
      for (final float f : floats)
        if (!Float.isFinite(f)) {
          final List<Object> out = new ArrayList<>(floats.length);
          for (final float x : floats)
            out.add(Float.isFinite(x) ? (Object) x : tag(x));
          return out;
        }
      return value;
    }
    if (value instanceof double[] doubles) {
      for (final double d : doubles)
        if (!Double.isFinite(d)) {
          final List<Object> out = new ArrayList<>(doubles.length);
          for (final double x : doubles)
            out.add(Double.isFinite(x) ? (Object) x : tag(x));
          return out;
        }
      return value;
    }
    if (value instanceof Collection<?> collection) {
      List<Object> out = null;
      int i = 0;
      for (final Object element : collection) {
        final Object kept = keepNonFinite(element);
        if (kept != element && out == null) {
          out = new ArrayList<>(collection.size());
          int j = 0;
          for (final Object before : collection) {
            if (j++ == i)
              break;
            out.add(before);
          }
        }
        if (out != null)
          out.add(kept);
        i++;
      }
      return out != null ? out : value;
    }
    if (value instanceof Map<?, ?> map) {
      Map<Object, Object> out = null;
      int i = 0;
      for (final Map.Entry<?, ?> entry : map.entrySet()) {
        final Object element = entry.getValue();
        final Object kept = keepNonFinite(element);
        if (kept != element && out == null) {
          out = new LinkedHashMap<>(map.size() * 2);
          int j = 0;
          for (final Map.Entry<?, ?> before : map.entrySet()) {
            if (j++ == i)
              break;
            out.put(before.getKey(), before.getValue());
          }
        }
        if (out != null)
          out.put(entry.getKey(), kept);
        i++;
      }
      return out != null ? out : value;
    }
    return value;
  }

  private static JSONObject tag(final double nonFinite) {
    return new JSONObject().put(TAG, new JSONArray(List.of("f", Double.toString(nonFinite))));
  }

  /**
   * Serialize up to {@code max} rows of the result set into a JSON array
   * string. Returns fewer than {@code max} rows only when the result set is
   * drained, so a caller can stop after a short batch instead of calling again
   * to see {@code "[]"}, and a drained result set is closed here, so the caller must not use it again.
   *
   * <p>The builder starts at the default size and grows: a 64 KB start was
   * allocated on every call and cost 2.5x the engine lookup on a one-row
   * result, while a large batch grows the default buffer at no measurable cost
   * (0.97x to 1.00x on a 9,892-row scan).
   */
  public static String nextJsonBatch(final ResultSet rs, final int max) {
    final StringBuilder sb = new StringBuilder();
    sb.append('[');
    int n = 0;
    while (n < max && rs.hasNext()) {
      final Result row = rs.next();
      if (n > 0)
        sb.append(',');
      appendRow(sb, row);
      n++;
    }
    if (n < max)
      closeDrained(rs);
    return sb.append(']').toString();
  }

  /**
   * The result set is drained, so the caller will not come back: close it here instead of making Python cross the
   * bridge once more just to call close() (a JPype call costs 3 to 4 microseconds, a fifth of a one-row read).
   * Best effort, like the Python-side close it replaces.
   */
  static void closeDrained(final ResultSet rs) {
    try {
      rs.close();
    } catch (final Exception ignored) {
      // closing is hygiene; the rows were read
    }
  }

  private static void appendRow(final StringBuilder sb, final Result row) {
    final JSONObject obj = new JSONObject();
    for (final String property : row.getPropertyNames()) {
      // No conversion of our own: JSONObject.put anchors a LocalDate at UTC midnight whatever the JVM zone (#9037), writes a
      // java.util.Calendar as its epoch milliseconds like a Date (#9630), and writes a date past the range of a long of
      // milliseconds as its ISO text (#10483), where the encoder this replaced threw and failed the whole batch. The members
      // of a list or a map go the same way
      // The one exception: NaN and the infinities, which the writers would turn into null, travel as a tag
      final Object value = keepNonFinite(row.getProperty(property));
      obj.put(property, value);
    }
    sb.append(obj);
  }
}
