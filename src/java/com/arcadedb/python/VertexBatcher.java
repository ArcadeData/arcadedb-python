/*
 * Python-bindings bridge: bulk vertex creation for GraphBatch.
 *
 * GraphBatch.create_vertices() from Python marshals an Object[][] property
 * matrix element-by-element (~19us/vertex measured vs 6.2us Java-native).
 * This helper accepts the rows as ONE JSON string (json.dumps costs
 * ~0.2us/vertex Python-side) and returns the created RIDs as ONE
 * semicolon-joined string, so the whole batch costs two bulk string copies.
 *
 * Property values are str/int/float/bool/null, nested lists and maps thereof,
 * and the tagged values DocumentBatcher.untag decodes (date-time, date,
 * BigDecimal, byte[]) sitting directly in a row; the Python side falls back to
 * the matrix path for anything else.
 */
package com.arcadedb.python;

import com.arcadedb.database.RID;
import com.arcadedb.graph.GraphBatch;
import com.arcadedb.serializer.json.JSONArray;
import com.arcadedb.serializer.json.JSONObject;

public final class VertexBatcher {

  private VertexBatcher() {
  }

  public static String createVerticesJson(final GraphBatch batch, final String typeName, final String jsonRows) {
    return createVerticesJson(batch, typeName, jsonRows, null, null);
  }

  /**
   * As {@link #createVerticesJson(GraphBatch, String, String)}, with float vectors that did not go through the JSON text:
   * {@code vectorNames[c]} is set on vertex i to row i of {@code vectors[c]}, a {@link FloatVectors.Matrix} with one row per
   * JSON row. Both are null when there are none.
   */
  public static String createVerticesJson(final GraphBatch batch, final String typeName, final String jsonRows,
      final String[] vectorNames, final Object[] vectors) {
    final JSONArray rows = new JSONArray(jsonRows);
    FloatVectors.check(vectorNames, vectors, rows.length());
    final int extra = vectorNames == null ? 0 : vectorNames.length;
    final Object[][] matrix = new Object[rows.length()][];
    for (int i = 0; i < rows.length(); i++) {
      final JSONObject row = rows.getJSONObject(i);
      final Object[] flat = new Object[(row.length() + extra) * 2];
      int j = 0;
      for (final String key : row.keySet()) {
        flat[j++] = key;
        flat[j++] = DocumentBatcher.value(row.isNull(key) ? null : row.get(key));
      }
      for (int c = 0; c < extra; c++) {
        flat[j++] = vectorNames[c];
        flat[j++] = ((FloatVectors.Matrix) vectors[c]).row(i);
      }
      matrix[i] = flat;
    }
    final RID[] rids = batch.createVertices(typeName, matrix);
    final StringBuilder sb = new StringBuilder(rids.length * 8);
    for (int i = 0; i < rids.length; i++) {
      if (i > 0)
        sb.append(';');
      sb.append(rids[i]);
    }
    return sb.toString();
  }

  /**
   * Typed vertex creation: the rows arrive as whole columns (the layout of {@link DocumentBatcher#insertColumns}: a long[], a
   * double[], a boolean[], a StringColumn, a FloatVectors.Matrix, ...) and cross once per column instead of as one JSON text, and the
   * created RIDs go back as one long[] of 2 * n numbers, {@code [bucket id, position]} per vertex in row order, instead of one joined
   * text. The engine call is the one {@link #createVerticesJson} makes, with the same key and value per property, so the stored
   * records are the same: {@code names[c]} is the property of column c, in the order of the rows' keys; {@code nulls}, or its element
   * c, is null when the column has no null, else one flag per row (a flagged row is set to null, as a JSON null is on the text path).
   * A long that fits an int is boxed as an Integer and any other as a Long, because that is what the engine's JSON parser makes of
   * the number token, and the text path stored it so.
   */
  public static long[] createVerticesColumns(final GraphBatch batch, final String typeName, final String[] names,
      final Object[] columns, final Object[] nulls, final int n) {
    DocumentBatcher.checkColumns(names, columns, n);
    checkNulls(names, nulls, n);
    final Object[][] matrix = new Object[n][];
    for (int i = 0; i < n; i++)
      matrix[i] = flatRow(names, columns, nulls, i);
    final RID[] rids = batch.createVertices(typeName, matrix);
    final long[] packed = new long[rids.length * 2];
    for (int i = 0; i < rids.length; i++) {
      packed[2 * i] = rids[i].getBucketId();
      packed[2 * i + 1] = rids[i].getPosition();
    }
    return packed;
  }

  /** One flag array per name or none at all, each as long as the rows. */
  static void checkNulls(final String[] names, final Object[] nulls, final int n) {
    if (nulls == null)
      return;
    if (nulls.length != names.length)
      throw new IllegalArgumentException("nulls has " + nulls.length + " entries for " + names.length + " names");
    for (int c = 0; c < nulls.length; c++)
      if (nulls[c] != null && (!(nulls[c] instanceof boolean[] flags) || flags.length != n))
        throw new IllegalArgumentException("the null flags of '" + names[c] + "' are not a boolean[] of " + n + " rows");
  }

  /** Row {@code i} as the engine's property varargs: name, value, name, value, ... (empty for a row without properties). */
  static Object[] flatRow(final String[] names, final Object[] columns, final Object[] nulls, final int i) {
    final Object[] flat = new Object[names.length * 2];
    for (int c = 0; c < names.length; c++) {
      flat[2 * c] = names[c];
      flat[2 * c + 1] = cell(columns[c], nulls == null ? null : (boolean[]) nulls[c], i);
    }
    return flat;
  }

  private static Object cell(final Object column, final boolean[] nulls, final int i) {
    if (nulls != null && nulls[i])
      return null;
    if (column instanceof long[] a) {
      final long v = a[i];
      return v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE ? (Object) Integer.valueOf((int) v) : (Object) Long.valueOf(v);
    }
    if (column instanceof double[] a)
      return Double.valueOf(a[i]);
    if (column instanceof boolean[] a)
      return Boolean.valueOf(a[i]);
    if (column instanceof StringColumn s)
      return s.get(i);
    if (column instanceof FloatVectors.Matrix m)
      return m.row(i);
    if (column instanceof TemporalColumn t)
      return t.get(i);
    return ((Object[]) column)[i];
  }
}
