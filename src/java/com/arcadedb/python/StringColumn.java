/*
 * Python-bindings bridge: a string column of Database.insert_columns that crosses as one UTF-8 blob plus an int32 offsets array.
 *
 * The JPype guidance names one conversion per element as the slow way across the boundary, and an Object[] of Strings costs one
 * conversion per row (and one Java String per distinct value, all alive at once). Here the whole column is ONE byte[] (the UTF-8
 * text of every row, back to back), ONE int[] of row + 1 offsets into it, and optionally ONE boolean[] marking the null rows.
 * Row i is blob[offsets[i], offsets[i + 1]), decoded when the document of that row is built, so the Strings are short-lived.
 *
 * The decoded String is the one Python's own str holds: UTF-8 round-trips every code point, so a character beyond the BMP becomes
 * its surrogate pair, U+0000 stays a NUL and an empty row is the empty String. Python sends no column it cannot encode (a lone
 * surrogate has no UTF-8 form): that column keeps the per-element path.
 *
 * Compiled into arcadedb-python-bridge.jar during the wheel build and consumed by DocumentBatcher.insertColumns() and
 * insertColumnsParallel().
 */
package com.arcadedb.python;

import java.nio.charset.StandardCharsets;

public final class StringColumn {
  private final byte[]    blob;
  private final int[]     offsets;
  private final boolean[] nulls;
  private final int       rows;

  /**
   * @param blob    the UTF-8 text of all rows, back to back
   * @param offsets rows + 1 non-decreasing offsets into {@code blob}, the first 0 and the last at most {@code blob.length}
   * @param nulls   null for a column without a null, else one flag per row (a flagged row has an empty range and reads as null)
   */
  public StringColumn(final byte[] blob, final int[] offsets, final boolean[] nulls) {
    if (offsets.length < 1)
      throw new IllegalArgumentException("a string column needs rows + 1 offsets, got none");
    this.rows = offsets.length - 1;
    if (offsets[0] != 0)
      throw new IllegalArgumentException("the first offset of a string column must be 0, got " + offsets[0]);
    for (int i = 0; i < rows; i++)
      if (offsets[i + 1] < offsets[i])
        throw new IllegalArgumentException("the offsets of a string column must not decrease (row " + i + ")");
    if (offsets[rows] > blob.length)
      throw new IllegalArgumentException(
          "the last offset " + offsets[rows] + " of a string column is beyond its " + blob.length + " bytes");
    if (nulls != null && nulls.length != rows)
      throw new IllegalArgumentException("a string column has " + nulls.length + " null flags for " + rows + " rows");
    this.blob = blob;
    this.offsets = offsets;
    this.nulls = nulls;
  }

  public int rows() {
    return rows;
  }

  /** The String of row {@code i}, or null for a null row. */
  public String get(final int i) {
    if (nulls != null && nulls[i])
      return null;
    final int from = offsets[i];
    return new String(blob, from, offsets[i + 1] - from, StandardCharsets.UTF_8);
  }
}
