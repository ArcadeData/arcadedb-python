/*
 * Python-bindings bridge: binary columnar transport.
 *
 * Encodes up to `max` rows of a ResultSet into ONE byte[]: a JSON header
 * (column names/types/sizes) followed by per-column buffers — fixed-width
 * little-endian for numerics/bools/temporals (epoch millis), offset+UTF-8
 * for strings, plus a null bitmap per column. A DECIMAL column ("dec" type) uses the string layout with each
 * BigDecimal written in full, so no digit is lost to a double. Columns whose values don't
 * fit those encodings are serialized as one JSON array ("json" type). Python decodes with numpy.frombuffer
 * (ResultSet.to_columns / the fast to_dataframe path). Measured ~1.2x of
 * Java-native iteration on 100k-row scans (vs 3.4x for the JSON row path).
 *
 * A "json" column is written with the tagged encoding of TypedRows, not the engine's JSON writer: a date, datetime,
 * DECIMAL or non-String map key inside a list or a map reaches Python as the type ResultSet.to_list() gives it, and the
 * values JSON cannot carry (NaN, RIDs, ...) are the engine objects of the side array that is returned with the buffer.
 * With {@code micros} a datetime column is written as epoch MICROseconds ("dtu", what ResultSet.to_dataframe reads) and
 * any column holding a DATE or a time-zone-aware temporal is a "json" column, so Python gets date and datetime objects as
 * the row readers return them.
 */
package com.arcadedb.python;

import com.arcadedb.query.sql.executor.Result;
import com.arcadedb.query.sql.executor.ResultSet;
import com.arcadedb.serializer.json.JSONArray;
import com.arcadedb.serializer.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class ColumnBatcher {

  private ColumnBatcher() {
  }

  /** Wall-clock-consistent epoch millis: naive temporals are read as UTC so
   * numpy's (naive) datetime64 shows the same wall-clock the record carries.
   * A DATE is its UTC midnight in milliseconds, not a count of days: the column is a datetime64[ms], and it is the
   * number RowBatcher gets from the engine's JSONObject for the same value (#10522, test_temporal_writers_agree.py).
   * It is not a JSON writer, so there is nothing for it to disagree with in Result.toJSON(). */
  private static long toEpochMillis(final Object v) {
    if (v instanceof Date d)
      return d.getTime();
    if (v instanceof Calendar c)
      return c.getTimeInMillis();
    if (v instanceof LocalDateTime ldt)
      return ldt.toInstant(ZoneOffset.UTC).toEpochMilli();
    if (v instanceof LocalDate ld)
      return ld.atStartOfDay().toInstant(ZoneOffset.UTC).toEpochMilli();
    if (v instanceof Instant i)
      return i.toEpochMilli();
    if (v instanceof ZonedDateTime z)
      return z.toInstant().toEpochMilli();
    if (v instanceof OffsetDateTime o)
      return o.toInstant().toEpochMilli();
    throw new IllegalArgumentException("not a temporal: " + v.getClass());
  }

  /** Epoch microseconds of a naive datetime: what {@link #toEpochMillis} is in milliseconds, without the cut. */
  private static long toEpochMicros(final Object v) {
    if (v instanceof LocalDateTime ldt)
      return Math.addExact(Math.multiplyExact(ldt.toEpochSecond(ZoneOffset.UTC), 1_000_000L), ldt.getNano() / 1000);
    if (v instanceof Date d)
      return Math.multiplyExact(d.getTime(), 1000L);
    if (v instanceof Calendar c)
      return Math.multiplyExact(c.getTimeInMillis(), 1000L);
    throw new IllegalArgumentException("not a naive datetime: " + v.getClass());
  }

  /** Python's datetime covers years 1 to 9999; a datetime outside it stays an engine object (TypedRows does the same). */
  private static boolean yearFits(final LocalDateTime ldt) {
    return ldt.getYear() >= 1 && ldt.getYear() <= 9999;
  }

  /** The columns of up to {@code max} rows, temporals in milliseconds: {@code {byte[] buffer, Object[] side}}. */
  public static Object[] nextColumnBatch(final ResultSet rs, final int max, final String joinedColumns) {
    return nextColumnBatch(rs, max, joinedColumns, false);
  }

  /**
   * The columns of up to {@code max} rows as {@code {byte[] buffer, Object[] side}}: {@code side} holds the engine objects
   * that the tagged JSON of a "json" column refers to by index. With {@code micros} see the class comment.
   */
  public static Object[] nextColumnBatch(final ResultSet rs, final int max, final String joinedColumns,
      final boolean micros) {
    // empty column spec: the column set is the union of every row's property names, in order of first appearance.
    // A document is schemaless, so the first row says nothing about the others: taking its names alone dropped every
    // property it lacked (humemai/arcadedb-embedded-python#113). Each batch reports its own set in its header.
    String[] columns = parseColumnSpec(joinedColumns);
    final List<Result> results = new ArrayList<>(Math.min(max, 1 << 16));
    while (results.size() < max && rs.hasNext())
      results.add(rs.next());
    if (columns == null) {
      final Set<String> names = new LinkedHashSet<>();
      // Rows of one type almost always list the same names in the same order, so a row is compared with the one
      // before it name by name (String.equals answers an identical instance at once) and only a row that differs
      // is hashed into the set.
      String[] previous = null;
      for (final Result row : results) {
        final Set<String> rowNames = row.getPropertyNames();
        if (previous != null && rowNames.size() == previous.length) {
          int i = 0;
          for (final String name : rowNames)
            if (!name.equals(previous[i++])) {
              i = -1;
              break;
            }
          if (i >= 0)
            continue;
        }
        names.addAll(rowNames);
        previous = rowNames.toArray(new String[0]);
      }
      columns = names.toArray(new String[0]);
    }
    final int n = columns.length;
    final List<Object[]> rows = new ArrayList<>(results.size());
    for (final Result row : results) {
      final Object[] vals = new Object[n];
      for (int c = 0; c < n; c++)
        vals[c] = row.getProperty(columns[c]);
      rows.add(vals);
    }
    final int count = rows.size();

    // The header goes through JSONObject/JSONArray rather than string concatenation: a column name is an
    // arbitrary query projection alias, so a name carrying a quote, a backslash or a control character used to
    // emit invalid JSON and break the decode of the WHOLE batch on the Python side. RowBatcher already does it
    // this way; the cost is n objects per batch, not per row.
    final JSONArray headerColumns = new JSONArray();
    final List<Object> side = new ArrayList<>();
    final ByteArrayOutputStream body = new ByteArrayOutputStream(count * n * 8);

    for (int c = 0; c < n; c++) {
      // null bitmap
      final byte[] nulls = new byte[(count + 7) / 8];
      // detect the column type across ALL values; mixed/unsupported -> json
      String type = null;
      for (int r = 0; r < count; r++) {
        final Object v = rows.get(r)[c];
        if (v == null)
          continue;
        final String t;
        if (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte)
          t = "i8";
        else if (v instanceof Double || v instanceof Float)
          t = "f8";
        else if (v instanceof Boolean)
          t = "b1";
        else if (v instanceof Date || v instanceof Calendar || v instanceof LocalDateTime
            || v instanceof LocalDate || v instanceof Instant
            || v instanceof ZonedDateTime || v instanceof OffsetDateTime) {
          if (!micros)
            t = "dt";
          else if (v instanceof LocalDateTime ldt ? yearFits(ldt) : v instanceof Date || v instanceof Calendar)
            // a naive datetime is epoch microseconds; a DATE, a time-zone-aware temporal or a year Python cannot hold
            // is read through the tagged JSON, as the row readers return it
            t = "dtu";
          else
            t = "json";
        }
        else if (v instanceof String || v instanceof Character)
          t = "str";
        else if (v instanceof java.math.BigDecimal)
          t = "dec";
        else if (v instanceof float[])
          t = "f4v";
        else if (v instanceof double[])
          t = "f8v";
        else
          t = "json";
        if (type == null)
          type = t;
        else if (!type.equals(t)) {
          // numeric widening is fine; anything else degrades to json
          if ((type.equals("i8") && t.equals("f8")) || (type.equals("f8") && t.equals("i8")))
            type = "f8";
          else {
            type = "json";
            break;
          }
        }
      }
      if (type == null)
        type = "str"; // all-null column

      int vdim = -1;
      if (type.equals("f4v") || type.equals("f8v")) {
        // vector columns must be uniform-length; ragged degrades to json
        for (int r = 0; r < count && vdim != -2; r++) {
          final Object v = rows.get(r)[c];
          if (v == null)
            continue;
          final int len = type.equals("f4v") ? ((float[]) v).length : ((double[]) v).length;
          if (vdim == -1)
            vdim = len;
          else if (vdim != len)
            vdim = -2;
        }
        if (vdim < 0)
          type = "json";
      }

      byte[] colBuf;
      if (type.equals("i8") || type.equals("dt") || type.equals("dtu")) {
        final ByteBuffer bb = ByteBuffer.allocate(count * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null) {
            nulls[r >> 3] |= (1 << (r & 7));
            bb.putLong(0);
          } else if (type.equals("dt"))
            bb.putLong(toEpochMillis(v));
          else if (type.equals("dtu"))
            bb.putLong(toEpochMicros(v));
          else
            bb.putLong(((Number) v).longValue());
        }
        colBuf = bb.array();
      } else if (type.equals("f8")) {
        final ByteBuffer bb = ByteBuffer.allocate(count * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null) {
            nulls[r >> 3] |= (1 << (r & 7));
            bb.putDouble(Double.NaN);
          } else
            bb.putDouble(((Number) v).doubleValue());
        }
        colBuf = bb.array();
      } else if (type.equals("b1")) {
        final ByteBuffer bb = ByteBuffer.allocate(count);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null) {
            nulls[r >> 3] |= (1 << (r & 7));
            bb.put((byte) 0);
          } else
            bb.put((byte) (((Boolean) v) ? 1 : 0));
        }
        colBuf = bb.array();
      } else if (type.equals("f4v")) {
        final ByteBuffer bb = ByteBuffer.allocate(count * vdim * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null) {
            nulls[r >> 3] |= (1 << (r & 7));
            for (int k = 0; k < vdim; k++)
              bb.putFloat(Float.NaN);
          } else
            for (final float x : (float[]) v)
              bb.putFloat(x);
        }
        colBuf = bb.array();
      } else if (type.equals("f8v")) {
        final ByteBuffer bb = ByteBuffer.allocate(count * vdim * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null) {
            nulls[r >> 3] |= (1 << (r & 7));
            for (int k = 0; k < vdim; k++)
              bb.putDouble(Double.NaN);
          } else
            for (final double x : (double[]) v)
              bb.putDouble(x);
        }
        colBuf = bb.array();
      } else if (type.equals("json")) {
        // one JSON array for the whole column, in the tagged encoding of TypedRows: what the engine's JSON writer turns
        // into epoch milliseconds, a double or a text key (a date or Decimal in a list, an int map key) stays what the
        // row readers return, and what JSON cannot carry goes to the side array
        final StringBuilder arr = new StringBuilder();
        arr.append('[');
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (r > 0)
            arr.append(',');
          if (v == null)
            nulls[r >> 3] |= (1 << (r & 7));
          TypedRows.append(arr, v, side);
        }
        arr.append(']');
        colBuf = arr.toString().getBytes(StandardCharsets.UTF_8);
      } else { // strings and decimals: int32 offsets (count+1) then utf8 bytes
        final ByteArrayOutputStream chars = new ByteArrayOutputStream(count * 16);
        final ByteBuffer offs = ByteBuffer.allocate((count + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        int pos = 0;
        offs.putInt(0);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null)
            nulls[r >> 3] |= (1 << (r & 7));
          else {
            // toPlainString: a BigDecimal's toString may use an exponent ("1E+3"), which is exact but needless here
            final String text = v instanceof java.math.BigDecimal d ? d.toPlainString() : v.toString();
            final byte[] b = text.getBytes(StandardCharsets.UTF_8);
            chars.write(b, 0, b.length);
            pos += b.length;
          }
          offs.putInt(pos);
        }
        final byte[] charBytes = chars.toByteArray();
        final ByteBuffer bb = ByteBuffer.allocate(offs.capacity() + charBytes.length);
        bb.put(offs.array());
        bb.put(charBytes);
        colBuf = bb.array();
      }

      final JSONObject columnHeader = new JSONObject();
      columnHeader.put("name", columns[c]);
      columnHeader.put("type", type);
      columnHeader.put("nulls", nulls.length);
      columnHeader.put("bytes", colBuf.length);
      if (vdim > 0 && (type.equals("f4v") || type.equals("f8v")))
        columnHeader.put("dim", vdim);
      headerColumns.put(columnHeader);
      body.write(nulls, 0, nulls.length);
      body.write(colBuf, 0, colBuf.length);
    }

    final JSONObject header = new JSONObject();
    header.put("count", count);
    header.put("cols", headerColumns);

    final byte[] headerBytes = header.toString().getBytes(StandardCharsets.UTF_8);
    final byte[] bodyBytes = body.toByteArray();
    final ByteBuffer out = ByteBuffer.allocate(4 + headerBytes.length + bodyBytes.length)
        .order(ByteOrder.LITTLE_ENDIAN);
    out.putInt(headerBytes.length);
    out.put(headerBytes);
    out.put(bodyBytes);
    return new Object[] { out.array(), side.toArray() };
  }

  /**
   * Decodes the caller-supplied column spec, which pins the column set across the batches of one result set.
   * <p>
   * The canonical form is a JSON array, because the names are arbitrary projection aliases: the legacy
   * {@code ';'}-joined form silently split a name that contained a semicolon into two bogus columns. The legacy
   * form is still accepted so an older Python side paired with a newer bridge jar keeps working.
   *
   * @return the pinned column names, or {@code null} when the spec is empty and the set must be derived from the
   *         first row
   */
  private static String[] parseColumnSpec(final String spec) {
    if (spec == null || spec.isEmpty())
      return null;
    if (spec.charAt(0) != '[')
      return spec.split(";");
    final JSONArray names = new JSONArray(spec);
    final String[] columns = new String[names.length()];
    for (int i = 0; i < columns.length; i++)
      columns[i] = names.getString(i);
    return columns;
  }
}
