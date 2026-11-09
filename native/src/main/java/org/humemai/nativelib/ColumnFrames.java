/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Result rows as one column frame for the Python native backend (arcadedb_n_next_col): the layout the JVM wheel's ColumnBatcher
 * writes, so one Python decoder (arcadedb/_columnar.py) reads both. A frame is
 *
 *   int32 LE header length, header JSON, then for each column its null bitmap and its buffer
 *
 * The header is {"count":n,"done":bool,"cols":[{"name":..,"type":..,"nulls":bytes,"bytes":bytes[,"dim":d]},..]} ("done" is an addition
 * to ColumnBatcher's header, which the decoder ignores). A null bitmap has bit i of byte i/8 set for a null row. A column type is
 *
 *   i8   int64 LE (Long, Integer, Short, Byte)      f8   float64 LE (Double, Float; a Float is its exact double)
 *   b1   one byte per row                           dt   int64 epoch milliseconds, dtu: epoch microseconds (see micros)
 *   f4v  dim float32 LE per row (float[])           f8v  dim float64 LE per row (double[])
 *   str  int32 LE offsets (n + 1) then UTF-8 bytes  dec  a BigDecimal as text in the str layout
 *   json one JSON array of the column in the tagged encoding of NativeCodec: what no fixed layout carries without changing a value
 *        (nested lists and maps, records and RIDs, BigInteger, a column that mixes kinds, a DATE or a time-zone-aware temporal next to
 *        datetimes with micros, a ragged vector, a Long beyond 2**53 next to a floating value)
 *
 * A frame holds at most {@code max} rows and stops once the rows read add up to about FRAME_BYTES, so the working set of a read is a frame
 * and not the table.
 */
package org.humemai.nativelib;

import com.arcadedb.query.sql.executor.Result;
import com.arcadedb.query.sql.executor.ResultSet;

import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
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
import java.util.Collection;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class ColumnFrames {
  /** The rows of a frame add up to about this many bytes of values. */
  static final long FRAME_BYTES = 1L << 20;
  private static final long TWO53 = 1L << 53;

  private ColumnFrames() {
  }

  /** One frame and whether the rows are over (the cursor was read to its end). */
  static final class Frame {
    final byte[] bytes;
    final boolean done;
    /** The index in {@code head} of the first row this frame did not take (a frame may end before the head does). */
    final int headUsed;

    Frame(final byte[] bytes, final boolean done, final int headUsed) {
      this.bytes = bytes;
      this.done = done;
      this.headUsed = headUsed;
    }
  }

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

  private static long toEpochMicros(final Object v) {
    if (v instanceof LocalDateTime ldt)
      return Math.addExact(Math.multiplyExact(ldt.toEpochSecond(ZoneOffset.UTC), 1_000_000L), ldt.getNano() / 1000);
    if (v instanceof Date d)
      return Math.multiplyExact(d.getTime(), 1000L);
    if (v instanceof Calendar c)
      return Math.multiplyExact(c.getTimeInMillis(), 1000L);
    throw new IllegalArgumentException("not a naive datetime: " + v.getClass());
  }

  private static boolean yearFits(final LocalDateTime ldt) {
    return ldt.getYear() >= 1 && ldt.getYear() <= 9999;
  }

  /** About how many bytes the values of a row take in a frame (a stop rule, not an exact size). */
  private static long estimate(final Result row) {
    long n = 0;
    for (final String name : row.getPropertyNames()) {
      final Object v = row.getProperty(name);
      if (v == null)
        n += 1;
      else if (v instanceof String s)
        n += s.length() + 4;
      else if (v instanceof float[] a)
        n += 4L * a.length;
      else if (v instanceof double[] a)
        n += 8L * a.length;
      else if (v instanceof Number || v instanceof Boolean || v instanceof Date || v instanceof LocalDate || v instanceof LocalDateTime
          || v instanceof Instant)
        n += 8;
      else if (v instanceof Collection<?> c)
        n += 16 + 16L * c.size();
      else if (v instanceof Map<?, ?> m)
        n += 16 + 32L * m.size();
      else
        n += 32;
    }
    return n;
  }

  /**
   * The next frame. {@code head} holds rows that were already read from the cursor when its first reply was encoded as JSON (the
   * frame starts at {@code head[skip]}, then goes on with the cursor), or is null. {@code pinned} is the column set, or null to take
   * the union of the names of the rows of this frame, in order of first appearance. With {@code micros} a naive datetime column is
   * epoch microseconds ("dtu") and a column with a DATE, a time-zone-aware temporal or a year Python cannot hold is "json".
   */
  static Frame next(final ResultSet rs, final List<Result> head, final int skip, final int max, final String[] pinned,
      final boolean micros) {
    final List<Result> results = new ArrayList<>(Math.min(Math.max(max, 1), 1 << 14));
    int h = Math.max(skip, 0);
    long budget = 0;
    boolean done = false;
    while (results.size() < max && budget < FRAME_BYTES) {
      final Result row;
      if (head != null && h < head.size())
        row = head.get(h++);
      else if (rs.hasNext())
        row = rs.next();
      else {
        done = true;
        break;
      }
      results.add(row);
      budget += estimate(row);
    }
    if (!done)
      done = (head == null || h >= head.size()) && !rs.hasNext();

    String[] columns = pinned;
    if (columns == null) {
      final Set<String> names = new LinkedHashSet<>();
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
    final int count = results.size();
    final List<Object[]> rows = new ArrayList<>(count);
    for (final Result row : results) {
      final Object[] vals = new Object[n];
      for (int c = 0; c < n; c++)
        vals[c] = row.getProperty(columns[c]);
      rows.add(vals);
    }

    final StringBuilder hdr = new StringBuilder("{\"count\":").append(count).append(",\"done\":").append(done).append(",\"cols\":[");
    final ByteArrayOutputStream body = new ByteArrayOutputStream(Math.max(count * n * 8, 16));

    for (int c = 0; c < n; c++) {
      final byte[] nulls = new byte[(count + 7) / 8];
      String type = null;
      boolean sawInt = false, sawFloating = false, bigInt = false;
      for (int r = 0; r < count; r++) {
        final Object v = rows.get(r)[c];
        if (v == null)
          continue;
        final String t;
        if (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte) {
          t = "i8";
          sawInt = true;
          final long l = ((Number) v).longValue();
          if (l > TWO53 || l < -TWO53)
            bigInt = true;
        } else if (v instanceof Double || v instanceof Float) {
          t = "f8";
          sawFloating = true;
        } else if (v instanceof Boolean)
          t = "b1";
        else if (v instanceof Date || v instanceof Calendar || v instanceof LocalDateTime || v instanceof LocalDate
            || v instanceof Instant || v instanceof ZonedDateTime || v instanceof OffsetDateTime) {
          if (!micros)
            t = "dt";
          else if (v instanceof LocalDateTime ldt ? yearFits(ldt) : v instanceof Date || v instanceof Calendar)
            t = "dtu";
          else
            t = "json";
        } else if (v instanceof String || v instanceof Character)
          t = "str";
        else if (v instanceof BigDecimal)
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
          if ((type.equals("i8") && t.equals("f8")) || (type.equals("f8") && t.equals("i8")))
            type = "f8";
          else {
            type = "json";
            break;
          }
        }
      }
      if (type == null)
        type = "str"; // an all-null column
      // an integer beyond 2**53 next to a floating value has no exact double: the values stay what they are, in a json column
      if (type.equals("f8") && sawInt && sawFloating && bigInt)
        type = "json";

      int vdim = -1;
      if (type.equals("f4v") || type.equals("f8v")) {
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
        if (vdim < 0 && vdim != -1)
          type = "json"; // ragged
        else if (vdim == -1)
          vdim = 0;
      }

      final byte[] colBuf;
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
        final StringBuilder arr = new StringBuilder();
        arr.append('[');
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (r > 0)
            arr.append(',');
          if (v == null)
            nulls[r >> 3] |= (1 << (r & 7));
          NativeCodec.append(arr, v);
        }
        arr.append(']');
        colBuf = arr.toString().getBytes(StandardCharsets.UTF_8);
      } else { // str and dec: int32 offsets (count + 1) then the UTF-8 bytes
        final ByteArrayOutputStream chars = new ByteArrayOutputStream(Math.max(count * 16, 16));
        final ByteBuffer offs = ByteBuffer.allocate((count + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
        int pos = 0;
        offs.putInt(0);
        for (int r = 0; r < count; r++) {
          final Object v = rows.get(r)[c];
          if (v == null)
            nulls[r >> 3] |= (1 << (r & 7));
          else {
            final String text = v instanceof BigDecimal d ? d.toPlainString() : v.toString();
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

      if (c > 0)
        hdr.append(',');
      hdr.append("{\"name\":");
      NativeCodec.quote(hdr, columns[c]);
      hdr.append(",\"type\":\"").append(type).append("\",\"nulls\":").append(nulls.length).append(",\"bytes\":").append(colBuf.length);
      if (vdim > 0 && (type.equals("f4v") || type.equals("f8v")))
        hdr.append(",\"dim\":").append(vdim);
      hdr.append('}');
      body.write(nulls, 0, nulls.length);
      body.write(colBuf, 0, colBuf.length);
    }
    hdr.append("]}");

    final byte[] headerBytes = hdr.toString().getBytes(StandardCharsets.UTF_8);
    final byte[] bodyBytes = body.toByteArray();
    final ByteBuffer out = ByteBuffer.allocate(4 + headerBytes.length + bodyBytes.length).order(ByteOrder.LITTLE_ENDIAN);
    out.putInt(headerBytes.length);
    out.put(headerBytes);
    out.put(bodyBytes);
    return new Frame(out.array(), done, h);
  }
}
