/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Phase 7: columnar bulk insert through one crossing. The documents are built here from whole columns that Python lays out
 * in native-endian buffers (int64, float64, bool bytes, int32, float32, float32 vector matrix, UTF-8 strings as Arrow offsets plus
 * bytes, optional Arrow null bitmaps), as DocumentBatcher.insertColumns does on the JPype path from long[], double[], boolean[] and
 * Object[] columns. Values are read straight from the caller's buffers; no JSON text, no parser, no intermediate arrays.
 */
package org.humemai.nativelib;

import com.arcadedb.database.Database;
import com.arcadedb.database.MutableDocument;
import com.arcadedb.database.async.DatabaseAsyncExecutor;
import com.arcadedb.database.async.ErrorCallback;
import com.arcadedb.exception.NeedRetryException;
import com.arcadedb.schema.EdgeType;
import com.arcadedb.schema.VertexType;

import org.graalvm.nativeimage.c.type.CTypeConversion;
import org.graalvm.word.Pointer;
import org.graalvm.word.WordFactory;

import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntConsumer;

final class BulkOps {
  private BulkOps() {
  }

  /** Slots per column in the address table: data (address, bytes), aux (address, bytes), nulls (address, bytes). */
  static final int SLOTS = 6;

  /**
   * The original single-buffer form (arcadedb_n_insert_cols): the columns of kind q, d, b or f lie back to back in one buffer, in
   * header order. The address table of the pointer form is derived from the layout, so one loop builds the documents for both.
   */
  @SuppressWarnings("unchecked")
  static long insertColumns(final Database db, final String type, final String header, final Pointer buf, final long bufBytes) {
    final Map<String, Object> h = (Map<String, Object>) Json.parse(header);
    final long n = ((Number) h.get("n")).longValue();
    final List<Object> cols = (List<Object>) h.get("cols");
    final long[] spec = new long[SLOTS * cols.size()];
    long pos = 0;
    for (int c = 0; c < cols.size(); c++) {
      final List<Object> col = (List<Object>) cols.get(c);
      final char kind = ((String) col.get(1)).charAt(0);
      final long dim = ((Number) col.get(2)).longValue();
      final long bytes = switch (kind) {
        case 'q', 'd' -> 8 * n;
        case 'b' -> n;
        case 'f' -> 4 * n * dim;
        default -> throw new IllegalArgumentException("unknown column kind " + kind);
      };
      if (pos + bytes > bufBytes)
        throw new IllegalArgumentException("column '" + col.get(0) + "' ends at byte " + (pos + bytes) + ", the buffer has " + bufBytes);
      spec[SLOTS * c] = buf.rawValue() + pos;
      spec[SLOTS * c + 1] = bytes;
      pos += bytes;
    }
    return insertColumns(db, type, h, spec, null);
  }

  /**
   * The pointer form (arcadedb_n_insert_colp): the table holds SLOTS longs per column, see {@link #insertColumns(Database, String, Map, long[], ErrorCallback)}.
   * Returns the reply text {"ok":true,"n":rows}. A header with "parallel":true must also carry "fid", a failure token that the caller made
   * with the "token" operation of arcadedb_n_async: the records go to the database's async executor, the call does not wait for the writers
   * and neither opens, commits nor rolls back a transaction ("commit" does not apply), exactly as insert_parallel does for rows. The
   * caller waits for the writers and reads (and so removes) the token with the "failures" operation, also when this call throws or never
   * returns to it: the token is the caller's, so this call never drops it.
   */
  @SuppressWarnings("unchecked")
  static String insertColumnsReply(final Database db, final String type, final String header, final long[] spec) {
    final Map<String, Object> h = (Map<String, Object>) Json.parse(header);
    if (!Boolean.TRUE.equals(h.get("parallel")))
      return "{\"ok\":true,\"n\":" + insertColumns(db, type, h, spec, null) + "}";
    if (!(h.get("fid") instanceof Number fid))
      throw new IllegalArgumentException("a parallel columnar insert needs a failure token (header \"fid\", from the \"token\" operation)");
    return "{\"ok\":true,\"n\":" + insertColumns(db, type, h, spec, NativeExtras.FailureToken.of(fid.longValue()).onError) + "}";
  }

  /**
   * header: {"n":rows,"commit":every,"cols":[[name,kind,dim],..]}. Kinds: q (int64), d (float64), b (bool byte), i (int32), e (float32),
   * f (float32 matrix, dim floats per row), s (UTF-8 string: int32 offsets[n+1] in the data slot, the bytes they index in the aux slot, as
   * an Arrow string array; offsets[0] need not be 0). The kinds that the JSON rows carry as tagged values (Json.decodeTag), read as the same
   * objects: T (int64 ticks since 1970-01-01T00:00:00 of the UTC wall clock, dim = ticks per second: 1, 1000, 10^6 or 10^9, read as a
   * LocalDateTime), D (int32 days since the epoch, a LocalDate), n (a decimal as text in the string layout, a BigDecimal) and x (bytes in
   * the string layout, a byte[]), so a column of these is stored as the row path stores it. spec: per column six longs, [data address, data bytes, aux address, aux bytes, null
   * bitmap address, null bitmap bytes]; the bitmap is Arrow's (bit i of byte i/8, least significant first; 1 = present), an address of 0
   * means no nulls. The memory belongs to the caller for the duration of the call; nothing is kept. Every length is checked here before
   * the first document is built, so a wrong table is an exception and nothing is written. Same transaction contract as
   * DocumentBatcher.insertColumns: batch commits and the rollback on failure apply only to a transaction opened here.
   * onError null: the synchronous mode above. Not null (header "parallel":true): every record goes to db.async().createRecord with that
   * callback and no transaction is touched, as DocumentBatcher.insertColumnsParallel does; an edge type is refused there, as it is by
   * insert_parallel. The columns are validated before the first record is handed over in both modes.
   */
  @SuppressWarnings("unchecked")
  private static long insertColumns(final Database db, final String type, final Map<String, Object> h, final long[] spec,
      final ErrorCallback onError) {
    final int commitEvery = ((Number) h.get("commit")).intValue();
    final Table table = Table.of(h, spec);
    final int n = table.n;
    final Object schemaType = db.getSchema().getType(type);
    final boolean vertex = schemaType instanceof VertexType;
    if (onError != null) {
      if (schemaType instanceof EdgeType)
        throw new IllegalArgumentException("Cannot bulk-insert into '" + type
            + "' because it is an edge type; load the vertices, then create the edges (graph_batch().new_edges() or CREATE EDGE)");
      final DatabaseAsyncExecutor async = db.async();
      int handed = 0;
      try {
        for (; handed < n; handed++) {
          final MutableDocument doc = vertex ? db.newVertex(type) : db.newDocument(type);
          setRow(doc, handed, table);
          async.createRecord(doc, null, onError);
        }
      } catch (final Throwable t) {
        // the records handed over before the failure are the writers' now, whatever this call does
        throw new IllegalStateException("after handing " + handed + " of " + n + " rows to the async writers (those may have been stored): " + t, t);
      }
      return n;
    }
    writeChunks(db, n, commitEvery, i -> {
      final MutableDocument doc = vertex ? db.newVertex(type) : db.newDocument(type);
      setRow(doc, i, table);
      doc.save();
    });
    return n;
  }

  /**
   * The retries of a chunk and the back-off before them are the engine's GraphBatch defaults (commitRetries 10, commitRetryDelayMs 1000,
   * doubling up to 10 s), so a load that went through createVertices before keeps the same patience.
   */
  private static final int  COMMIT_RETRIES          = 10;
  private static final long COMMIT_RETRY_DELAY_MS   = 1_000L;
  private static final long MAX_COMMIT_RETRY_DELAY_MS = 10_000L;

  /**
   * Writes rows [0, n) with writeRow (build and save row i), in transactions opened here: one per commitEvery rows (0: one for all). A
   * chunk whose save or commit raises a NeedRetryException (an MVCC conflict with another writer, a lock timeout) is rolled back and
   * written again from its first row after a back-off, up to COMMIT_RETRIES times; after the last attempt the error is thrown unchanged.
   * Any other failure rolls the open chunk back and is thrown; the chunks committed before it stay. A transaction the caller opened is the
   * caller's: the rows join it, nothing is committed, rolled back or retried here.
   */
  private static void writeChunks(final Database db, final int n, final int commitEvery, final IntConsumer writeRow) {
    if (db.isTransactionActive()) {
      for (int i = 0; i < n; i++)
        writeRow.accept(i);
      return;
    }
    for (int from = 0; from < n;) {
      final int to = commitEvery > 0 ? (int) Math.min((long) n, (long) from + commitEvery) : n;
      for (int attempt = 0;;) {
        try {
          db.begin();
          for (int i = from; i < to; i++)
            writeRow.accept(i);
          db.commit();
          break;
        } catch (final NeedRetryException e) {
          if (!rollbackQuietly(db, e) || ++attempt > COMMIT_RETRIES)
            throw e;
          backoffBeforeRetry(attempt);
        } catch (final Throwable e) {
          rollbackQuietly(db, e);
          throw e;
        }
      }
      from = to;
    }
  }

  /** Rolls back the open transaction, if any; false when the rollback itself failed (the failure is added to e as suppressed). */
  private static boolean rollbackQuietly(final Database db, final Throwable e) {
    if (!db.isTransactionActive())
      return true;
    try {
      db.rollback();
      return true;
    } catch (final Throwable rollbackError) {
      e.addSuppressed(rollbackError);
      return false;
    }
  }

  private static void backoffBeforeRetry(final int attempt) {
    long delay = COMMIT_RETRY_DELAY_MS;
    for (int i = 1; i < attempt && delay < MAX_COMMIT_RETRY_DELAY_MS; i++)
      delay = Math.min(delay * 2, MAX_COMMIT_RETRY_DELAY_MS);
    try {
      Thread.sleep(delay);
    } catch (final InterruptedException ie) {
      Thread.currentThread().interrupt();
      throw new NeedRetryException("Interrupted while waiting to retry the commit of a bulk-insert chunk");
    }
  }

  /** Sets the properties of row i on doc from the columns. */
  private static void setRow(final MutableDocument doc, final int i, final Table t) {
    for (int c = 0; c < t.k; c++)
      doc.set(t.names[c], t.cell(c, i));
  }

  /**
   * The rows of a column table as GraphBatch.createVertices takes them: per row {name, value, name, value, ...}. The names are in the
   * iteration order of a java.util.HashMap that took them in column order: the order in which the JSON rows of the same call set their
   * properties (Json.rows reads every row, which has the keys of the first row in the same order, into a fresh HashMap), so a vertex is
   * stored with the property order it has always had. The values are the objects the JSON rows give (Table.cell).
   */
  static Object[][] vertexRows(final Table t) {
    final HashMap<String, Integer> byName = new HashMap<>();
    for (int c = 0; c < t.k; c++)
      byName.put(t.names[c], c);
    if (byName.size() != t.k)
      throw new IllegalArgumentException("two columns have the same name");
    final int[] order = new int[t.k];
    int at = 0;
    for (final Integer c : byName.values())
      order[at++] = c;
    final Object[][] rows = new Object[t.n][];
    for (int i = 0; i < t.n; i++) {
      final Object[] row = new Object[2 * t.k];
      for (int p = 0; p < t.k; p++) {
        row[2 * p] = t.names[order[p]];
        row[2 * p + 1] = t.cell(order[p], i);
      }
      rows[i] = row;
    }
    return rows;
  }

  /**
   * The columns of one call: the header ("n" rows, "cols" [name, kind, dim]) and the address table (SLOTS longs per column), validated
   * before anything is built, and the one place that reads a value out of a buffer ({@link #cell}). The memory belongs to the caller for
   * the duration of the call.
   */
  static final class Table {
    final int n, k;
    final String[] names;
    final char[] kinds;
    final int[] dims;
    final long[] data; // addresses: a Word value cannot live in an array or be compared with null
    final long[] nulls; // 0 = no nulls
    final ByteBuffer[] blobs;
    final byte[] scratch;

    @SuppressWarnings("unchecked")
    static Table of(final Map<String, Object> h, final long[] spec) {
      final long nRows = ((Number) h.get("n")).longValue();
      if (nRows < 0 || nRows > Integer.MAX_VALUE)
        throw new IllegalArgumentException("row count out of range: " + nRows);
      final List<Object> cols = (List<Object>) h.get("cols");
      if (spec.length != SLOTS * cols.size())
        throw new IllegalArgumentException("address table has " + spec.length + " entries for " + cols.size() + " columns, expected " + SLOTS * cols.size());
      return new Table((int) nRows, cols, spec);
    }

    @SuppressWarnings("unchecked")
    private Table(final int n, final List<Object> cols, final long[] spec) {
      this.n = n;
      k = cols.size();
      names = new String[k];
      kinds = new char[k];
      dims = new int[k];
      data = new long[k];
      nulls = new long[k];
      blobs = new ByteBuffer[k];
      int longest = 0;
      for (int c = 0; c < k; c++) {
        final List<Object> col = (List<Object>) cols.get(c);
        names[c] = (String) col.get(0);
        kinds[c] = ((String) col.get(1)).charAt(0);
        dims[c] = ((Number) col.get(2)).intValue();
        final long dataAddr = spec[SLOTS * c], dataBytes = spec[SLOTS * c + 1];
        final long auxAddr = spec[SLOTS * c + 2], auxBytes = spec[SLOTS * c + 3];
        final long nullAddr = spec[SLOTS * c + 4], nullBytes = spec[SLOTS * c + 5];
        final long need = switch (kinds[c]) {
          case 'q', 'd' -> 8L * n;
          case 'b' -> n;
          case 'i', 'e', 'D' -> 4L * n;
          case 'T' -> {
            if (dims[c] != 1 && dims[c] != 1_000 && dims[c] != 1_000_000 && dims[c] != 1_000_000_000)
              throw new IllegalArgumentException("column '" + names[c] + "': a date-time column has 1, 1000, 10^6 or 10^9 ticks per second, not " + dims[c]);
            yield 8L * n;
          }
          case 'f' -> {
            if (dims[c] <= 0)
              throw new IllegalArgumentException("column '" + names[c] + "': a vector column needs a dimension above 0");
            yield 4L * n * dims[c];
          }
          case 's', 'n', 'x' -> 4L * (n + 1L);
          default -> throw new IllegalArgumentException("unknown column kind " + kinds[c]);
        };
        if (dataBytes < need || (need > 0 && dataAddr == 0))
          throw new IllegalArgumentException("column '" + names[c] + "' has " + dataBytes + " bytes, " + need + " needed");
        data[c] = dataAddr;
        if (nullAddr != 0) {
          if (nullBytes < (n + 7L) / 8)
            throw new IllegalArgumentException("column '" + names[c] + "' has a null bitmap of " + nullBytes + " bytes, " + (n + 7L) / 8 + " needed");
          nulls[c] = nullAddr;
        }
        if (kinds[c] == 's' || kinds[c] == 'n' || kinds[c] == 'x') {
          if (auxBytes < 0 || auxBytes > Integer.MAX_VALUE || (auxBytes > 0 && auxAddr == 0))
            throw new IllegalArgumentException("column '" + names[c] + "': the string bytes are missing or beyond 2 GiB");
          blobs[c] = auxBytes == 0 ? ByteBuffer.allocate(0) : CTypeConversion.asByteBuffer(WordFactory.pointer(auxAddr), (int) auxBytes);
          final Pointer offsets = WordFactory.pointer(data[c]);
          int prev = offsets.readInt(WordFactory.signed(0));
          if (prev < 0)
            throw new IllegalArgumentException("column '" + names[c] + "': negative string offset");
          for (int i = 1; i <= n; i++) {
            final int off = offsets.readInt(WordFactory.signed(4L * i));
            if (off < prev || off > auxBytes)
              throw new IllegalArgumentException("column '" + names[c] + "': string offsets are not increasing within the " + auxBytes + " bytes (row " + (i - 1) + ")");
            longest = Math.max(longest, off - prev);
            prev = off;
          }
        }
      }
      scratch = new byte[longest];
    }

    /** The value of column c in row i (null for a null), as the JSON rows give it: Long, Double, Boolean, String, LocalDateTime, LocalDate, BigDecimal, byte[], float[]. */
    Object cell(final int c, final int i) {
      if (nulls[c] != 0 && ((WordFactory.<Pointer>pointer(nulls[c]).readByte(WordFactory.signed(i >> 3)) >> (i & 7)) & 1) == 0)
        return null;
      final Pointer base = WordFactory.pointer(data[c]);
      switch (kinds[c]) {
        case 'q':
          return Long.valueOf(base.readLong(WordFactory.signed(8L * i)));
        case 'd':
          return Double.valueOf(base.readDouble(WordFactory.signed(8L * i)));
        case 'b':
          return Boolean.valueOf(base.readByte(WordFactory.signed(i)) != 0);
        case 'i':
          return Integer.valueOf(base.readInt(WordFactory.signed(4L * i)));
        case 'e':
          return Float.valueOf(base.readFloat(WordFactory.signed(4L * i)));
        case 'T': {
          final long ticks = base.readLong(WordFactory.signed(8L * i));
          final long perSecond = dims[c];
          return LocalDateTime.ofEpochSecond(Math.floorDiv(ticks, perSecond),
              (int) (Math.floorMod(ticks, perSecond) * (1_000_000_000L / perSecond)), ZoneOffset.UTC);
        }
        case 'D':
          return LocalDate.ofEpochDay(base.readInt(WordFactory.signed(4L * i)));
        case 'n': {
          final int from = base.readInt(WordFactory.signed(4L * i));
          final int len = base.readInt(WordFactory.signed(4L * (i + 1L))) - from;
          blobs[c].get(from, scratch, 0, len);
          return new BigDecimal(new String(scratch, 0, len, StandardCharsets.US_ASCII));
        }
        case 'x': {
          final int from = base.readInt(WordFactory.signed(4L * i));
          final byte[] bytes = new byte[base.readInt(WordFactory.signed(4L * (i + 1L))) - from];
          blobs[c].get(from, bytes, 0, bytes.length);
          return bytes;
        }
        case 's': {
          final int from = base.readInt(WordFactory.signed(4L * i));
          final int len = base.readInt(WordFactory.signed(4L * (i + 1L))) - from;
          if (len == 0)
            return "";
          blobs[c].get(from, scratch, 0, len);
          return new String(scratch, 0, len, StandardCharsets.UTF_8);
        }
        default: {
          final int dim = dims[c];
          final float[] v = new float[dim];
          final long at = 4L * i * dim;
          for (int j = 0; j < dim; j++)
            v[j] = base.readFloat(WordFactory.signed(at + 4L * j));
          return v;
        }
      }
    }
  }

  /**
   * Documents (or vertices of a vertex type) from rows [{property: value},..] in the UTF-8 JSON text b[0, n): one record per row through
   * newDocument/newVertex, set per property and save, as DocumentBatcher.insertManyJson does on the JPype path; no transaction is opened
   * or closed here. The rows are parsed one at a time into one reused map (the property order of a fresh HashMap), so no tree of the
   * chunk and no String of the whole text is built.
   */
  static int insertRows(final Database db, final String type, final byte[] b, final int n) {
    final boolean vertex = db.getSchema().getType(type) instanceof VertexType;
    final int[] count = new int[1];
    Json.rows(b, n, row -> {
      final MutableDocument doc = vertex ? db.newVertex(type) : db.newDocument(type);
      for (final Map.Entry<String, Object> e : row.entrySet())
        doc.set(e.getKey(), e.getValue());
      doc.save();
      count[0]++;
    });
    return count[0];
  }
}
