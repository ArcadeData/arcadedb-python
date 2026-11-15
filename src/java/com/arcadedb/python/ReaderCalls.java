/*
 * Python-bindings bridge: the readers of records and rows in one crossing.
 *
 * A JPype call costs a few microseconds of dispatch, and so does every read of a Java object that comes back
 * (a String, an array element, a set element). A record of three properties took 18 of them to become a dict
 * (getPropertyNames, one hasNext/next pair per name, one get per property, one conversion per name and value), a row
 * from a result set 14, a call for the property names 12, and each row of a result set two (hasNext, next).
 *
 * This writes a record or a row as the typed JSON of TypedRows (see there for the tags: the values JSON cannot carry
 * exactly are tagged, the engine's own objects such as RIDs, embedded documents and float[] go to a side array that
 * Python reads only when the JSON refers to it), so Python makes one call, reads one String, and parses it with the C
 * json module. Values keep the types convert_java_to_python gives them: the typed JSON is the transport
 * ResultSet.to_list() already uses, and a test compares every value type with the per-property path.
 *
 * Compiled into arcadedb-python-bridge.jar during the wheel build (scripts/Dockerfile.build and
 * scripts/build-native.sh) and consumed by Result.to_dict(), Document.to_dict(), the property-name readers of both,
 * and ResultSet.__next__() in the Python bindings.
 */
package com.arcadedb.python;

import com.arcadedb.database.Document;
import com.arcadedb.database.RID;
import com.arcadedb.query.sql.executor.Result;
import com.arcadedb.query.sql.executor.ResultSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public final class ReaderCalls {

  private ReaderCalls() {
  }

  /**
   * The next row of a result set, or null when it has no more: the {@code hasNext()} and {@code next()} of one Python
   * {@code __next__} in one crossing. A row is still read when it is asked for (no batch is taken ahead), and an error
   * in the statement surfaces from here as it did from either call.
   */
  public static Result nextOrNull(final ResultSet rs) {
    return rs.hasNext() ? rs.next() : null;
  }

  /**
   * One row as {@code {String json, Object[] side}}: {@code json} is a row object in the typed JSON of
   * {@link TypedRows}, {@code side} holds the engine objects the JSON refers to by index. The properties are in the
   * order {@code getPropertyNames()} gives them.
   */
  public static Object[] rowJson(final Result row) {
    final StringBuilder sb = new StringBuilder(128);
    final List<Object> side = new ArrayList<>();
    TypedRows.appendRow(sb, row, side);
    return new Object[] { sb.toString(), side.toArray() };
  }

  /**
   * One record as {@code {String json, Object[] side}}, like {@link #rowJson(Result)}: the properties are the ones
   * {@code getPropertyNames()} lists, each value is {@code get(name)}.
   */
  public static Object[] documentJson(final Document doc) {
    final String[] names = doc.getPropertyNames().toArray(new String[0]);
    final StringBuilder sb = new StringBuilder(128);
    final List<Object> side = new ArrayList<>();
    for (final String name : names)
      if (TypedRows.TAG.equals(name)) {
        // a property named like the tag: hand the names and values over as they are, Python pairs them
        final Object[] values = new Object[names.length];
        for (int i = 0; i < names.length; i++)
          values[i] = doc.get(names[i]);
        sb.append("{\"\\u0001\":[\"r\",0]}");
        side.add(new Object[] { names, values });
        return new Object[] { sb.toString(), side.toArray() };
      }
    sb.append('{');
    for (int i = 0; i < names.length; i++) {
      if (i > 0)
        sb.append(',');
      TypedRows.quote(sb, names[i]);
      sb.append(':');
      TypedRows.append(sb, doc.get(names[i]), side);
    }
    sb.append('}');
    return new Object[] { sb.toString(), side.toArray() };
  }

  /**
   * The record id of a row as text, or null when the row is not a record: the {@code getIdentity()}, {@code isPresent()},
   * {@code get()} and {@code toString()} of {@code Result.get_rid()} in one crossing.
   */
  public static String ridOf(final Result row) {
    return row.getIdentity().map(Object::toString).orElse(null);
  }

  /** The record id of a record as text, or null when it has none yet: the {@code getIdentity()} and the {@code str()} of {@code Document.get_rid()}. */
  public static String documentRid(final Document doc) {
    final RID rid = doc.getIdentity();
    return rid == null ? null : rid.toString();
  }

  /** The vertex of a row, or null when the row is not a vertex: {@code getVertex()}, {@code isPresent()} and {@code get()} in one crossing. */
  public static Object vertexOrNull(final Result row) {
    return row.getVertex().orElse(null);
  }

  /** The edge of a row, or null when the row is not an edge: {@code getEdge()}, {@code isPresent()} and {@code get()} in one crossing. */
  public static Object edgeOrNull(final Result row) {
    return row.getEdge().orElse(null);
  }

  /** The record of a row, or null when the row is not a record: {@code getElement()}, {@code isPresent()} and {@code get()} in one crossing. */
  public static Object elementOrNull(final Result row) {
    return row.getElement().orElse(null);
  }

  /** The property names of a row as one JSON array of strings, in the order {@code getPropertyNames()} gives them. */
  public static String rowNames(final Result row) {
    return namesJson(row.getPropertyNames());
  }

  /** The property names of a record as one JSON array of strings, in the order {@code getPropertyNames()} gives them. */
  public static String documentNames(final Document doc) {
    return namesJson(doc.getPropertyNames());
  }

  private static String namesJson(final Set<String> names) {
    final StringBuilder sb = new StringBuilder(16 * names.size() + 2);
    sb.append('[');
    boolean first = true;
    for (final String name : names) {
      if (!first)
        sb.append(',');
      first = false;
      TypedRows.quote(sb, name);
    }
    return sb.append(']').toString();
  }
}
