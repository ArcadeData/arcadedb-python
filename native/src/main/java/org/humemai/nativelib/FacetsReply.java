/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The facets of a result set (ResultSet.getFacets(), columnar faceting, engine 26.12.1) as the JSON object that
 * ResultSet.facets returns in Python:
 *
 *   {"meta":{"exact":b,"patched":n,"lag_ms":n},"<alias>":{"type":..,"source":..,"reason":..,"buckets":[..],"other_count":n,"missing":n}}
 *
 * Every value of the data (a term, a bound, the min and max of a STATS facet, a cursor) is written as NativeCodec writes a row's value,
 * so NaN and the infinities, dates and decimals reach Python as they do in a row (the engine's own JSON writers turn NaN into null).
 *
 * The engine classes are read by reflection, never named at compile time: the repository also builds against engines before 26.12.1
 * (the pull request gate links 26.11.1), which have neither ResultSet.getFacets() nor com.arcadedb.query.facet. On such an engine
 * GET_FACETS is null and no statement has facets. The classes read here are listed in reachability-metadata.json.
 */
package org.humemai.nativelib;

import com.arcadedb.query.sql.executor.ResultSet;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

final class FacetsReply {
  private static final String UNAVAILABLE = "unavailable";

  /** ResultSet.getFacets(), null on an engine that has none. */
  private static final Method GET_FACETS = find(ResultSet.class, "getFacets");
  /** FacetResults.get(String), null on an engine that has none. */
  private static final Method GET_FACET = find("com.arcadedb.query.facet.FacetResults", "get", String.class);

  /** The public no-argument methods read so far, by class and name. */
  private static final ClassValue<Map<String, Method>> METHODS = new ClassValue<>() {
    @Override
    protected Map<String, Method> computeValue(final Class<?> type) {
      return new ConcurrentHashMap<>();
    }
  };

  private FacetsReply() {
  }

  private static Method find(final String type, final String name, final Class<?>... parameters) {
    try {
      return find(Class.forName(type), name, parameters);
    } catch (final ReflectiveOperationException | LinkageError e) {
      return null;
    }
  }

  private static Method find(final Class<?> type, final String name, final Class<?>... parameters) {
    try {
      return type.getMethod(name, parameters);
    } catch (final ReflectiveOperationException | LinkageError e) {
      return null;
    }
  }

  /** The cursor whose getFacets() answers: a Prefetched (the lazy cursors) holds the engine's result set as its tail. */
  private static ResultSet engineSet(final ResultSet rs) {
    return rs instanceof Prefetched prefetched ? prefetched.tail() : rs;
  }

  /**
   * Appends the facets of rs and returns true; returns false, appending nothing, for a statement that has none. Computing them reads
   * the matches the rows did not reach and ends the rows (read the rows first, as the engine documents).
   */
  static boolean append(final StringBuilder sb, final ResultSet rs) throws Throwable {
    if (GET_FACETS == null || GET_FACET == null)
      return false;
    final Object facets;
    try {
      facets = GET_FACETS.invoke(engineSet(rs));
    } catch (final InvocationTargetException e) {
      throw e.getCause();
    }
    if (facets == null)
      return false;
    appendFacets(sb, facets);
    return true;
  }

  /** The public no-argument method {@code name} of the object, called. A failure of the method itself is rethrown as it is. */
  private static Object call(final Object target, final String name) throws Throwable {
    final Map<String, Method> methods = METHODS.get(target.getClass());
    Method method = methods.get(name);
    if (method == null) {
      method = target.getClass().getMethod(name);
      methods.put(name, method);
    }
    try {
      return method.invoke(target);
    } catch (final InvocationTargetException e) {
      throw e.getCause();
    }
  }

  private static void appendFacets(final StringBuilder sb, final Object facets) throws Throwable {
    sb.append("{\"meta\":{\"exact\":").append(call(facets, "exact")).append(",\"patched\":").append(call(facets, "patched"))
        .append(",\"lag_ms\":").append(call(facets, "lagMs")).append('}');
    for (final Object alias : (Set<?>) call(facets, "aliases")) {
      sb.append(',');
      NativeCodec.quote(sb, (String) alias);
      sb.append(':');
      final Object result;
      try {
        result = GET_FACET.invoke(facets, alias);
      } catch (final InvocationTargetException e) {
        throw e.getCause();
      }
      appendFacet(sb, result);
    }
    sb.append('}');
  }

  private static void appendFacet(final StringBuilder sb, final Object result) throws Throwable {
    final String kind = (String) call(result, "kind");
    final String source = (String) call(result, "source");
    final String reason = (String) call(result, "reason");
    sb.append("{\"type\":");
    NativeCodec.quote(sb, kind);
    sb.append(",\"source\":");
    NativeCodec.quote(sb, source);
    if (reason != null) {
      sb.append(",\"reason\":");
      NativeCodec.quote(sb, reason);
    }
    // an unavailable facet has no buckets, no counts and no cursor, only its reason
    if (UNAVAILABLE.equals(source)) {
      sb.append('}');
      return;
    }
    final Object stats = call(result, "stats");
    if (stats != null) {
      sb.append(",\"count\":").append(call(stats, "count")).append(",\"missing\":").append(call(stats, "missing"));
      appendMember(sb, "min", call(stats, "min"));
      appendMember(sb, "max", call(stats, "max"));
      appendMember(sb, "sum", call(stats, "sum"));
      appendMember(sb, "avg", call(stats, "avg"));
    } else {
      final Object distinct = call(result, "distinct");
      if (distinct != null)
        sb.append(",\"value\":").append(call(distinct, "value")).append(",\"exact\":").append(call(distinct, "exact"));
      else {
        sb.append(",\"buckets\":[");
        boolean first = true;
        for (final Object bucket : (List<?>) call(result, "buckets")) {
          if (!first)
            sb.append(',');
          first = false;
          appendBucket(sb, bucket);
        }
        sb.append(']');
        // the counts of the values not returned: a terms or hierarchy facet has them, a range or histogram has every interval
        if (!"range".equals(kind) && !"histogram".equals(kind))
          sb.append(",\"other_count\":").append(call(result, "otherCount"));
      }
      sb.append(",\"missing\":").append(call(result, "missing"));
    }
    if ((Boolean) call(result, "hasAfter"))
      appendMember(sb, "after", call(result, "after"));
    sb.append('}');
  }

  private static void appendBucket(final StringBuilder sb, final Object bucket) throws Throwable {
    final String key = (String) call(bucket, "key");
    sb.append('{');
    if (key != null) {
      sb.append("\"key\":");
      NativeCodec.quote(sb, key);
      if ((Boolean) call(bucket, "isDefault"))
        sb.append(",\"default\":true");
      else {
        final Object from = call(bucket, "from");
        if (from != null)
          appendMember(sb, "from", from);
        final Object to = call(bucket, "to");
        if (to != null)
          appendMember(sb, "to", to);
      }
    } else if ((Boolean) call(bucket, "isMissing"))
      // the records without a value
      sb.append("\"value\":null,\"missing\":true");
    else {
      sb.append("\"value\":");
      NativeCodec.append(sb, call(bucket, "value"));
    }
    sb.append(",\"count\":").append(call(bucket, "count"));
    // the extra members of a hierarchy bucket (parent, depth, orphan): the text of a JSON object, written without its braces
    final String attributes = String.valueOf(call(bucket, "attributes"));
    if (attributes.length() > 2 && attributes.charAt(0) == '{')
      sb.append(',').append(attributes, 1, attributes.length() - 1);
    sb.append('}');
  }

  private static void appendMember(final StringBuilder sb, final String name, final Object value) {
    sb.append(",\"").append(name).append("\":");
    NativeCodec.append(sb, value);
  }
}
