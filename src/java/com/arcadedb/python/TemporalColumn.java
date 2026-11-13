/*
 * Python-bindings bridge: a timestamp or date column of Database.insert_columns that crosses as one long[] of epoch ticks.
 *
 * Building a LocalDateTime or LocalDate in Python costs one JPype call per cell (and the list of them one conversion per element
 * again), so a million-row timestamp column was two million calls. Here the whole column is ONE long[] (the integer epoch value of every
 * row, as Arrow holds it), the tick size, and optionally ONE boolean[] marking the null rows. Row i is read when the document of that
 * row is built, so the values are short-lived.
 *
 * A date-time is {@code ticks / ticksPerSecond} whole seconds (rounded toward negative infinity, so a tick before the epoch is the
 * last tick of 1969) and the rest as nanoseconds, read as the UTC wall clock: the engine stores a DATETIME as a UTC wall clock, a
 * zone-aware Arrow timestamp is already UTC epoch time and a zone-less one is taken as the wall clock it shows. The unit's precision is
 * kept: a nanosecond tick is not cut to microseconds. A date is {@code ticks / ticksPerDay} days since the epoch, rounded the same way
 * (a date32 is days, a date64 is milliseconds).
 *
 * Compiled into arcadedb-python-bridge.jar during the wheel build and consumed by DocumentBatcher.insertColumns() and
 * insertColumnsParallel().
 */
package com.arcadedb.python;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

public final class TemporalColumn {
  private static final long NANOS_PER_SECOND = 1_000_000_000L;

  private final long[]    ticks;
  private final boolean[] nulls;
  private final boolean   date;
  /** Ticks in one second (a date-time) or in one day (a date). */
  private final long      ticksPerUnit;
  /** Nanoseconds in one tick, for a date-time. */
  private final long      nanosPerTick;

  private TemporalColumn(final long[] ticks, final boolean[] nulls, final boolean date, final long ticksPerUnit,
      final long nanosPerTick) {
    if (nulls != null && nulls.length != ticks.length)
      throw new IllegalArgumentException("a temporal column has " + nulls.length + " null flags for " + ticks.length + " rows");
    this.ticks = ticks;
    this.nulls = nulls;
    this.date = date;
    this.ticksPerUnit = ticksPerUnit;
    this.nanosPerTick = nanosPerTick;
    checkRange();
  }

  /**
   * A column of date-times.
   *
   * @param ticks          the epoch value of every row, in units of {@code 1 / ticksPerSecond} second since 1970-01-01T00:00:00
   * @param ticksPerSecond 1 (seconds), 1,000, 1,000,000 or 1,000,000,000: any divisor of a second into whole nanoseconds
   * @param nulls          null for a column without a null, else one flag per row (a flagged row reads as null and its tick is ignored)
   * @throws IllegalArgumentException for a tick size that is not a whole number of nanoseconds, or a wrong number of null flags.
   *                                  A row past the years a LocalDateTime holds raises the DateTimeException of LocalDateTime itself,
   *                                  here, before anything is written
   */
  public static TemporalColumn dateTimes(final long[] ticks, final long ticksPerSecond, final boolean[] nulls) {
    if (ticksPerSecond < 1 || NANOS_PER_SECOND % ticksPerSecond != 0)
      throw new IllegalArgumentException(
          "ticks per second must divide 1,000,000,000 into whole nanoseconds, got " + ticksPerSecond);
    return new TemporalColumn(ticks, nulls, false, ticksPerSecond, NANOS_PER_SECOND / ticksPerSecond);
  }

  /**
   * A column of dates.
   *
   * @param ticks       the epoch value of every row, in units of {@code 1 / ticksPerDay} day since 1970-01-01
   * @param ticksPerDay 1 for a date32 (days), 86,400,000 for a date64 (milliseconds)
   * @param nulls       null for a column without a null, else one flag per row (a flagged row reads as null and its tick is ignored)
   * @throws IllegalArgumentException for a tick size below 1, or a wrong number of null flags. A row past the days a LocalDate holds
   *                                  raises the DateTimeException of LocalDate itself, here, before anything is written
   */
  public static TemporalColumn dates(final long[] ticks, final long ticksPerDay, final boolean[] nulls) {
    if (ticksPerDay < 1)
      throw new IllegalArgumentException("ticks per day must be at least 1, got " + ticksPerDay);
    return new TemporalColumn(ticks, nulls, true, ticksPerDay, 0L);
  }

  public int rows() {
    return ticks.length;
  }

  /** The LocalDateTime or LocalDate of row {@code i}, or null for a null row. */
  public Object get(final int i) {
    if (nulls != null && nulls[i])
      return null;
    return valueOf(ticks[i]);
  }

  private Object valueOf(final long tick) {
    if (date)
      return LocalDate.ofEpochDay(Math.floorDiv(tick, ticksPerUnit));
    return LocalDateTime.ofEpochSecond(Math.floorDiv(tick, ticksPerUnit), (int) (Math.floorMod(tick, ticksPerUnit) * nanosPerTick),
        ZoneOffset.UTC);
  }

  /**
   * The value of a row grows with its tick, so the smallest and the largest tick of the non-null rows are the two that can fall outside
   * the years of java.time: a value past them raises here, as one call per cell did in Python, and nothing of the batch is written.
   */
  private void checkRange() {
    long min = Long.MAX_VALUE;
    long max = Long.MIN_VALUE;
    for (int i = 0; i < ticks.length; i++) {
      if (nulls != null && nulls[i])
        continue;
      final long tick = ticks[i];
      if (tick < min)
        min = tick;
      if (tick > max)
        max = tick;
    }
    if (min <= max) {
      valueOf(min);
      valueOf(max);
    }
  }
}
