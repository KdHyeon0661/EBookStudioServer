package com.ebookstudio.server.usage;

import com.ebookstudio.server.auth.JwtPrincipal;
import com.ebookstudio.server.config.EBookStudioProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;

@Service
public class UsageService {
    private static final int MAX_BATCH_SIZE = 100;
    private static final long MAX_EVENT_AGE_SECONDS = 366L * 24 * 60 * 60;
    private static final Pattern BOOK_ID = Pattern.compile("[A-Za-z0-9._-]{1,128}");
    private static final Set<String> EVENT_TYPES = Set.of("app_session", "reading_session");

    private final JdbcTemplate jdbc;
    private final TransactionTemplate transactions;
    private final boolean postgresql;
    private final ZoneId usageZone;

    public UsageService(JdbcTemplate jdbc, PlatformTransactionManager transactionManager,
                        EBookStudioProperties properties) {
        this.jdbc = jdbc;
        this.transactions = new TransactionTemplate(transactionManager);
        this.postgresql = Boolean.TRUE.equals(jdbc.execute(
                (org.springframework.jdbc.core.ConnectionCallback<Boolean>) connection ->
                        "PostgreSQL".equalsIgnoreCase(
                                connection.getMetaData().getDatabaseProductName())));
        try {
            this.usageZone = ZoneId.of(properties.usageTimezone());
        } catch (RuntimeException error) {
            throw new IllegalStateException("Invalid ebookstudio.usage-timezone", error);
        }
    }

    public BatchResult ingest(JwtPrincipal principal, List<UsageEventInput> events) {
        if (events == null || events.isEmpty()) throw new IllegalArgumentException("Events are required");
        if (events.size() > MAX_BATCH_SIZE) throw new IllegalArgumentException("At most 100 events are allowed");
        events.forEach(this::validate);

        Integer inserted = transactions.execute(status -> {
            int count = 0;
            long createdAt = Instant.now().getEpochSecond();
            for (UsageEventInput event : events) {
                count += insertUsageEvent(principal, event, createdAt);
            }
            return count;
        });
        return new BatchResult(events.size(), inserted == null ? 0 : inserted);
    }

    private int insertUsageEvent(JwtPrincipal principal, UsageEventInput event, long createdAt) {
        String conflictClause = postgresql
                ? " ON CONFLICT (user_uuid, event_id) DO NOTHING" : "";
        try {
            return jdbc.update("""
                    INSERT INTO usage_events(
                        user_uuid, event_id, event_type, book_id, occurred_at,
                        duration_seconds, page_turns, progress_percent, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """ + conflictClause, principal.publicId(), event.eventId(), event.eventType(),
                    normalizedBookId(event), event.occurredAt(), event.durationSeconds(),
                    event.pageTurns(), event.progressPercent(), createdAt);
        } catch (DuplicateKeyException duplicateEvent) {
            // H2 does not support PostgreSQL's conflict clause. Its failed statement does not
            // poison the surrounding transaction, so the compatibility path remains idempotent.
            if (postgresql) throw duplicateEvent;
            return 0;
        }
    }

    public UsageSummary summary(JwtPrincipal principal) {
        long sevenDaysAgo = Instant.now().minusSeconds(7L * 24 * 60 * 60).getEpochSecond();
        UsageSummary totals = jdbc.query("""
                SELECT
                    COALESCE(SUM(CASE WHEN event_type='app_session' THEN duration_seconds ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type='reading_session' THEN duration_seconds ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type='reading_session' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type='reading_session' THEN page_turns ELSE 0 END), 0),
                    COUNT(DISTINCT CASE WHEN event_type='reading_session' THEN book_id END),
                    COALESCE(SUM(CASE WHEN event_type='app_session' AND occurred_at >= ?
                                      THEN duration_seconds ELSE 0 END), 0),
                    MAX(occurred_at)
                FROM usage_events WHERE user_uuid=?
                """, rs -> {
            rs.next();
            long lastActiveAt = rs.getLong(7);
            Long nullableLastActiveAt = rs.wasNull() ? null : lastActiveAt;
            return new UsageSummary(
                    rs.getLong(1), rs.getLong(2), rs.getLong(3), rs.getLong(4),
                    rs.getLong(5), 0, rs.getLong(6), nullableLastActiveAt);
        }, sevenDaysAgo, principal.publicId());

        Set<LocalDate> activeDates = new HashSet<>();
        jdbc.query("SELECT occurred_at FROM usage_events WHERE user_uuid=?",
                (org.springframework.jdbc.core.RowCallbackHandler) rs -> activeDates.add(
                        Instant.ofEpochSecond(rs.getLong(1)).atZone(usageZone).toLocalDate()),
                principal.publicId());
        return new UsageSummary(totals.totalAppSeconds(), totals.totalReadingSeconds(),
                totals.readingSessionCount(), totals.pageTurnCount(), totals.booksReadCount(),
                activeDates.size(), totals.last7DaysAppSeconds(), totals.lastActiveAt());
    }

    public List<BookUsageSummary> books(JwtPrincipal principal) {
        return jdbc.query("""
                SELECT book_id, COALESCE(SUM(duration_seconds), 0), COUNT(*),
                       COALESCE(SUM(page_turns), 0), MAX(progress_percent), MAX(occurred_at)
                FROM usage_events
                WHERE user_uuid=? AND event_type='reading_session' AND book_id IS NOT NULL
                GROUP BY book_id
                ORDER BY MAX(occurred_at) DESC, book_id
                """, (rs, rowNum) -> new BookUsageSummary(
                rs.getString(1), rs.getLong(2), rs.getLong(3), rs.getLong(4),
                rs.getInt(5), rs.getLong(6)), principal.publicId());
    }

    public DailyUsageSeries daily(JwtPrincipal principal, int days) {
        if (days < 1 || days > 90) {
            throw new IllegalArgumentException("Days must be between 1 and 90");
        }
        LocalDate today = LocalDate.now(usageZone);
        long todayEpochDay = today.toEpochDay();
        long firstEpochDay = todayEpochDay - days + 1;
        long firstEpochSecond = LocalDate.ofEpochDay(firstEpochDay)
                .atStartOfDay(usageZone).toEpochSecond();
        Map<Long, DailyAccumulator> totals = new LinkedHashMap<>();
        for (long epochDay = firstEpochDay; epochDay <= todayEpochDay; epochDay++) {
            totals.put(epochDay, new DailyAccumulator());
        }
        jdbc.query("""
                SELECT event_type, occurred_at, duration_seconds, page_turns
                FROM usage_events
                WHERE user_uuid=? AND occurred_at>=?
                ORDER BY occurred_at
                """, (org.springframework.jdbc.core.RowCallbackHandler) rs -> {
            long epochDay = Instant.ofEpochSecond(rs.getLong("occurred_at"))
                    .atZone(usageZone).toLocalDate().toEpochDay();
            DailyAccumulator day = totals.get(epochDay);
            if (day == null) return;
            if ("app_session".equals(rs.getString("event_type"))) {
                day.appSeconds += rs.getLong("duration_seconds");
            } else if ("reading_session".equals(rs.getString("event_type"))) {
                day.readingSeconds += rs.getLong("duration_seconds");
                day.readingSessionCount++;
                day.pageTurnCount += rs.getLong("page_turns");
            }
        }, principal.publicId(), firstEpochSecond);
        List<DailyUsage> daily = totals.entrySet().stream()
                .map(entry -> new DailyUsage(LocalDate.ofEpochDay(entry.getKey()).toString(),
                        entry.getValue().appSeconds, entry.getValue().readingSeconds,
                        entry.getValue().readingSessionCount, entry.getValue().pageTurnCount))
                .toList();
        return new DailyUsageSeries(days, usageZone.getId(), daily);
    }

    private static final class DailyAccumulator {
        private long appSeconds;
        private long readingSeconds;
        private long readingSessionCount;
        private long pageTurnCount;
    }

    private void validate(UsageEventInput event) {
        if (event == null) throw new IllegalArgumentException("Event is required");
        try {
            if (!UUID.fromString(event.eventId()).toString().equals(event.eventId()))
                throw new IllegalArgumentException("Event ID must be a canonical UUID");
        } catch (RuntimeException error) {
            throw new IllegalArgumentException("Event ID must be a canonical UUID");
        }
        if (!EVENT_TYPES.contains(event.eventType()))
            throw new IllegalArgumentException("Unsupported event type");
        String bookId = normalizedBookId(event);
        if ("reading_session".equals(event.eventType()) && bookId == null)
            throw new IllegalArgumentException("Reading sessions require a book ID");
        if (bookId != null && !BOOK_ID.matcher(bookId).matches())
            throw new IllegalArgumentException("Invalid book ID");
        long now = Instant.now().getEpochSecond();
        if (event.occurredAt() < now - MAX_EVENT_AGE_SECONDS || event.occurredAt() > now + 300)
            throw new IllegalArgumentException("Event timestamp is outside the accepted range");
        if (event.durationSeconds() < 1 || event.durationSeconds() > 24 * 60 * 60)
            throw new IllegalArgumentException("Duration must be between 1 and 86400 seconds");
        if (event.pageTurns() < 0 || event.pageTurns() > 100_000)
            throw new IllegalArgumentException("Invalid page turn count");
        if (event.progressPercent() < 0 || event.progressPercent() > 100)
            throw new IllegalArgumentException("Progress must be between 0 and 100");
        if ("app_session".equals(event.eventType())
                && (bookId != null || event.pageTurns() != 0 || event.progressPercent() != 0))
            throw new IllegalArgumentException("App sessions cannot contain book activity");
    }

    private static String normalizedBookId(UsageEventInput event) {
        if (event.bookId() == null || event.bookId().isBlank()) return null;
        return event.bookId().trim();
    }

    public record BookUsageSummary(
            @JsonProperty("book_id") String bookId,
            @JsonProperty("total_reading_seconds") long totalReadingSeconds,
            @JsonProperty("reading_session_count") long readingSessionCount,
            @JsonProperty("page_turn_count") long pageTurnCount,
            @JsonProperty("highest_progress_percent") int highestProgressPercent,
            @JsonProperty("last_read_at") long lastReadAt) { }

    public record DailyUsageSeries(
            @JsonProperty("window_days") int windowDays,
            @JsonProperty("timezone") String timezone,
            @JsonProperty("daily") List<DailyUsage> daily) { }

    public record DailyUsage(
            @JsonProperty("date") String date,
            @JsonProperty("app_seconds") long appSeconds,
            @JsonProperty("reading_seconds") long readingSeconds,
            @JsonProperty("reading_session_count") long readingSessionCount,
            @JsonProperty("page_turn_count") long pageTurnCount) { }

    public record UsageEventInput(
            @JsonProperty("event_id") String eventId,
            @JsonProperty("event_type") String eventType,
            @JsonProperty("book_id") String bookId,
            @JsonProperty("occurred_at") long occurredAt,
            @JsonProperty("duration_seconds") int durationSeconds,
            @JsonProperty("page_turns") int pageTurns,
            @JsonProperty("progress_percent") int progressPercent) { }

    public record BatchResult(
            @JsonProperty("received_count") int receivedCount,
            @JsonProperty("inserted_count") int insertedCount) { }

    public record UsageSummary(
            @JsonProperty("total_app_seconds") long totalAppSeconds,
            @JsonProperty("total_reading_seconds") long totalReadingSeconds,
            @JsonProperty("reading_session_count") long readingSessionCount,
            @JsonProperty("page_turn_count") long pageTurnCount,
            @JsonProperty("books_read_count") long booksReadCount,
            @JsonProperty("active_day_count") long activeDayCount,
            @JsonProperty("last_7_days_app_seconds") long last7DaysAppSeconds,
            @JsonProperty("last_active_at") Long lastActiveAt) { }
}
