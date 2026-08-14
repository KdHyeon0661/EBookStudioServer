package com.ebookstudio.server.usage;

import com.ebookstudio.server.auth.JwtPrincipal;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.time.Instant;
import java.util.List;
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

    public UsageService(JdbcTemplate jdbc, PlatformTransactionManager transactionManager) {
        this.jdbc = jdbc;
        this.transactions = new TransactionTemplate(transactionManager);
    }

    public BatchResult ingest(JwtPrincipal principal, List<UsageEventInput> events) {
        if (events == null || events.isEmpty()) throw new IllegalArgumentException("Events are required");
        if (events.size() > MAX_BATCH_SIZE) throw new IllegalArgumentException("At most 100 events are allowed");
        events.forEach(this::validate);

        Integer inserted = transactions.execute(status -> {
            int count = 0;
            long createdAt = Instant.now().getEpochSecond();
            for (UsageEventInput event : events) {
                count += jdbc.update("""
                        INSERT OR IGNORE INTO usage_events(
                            user_uuid, event_id, event_type, book_id, occurred_at,
                            duration_seconds, page_turns, progress_percent, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, principal.publicId(), event.eventId(), event.eventType(),
                        normalizedBookId(event), event.occurredAt(), event.durationSeconds(),
                        event.pageTurns(), event.progressPercent(), createdAt);
            }
            return count;
        });
        return new BatchResult(events.size(), inserted == null ? 0 : inserted);
    }

    public UsageSummary summary(JwtPrincipal principal) {
        long sevenDaysAgo = Instant.now().minusSeconds(7L * 24 * 60 * 60).getEpochSecond();
        return jdbc.query("""
                SELECT
                    COALESCE(SUM(CASE WHEN event_type='app_session' THEN duration_seconds ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type='reading_session' THEN duration_seconds ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type='reading_session' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN event_type='reading_session' THEN page_turns ELSE 0 END), 0),
                    COUNT(DISTINCT CASE WHEN event_type='reading_session' THEN book_id END),
                    COUNT(DISTINCT date(occurred_at, 'unixepoch')),
                    COALESCE(SUM(CASE WHEN event_type='app_session' AND occurred_at >= ?
                                      THEN duration_seconds ELSE 0 END), 0),
                    MAX(occurred_at)
                FROM usage_events WHERE user_uuid=?
                """, rs -> {
            rs.next();
            long lastActiveAt = rs.getLong(8);
            Long nullableLastActiveAt = rs.wasNull() ? null : lastActiveAt;
            return new UsageSummary(
                    rs.getLong(1), rs.getLong(2), rs.getLong(3), rs.getLong(4),
                    rs.getLong(5), rs.getLong(6), rs.getLong(7), nullableLastActiveAt);
        }, sevenDaysAgo, principal.publicId());
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
