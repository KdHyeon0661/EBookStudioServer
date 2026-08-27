package com.ebookstudio.server.auth;

import com.ebookstudio.server.common.RateLimitExceededException;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Locale;

@Service
public class RateLimitService {
    private final JdbcTemplate jdbc;
    private final JwtService jwtService;

    public RateLimitService(JdbcTemplate jdbc, JwtService jwtService) {
        this.jdbc = jdbc;
        this.jwtService = jwtService;
    }

    @Transactional
    public void requireAllowed(String scope, String identity, int maximum, long windowSeconds) {
        if (maximum < 1 || windowSeconds < 1) throw new IllegalArgumentException("Invalid rate limit");
        long now = Instant.now().getEpochSecond();
        long expiredBefore = now - windowSeconds;
        String normalized = identity == null ? "unknown" : identity.trim().toLowerCase(Locale.ROOT);
        String key = jwtService.keyedHash("rate-limit:" + scope, normalized);

        int updated = updateWindow(key, scope, now, expiredBefore);
        if (updated == 0) {
            try {
                jdbc.update("""
                        INSERT INTO request_rate_limits(key_hash, scope, window_started_at, request_count)
                        VALUES (?, ?, ?, 1)
                        """, key, scope, now);
            } catch (DuplicateKeyException concurrentInsert) {
                updateWindow(key, scope, now, expiredBefore);
            }
        }

        RateRow row = jdbc.query("""
                SELECT window_started_at, request_count
                FROM request_rate_limits WHERE key_hash=?
                """, result -> result.next()
                ? new RateRow(result.getLong("window_started_at"), result.getInt("request_count"))
                : null, key);
        if (row == null) throw new IllegalStateException("Unable to apply rate limit");
        if (row.requestCount() > maximum) {
            throw new RateLimitExceededException(row.windowStartedAt() + windowSeconds - now);
        }
        jdbc.update("DELETE FROM request_rate_limits WHERE window_started_at < ?", now - 172800);
    }

    private int updateWindow(String key, String scope, long now, long expiredBefore) {
        return jdbc.update("""
                UPDATE request_rate_limits SET
                    scope=?,
                    window_started_at=CASE WHEN window_started_at <= ? THEN ? ELSE window_started_at END,
                    request_count=CASE WHEN window_started_at <= ? THEN 1 ELSE request_count + 1 END
                WHERE key_hash=?
                """, scope, expiredBefore, now, expiredBefore, key);
    }

    public void clear(String scope, String identity) {
        String normalized = identity == null ? "unknown" : identity.trim().toLowerCase(Locale.ROOT);
        String key = jwtService.keyedHash("rate-limit:" + scope, normalized);
        jdbc.update("DELETE FROM request_rate_limits WHERE key_hash = ?", key);
    }

    private record RateRow(long windowStartedAt, int requestCount) { }
}
