package com.ebookstudio.server.auth;

import com.ebookstudio.server.common.RateLimitExceededException;
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
        String normalized = identity == null ? "unknown" : identity.trim().toLowerCase(Locale.ROOT);
        String key = jwtService.keyedHash("rate-limit:" + scope, normalized);
        RateRow row = jdbc.query("""
                INSERT INTO request_rate_limits(key_hash, scope, window_started_at, request_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(key_hash) DO UPDATE SET
                    scope=excluded.scope,
                    window_started_at=CASE
                        WHEN request_rate_limits.window_started_at <= ? THEN excluded.window_started_at
                        ELSE request_rate_limits.window_started_at END,
                    request_count=CASE
                        WHEN request_rate_limits.window_started_at <= ? THEN 1
                        ELSE request_rate_limits.request_count + 1 END
                RETURNING window_started_at, request_count
                """, result -> result.next()
                        ? new RateRow(result.getLong("window_started_at"), result.getInt("request_count"))
                        : null,
                key, scope, now, now - windowSeconds, now - windowSeconds);
        if (row == null) throw new IllegalStateException("Unable to apply rate limit");
        if (row.requestCount() > maximum) {
            throw new RateLimitExceededException(row.windowStartedAt() + windowSeconds - now);
        }
        jdbc.update("DELETE FROM request_rate_limits WHERE window_started_at < ?", now - 172800);
    }

    public void clear(String scope, String identity) {
        String normalized = identity == null ? "unknown" : identity.trim().toLowerCase(Locale.ROOT);
        String key = jwtService.keyedHash("rate-limit:" + scope, normalized);
        jdbc.update("DELETE FROM request_rate_limits WHERE key_hash = ?", key);
    }

    private record RateRow(long windowStartedAt, int requestCount) { }
}