package com.ebookstudio.server.common;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class HealthController {
    private final JdbcTemplate appJdbc;
    private final JdbcTemplate queueJdbc;
    private final EBookStudioProperties properties;

    public HealthController(JdbcTemplate appJdbc,
                            @Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc,
                            EBookStudioProperties properties) {
        this.appJdbc = appJdbc;
        this.queueJdbc = queueJdbc;
        this.properties = properties;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        appJdbc.queryForObject("SELECT 1", Integer.class);
        long now = Instant.now().getEpochSecond();
        Map<String, Boolean> workers = new LinkedHashMap<>();
        workers.put("analyze", workerAlive("analyze", now));
        workers.put("music_generation", workerAlive("music_generation", now));
        Integer queued = queueJdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE status='queued'", Integer.class);
        Integer running = queueJdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE status='running'", Integer.class);
        return Map.of(
                "status", "ok",
                "database", "ok",
                "persistence", "postgresql",
                "email", emailMode(),
                "workers", workers,
                "queue", Map.of(
                        "backend", "sqlite",
                        "queued", queued == null ? 0 : queued,
                        "running", running == null ? 0 : running)
        );
    }

    private String emailMode() {
        if (properties.emailDeliveryEnabled()) return "smtp";
        if (properties.emailExposeDevelopmentCode()) return "development-code";
        return "disabled";
    }

    private boolean workerAlive(String type, long now) {
        Long heartbeat = queueJdbc.queryForObject("""
                SELECT MAX(heartbeat_at) FROM (
                    SELECT MAX(heartbeat_at) AS heartbeat_at FROM worker_nodes WHERE job_type = ?
                    UNION ALL
                    SELECT MAX(heartbeat_at) FROM jobs
                    WHERE type = ? AND status IN ('running', 'cancel_requested')
                )
                """, Long.class, type, type);
        return heartbeat != null && now - heartbeat <= 90;
    }
}
