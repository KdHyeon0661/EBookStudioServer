package com.ebookstudio.server.common;

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

    public HealthController(JdbcTemplate appJdbc,
                            @Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc) {
        this.appJdbc = appJdbc;
        this.queueJdbc = queueJdbc;
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
                "workers", workers,
                "queue", Map.of(
                        "backend", "sqlite",
                        "queued", queued == null ? 0 : queued,
                        "running", running == null ? 0 : running)
        );
    }

    private boolean workerAlive(String type, long now) {
        Long heartbeat = queueJdbc.query("SELECT MAX(heartbeat_at) FROM worker_nodes WHERE job_type = ?",
                rs -> rs.next() && rs.getObject(1) != null ? rs.getLong(1) : null, type);
        return heartbeat != null && now - heartbeat <= 90;
    }
}
