package com.ebookstudio.server.common;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class HealthController {
    private final JdbcTemplate jdbc;

    public HealthController(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        long now = Instant.now().getEpochSecond();
        Map<String, Boolean> workers = new LinkedHashMap<>();
        workers.put("analyze", workerAlive("analyze", now));
        workers.put("music_generation", workerAlive("music_generation", now));
        Integer queued = jdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE status='queued'", Integer.class);
        Integer running = jdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE status='running'", Integer.class);
        return Map.of(
                "status", "ok",
                "database", "ok",
                "workers", workers,
                "queue", Map.of("queued", queued == null ? 0 : queued, "running", running == null ? 0 : running)
        );
    }

    private boolean workerAlive(String type, long now) {
        Long heartbeat = jdbc.query("SELECT MAX(heartbeat_at) FROM worker_nodes WHERE job_type = ?",
                rs -> rs.next() && rs.getObject(1) != null ? rs.getLong(1) : null, type);
        return heartbeat != null && now - heartbeat <= 90;
    }
}