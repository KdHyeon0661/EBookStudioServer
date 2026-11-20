package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;

@Service
public class AccountCleanupService {
    private static final Logger log = LoggerFactory.getLogger(AccountCleanupService.class);

    private final JdbcTemplate appJdbc;
    private final JdbcTemplate queueJdbc;
    private final TransactionTemplate queueTransactions;
    private final Path trashRoot;

    public AccountCleanupService(JdbcTemplate appJdbc,
                                 @Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc,
                                 @Qualifier("queueTransactionManager")
                                 PlatformTransactionManager queueTransactionManager,
                                 EBookStudioProperties properties) {
        this.appJdbc = appJdbc;
        this.queueJdbc = queueJdbc;
        this.queueTransactions = new TransactionTemplate(queueTransactionManager);
        this.trashRoot = Path.of(properties.storageRoot()).toAbsolutePath().normalize()
                .resolve(".trash").resolve("accounts");
    }

    @Scheduled(initialDelayString = "${ebookstudio.account-cleanup-interval-ms:30000}",
            fixedDelayString = "${ebookstudio.account-cleanup-interval-ms:30000}")
    public void cleanup() {
        List<String> pending = queueJdbc.query(
                "SELECT user_uuid FROM account_cleanup_tasks ORDER BY created_at",
                (rs, rowNum) -> rs.getString("user_uuid"));
        for (String publicId : pending) {
            try {
                Integer users = appJdbc.queryForObject(
                        "SELECT COUNT(*) FROM users WHERE public_id=?", Integer.class, publicId);
                if (users != null && users > 0) {
                    // AccountService records the cleanup task before committing the primary
                    // database deletion. Keep it while the user exists so this scheduler cannot
                    // race a deletion that is still in progress.
                    continue;
                }
                queueTransactions.executeWithoutResult(status -> {
                    queueJdbc.update("DELETE FROM jobs WHERE user_uuid=?", publicId);
                    queueJdbc.update("DELETE FROM account_cleanup_tasks WHERE user_uuid=?", publicId);
                });
            } catch (RuntimeException error) {
                log.warn("Unable to complete deferred account cleanup for {}", publicId, error);
            }
        }
        cleanupQuarantinedDirectories();
    }

    private void cleanupQuarantinedDirectories() {
        if (!Files.isDirectory(trashRoot)) return;
        try (Stream<Path> entries = Files.list(trashRoot)) {
            for (Path entry : entries.filter(Files::isDirectory).toList()) {
                String name = entry.getFileName().toString();
                if (name.length() < 36) continue;
                String publicId = name.substring(0, 36);
                try {
                    UUID.fromString(publicId);
                    Integer users = appJdbc.queryForObject(
                            "SELECT COUNT(*) FROM users WHERE public_id=?", Integer.class, publicId);
                    if (users == null || users == 0) deleteTree(entry);
                } catch (IllegalArgumentException ignored) {
                    // Do not touch unknown directories.
                } catch (RuntimeException | IOException error) {
                    log.warn("Unable to clean quarantined account directory {}", entry, error);
                }
            }
        } catch (IOException error) {
            log.warn("Unable to scan account quarantine {}", trashRoot, error);
        }
    }

    private static void deleteTree(Path root) throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }
}
