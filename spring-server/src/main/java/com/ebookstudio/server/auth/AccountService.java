package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.stream.Stream;

@Service
public class AccountService {
    private static final Logger log = LoggerFactory.getLogger(AccountService.class);

    private final JdbcTemplate appJdbc;
    private final JdbcTemplate queueJdbc;
    private final Path usersRoot;
    private final Path trashRoot;
    private final TransactionTemplate transactions;
    private final TransactionTemplate queueTransactions;

    public AccountService(JdbcTemplate appJdbc,
                          @Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc,
                          EBookStudioProperties properties,
                          @Qualifier("transactionManager") PlatformTransactionManager transactionManager,
                          @Qualifier("queueTransactionManager") PlatformTransactionManager queueTransactionManager) {
        this.appJdbc = appJdbc;
        this.queueJdbc = queueJdbc;
        Path storageRoot = Path.of(properties.storageRoot()).toAbsolutePath().normalize();
        this.usersRoot = storageRoot.resolve("users");
        this.trashRoot = storageRoot.resolve(".trash").resolve("accounts");
        this.transactions = new TransactionTemplate(transactionManager);
        this.queueTransactions = new TransactionTemplate(queueTransactionManager);
    }

    public void delete(JwtPrincipal principal) {
        String email = appJdbc.query("SELECT email FROM users WHERE public_id = ?",
                rs -> rs.next() ? rs.getString("email") : null, principal.publicId());
        if (email == null) throw new IllegalArgumentException("Account not found");

        long now = System.currentTimeMillis() / 1000;
        List<QueuedJob> queuedJobs = queueJdbc.query("""
                SELECT id, available_at FROM jobs
                WHERE user_uuid=? AND status='queued'
                """, (rs, rowNum) -> new QueuedJob(rs.getString("id"),
                rs.getObject("available_at") == null ? null : rs.getLong("available_at")),
                principal.publicId());
        queueJdbc.update("""
                UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?, available_at=NULL
                WHERE user_uuid=? AND status='queued'
                """, now, now, principal.publicId());
        queueJdbc.update("""
                UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                WHERE user_uuid=? AND status='running'
                """, now, principal.publicId());
        Integer activeJobs = queueJdbc.queryForObject("""
                SELECT COUNT(*) FROM jobs
                WHERE user_uuid=? AND status='cancel_requested'
                """, Integer.class, principal.publicId());
        if (activeJobs != null && activeJobs > 0) {
            restoreQueuedJobs(principal.publicId(), queuedJobs);
            throw new IllegalStateException("Background processing is stopping; retry account deletion shortly");
        }

        Path userDirectory = usersRoot.resolve(principal.publicId()).normalize();
        if (!userDirectory.startsWith(usersRoot)) throw new SecurityException("Unsafe account path");

        Path quarantined = null;
        if (Files.exists(userDirectory)) {
            try {
                Files.createDirectories(trashRoot);
                quarantined = trashRoot.resolve(principal.publicId() + "-" + System.currentTimeMillis()
                        + "-" + UUID.randomUUID()).normalize();
                if (!quarantined.startsWith(trashRoot)) throw new SecurityException("Unsafe quarantine path");
                moveDirectory(userDirectory, quarantined);
            } catch (IOException e) {
                restoreQueuedJobs(principal.publicId(), queuedJobs);
                throw new IllegalStateException("Unable to quarantine account files", e);
            }
        }

        Path finalQuarantined = quarantined;
        try {
            queueJdbc.update("""
                    INSERT INTO account_cleanup_tasks(user_uuid, created_at) VALUES (?, ?)
                    ON CONFLICT(user_uuid) DO UPDATE SET created_at=excluded.created_at
                    """, principal.publicId(), now);
        } catch (RuntimeException cleanupRegistrationError) {
            restoreQueuedJobs(principal.publicId(), queuedJobs);
            if (finalQuarantined != null) {
                try {
                    moveDirectory(finalQuarantined, userDirectory);
                } catch (IOException restoreError) {
                    cleanupRegistrationError.addSuppressed(restoreError);
                }
            }
            throw new IllegalStateException("Unable to register account cleanup", cleanupRegistrationError);
        }

        try {
            transactions.executeWithoutResult(status -> {
                appJdbc.update("DELETE FROM usage_events WHERE user_uuid = ?", principal.publicId());
                appJdbc.update("DELETE FROM books WHERE owner_public_id = ?", principal.publicId());
                appJdbc.update("DELETE FROM verification_codes WHERE email = ?", email);
                appJdbc.update("DELETE FROM users WHERE public_id = ?", principal.publicId());
            });
        } catch (RuntimeException databaseError) {
            restoreQueuedJobs(principal.publicId(), queuedJobs);
            try {
                queueJdbc.update("DELETE FROM account_cleanup_tasks WHERE user_uuid=?",
                        principal.publicId());
            } catch (RuntimeException cleanupError) {
                databaseError.addSuppressed(cleanupError);
            }
            if (finalQuarantined != null) {
                try {
                    moveDirectory(finalQuarantined, userDirectory);
                } catch (IOException restoreError) {
                    databaseError.addSuppressed(restoreError);
                }
            }
            throw databaseError;
        }

        try {
            cleanupQueueRows(principal.publicId());
        } catch (RuntimeException cleanupError) {
            // The durable cleanup task remains and AccountCleanupService will retry it.
            log.warn("Queue cleanup deferred for deleted account {}", principal.publicId(), cleanupError);
        }

        if (finalQuarantined != null) {
            try {
                deleteTree(finalQuarantined);
            } catch (IOException cleanupError) {
                log.warn("Account data remains quarantined at {}", finalQuarantined, cleanupError);
            }
        }
    }

    private void cleanupQueueRows(String publicId) {
        queueTransactions.executeWithoutResult(status -> {
            queueJdbc.update("DELETE FROM jobs WHERE user_uuid = ?", publicId);
            queueJdbc.update("DELETE FROM account_cleanup_tasks WHERE user_uuid = ?", publicId);
        });
    }

    private void restoreQueuedJobs(String publicId, List<QueuedJob> queuedJobs) {
        for (QueuedJob job : queuedJobs) {
            queueJdbc.update("""
                    UPDATE jobs SET status='queued', finished_at=NULL, cancel_requested_at=NULL, available_at=?
                    WHERE id=? AND user_uuid=? AND status='cancelled'
                    """, job.availableAt(), job.id(), publicId);
        }
    }

    private static void moveDirectory(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(source, target);
        }
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) return;
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    private record QueuedJob(String id, Long availableAt) { }
}
