package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import java.util.UUID;
import java.util.stream.Stream;

@Service
public class AccountService {
    private static final Logger log = LoggerFactory.getLogger(AccountService.class);

    private final JdbcTemplate jdbc;
    private final Path usersRoot;
    private final Path trashRoot;
    private final TransactionTemplate transactions;

    public AccountService(JdbcTemplate jdbc, EBookStudioProperties properties,
                          PlatformTransactionManager transactionManager) {
        this.jdbc = jdbc;
        Path storageRoot = Path.of(properties.storageRoot()).toAbsolutePath().normalize();
        this.usersRoot = storageRoot.resolve("users");
        this.trashRoot = storageRoot.resolve(".trash").resolve("accounts");
        this.transactions = new TransactionTemplate(transactionManager);
    }

    public void delete(JwtPrincipal principal) {
        String email = jdbc.query("SELECT email FROM users WHERE public_id = ?",
                rs -> rs.next() ? rs.getString("email") : null, principal.publicId());
        if (email == null) throw new IllegalArgumentException("Account not found");

        long now = System.currentTimeMillis() / 1000;
        jdbc.update("""
                UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?, available_at=NULL
                WHERE user_uuid=? AND status='queued'
                """, now, now, principal.publicId());
        jdbc.update("""
                UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                WHERE user_uuid=? AND status='running'
                """, now, principal.publicId());
        Integer activeJobs = jdbc.queryForObject("""
                SELECT COUNT(*) FROM jobs
                WHERE user_uuid=? AND status='cancel_requested'
                """, Integer.class, principal.publicId());
        if (activeJobs != null && activeJobs > 0) {
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
                throw new IllegalStateException("Unable to quarantine account files", e);
            }
        }

        Path finalQuarantined = quarantined;
        try {
            transactions.executeWithoutResult(status -> {
                jdbc.update("DELETE FROM usage_events WHERE user_uuid = ?", principal.publicId());
                jdbc.update("DELETE FROM jobs WHERE user_uuid = ?", principal.publicId());
                jdbc.update("DELETE FROM verification_codes WHERE email = ?", email);
                jdbc.update("DELETE FROM users WHERE public_id = ?", principal.publicId());
            });
        } catch (RuntimeException databaseError) {
            if (finalQuarantined != null) {
                try {
                    moveDirectory(finalQuarantined, userDirectory);
                } catch (IOException restoreError) {
                    databaseError.addSuppressed(restoreError);
                }
            }
            throw databaseError;
        }

        if (finalQuarantined != null) {
            try {
                deleteTree(finalQuarantined);
            } catch (IOException cleanupError) {
                log.warn("Account data remains quarantined at {}", finalQuarantined, cleanupError);
            }
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
}