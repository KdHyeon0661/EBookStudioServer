package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import io.jsonwebtoken.Claims;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;

@Service
public class AuthService {
    private static final int MAX_CODE_ATTEMPTS = 5;
    private static final SecureRandom RANDOM = new SecureRandom();

    private final JdbcTemplate jdbc;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private final EBookStudioProperties properties;
    private final VerificationMailService mailService;

    public AuthService(JdbcTemplate jdbc, PasswordEncoder passwordEncoder, JwtService jwtService,
                       EBookStudioProperties properties, VerificationMailService mailService) {
        this.jdbc = jdbc;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
        this.properties = properties;
        this.mailService = mailService;
    }

    @Transactional
    public CodeDelivery sendCode(String rawEmail, String rawPurpose) {
        String email = normalizeEmail(rawEmail);
        String purpose = normalizePurpose(rawPurpose);
        if (!properties.emailDeliveryEnabled() && !properties.emailExposeDevelopmentCode()) {
            throw new IllegalStateException("Email delivery is not configured");
        }
        boolean exists = emailExists(email);
        if ("register".equals(purpose) && exists) {
            throw new IllegalStateException("Email is already registered");
        }
        if ("recovery".equals(purpose) && !exists) {
            return new CodeDelivery(false, null);
        }

        long now = Instant.now().getEpochSecond();
        List<Long> lastSent = jdbc.query(
                "SELECT last_sent_at FROM verification_codes WHERE email = ? AND purpose = ?",
                (rs, rowNum) -> rs.getLong("last_sent_at"), email, purpose);
        if (!lastSent.isEmpty() && now - lastSent.get(0) < properties.verificationSendCooldownSeconds()) {
            throw new IllegalStateException("Please wait before requesting another code");
        }

        String code = String.valueOf(100000 + RANDOM.nextInt(900000));
        long expiresAt = now + properties.verificationCodeSeconds();
        String codeHash = hashVerificationCode(email, purpose, code);
        int updated = jdbc.update("""
                UPDATE verification_codes
                SET code=?, expires_at=?, purpose=?, last_sent_at=?, failed_attempts=0
                WHERE email=?
                """, codeHash, expiresAt, purpose, now, email);
        if (updated == 0) {
            try {
                jdbc.update("""
                        INSERT INTO verification_codes(
                            email, code, expires_at, purpose, last_sent_at, failed_attempts)
                        VALUES (?, ?, ?, ?, ?, 0)
                        """, email, codeHash, expiresAt, purpose, now);
            } catch (DuplicateKeyException concurrentInsert) {
                jdbc.update("""
                        UPDATE verification_codes
                        SET code=?, expires_at=?, purpose=?, last_sent_at=?, failed_attempts=0
                        WHERE email=?
                        """, codeHash, expiresAt, purpose, now, email);
            }
        }
        mailService.send(email, code, purpose);
        String developmentCode = properties.emailExposeDevelopmentCode() && !properties.emailDeliveryEnabled()
                ? code : null;
        return new CodeDelivery(properties.emailDeliveryEnabled(), developmentCode);
    }

    @Transactional
    public boolean verifyCode(String rawEmail, String code, String rawPurpose, boolean consume) {
        String email = normalizeEmail(rawEmail);
        String purpose = normalizePurpose(rawPurpose);
        if (!properties.emailDeliveryEnabled() && !properties.emailExposeDevelopmentCode()) {
            throw new IllegalStateException("Email delivery is not configured");
        }
        List<CodeRow> rows = jdbc.query("""
                SELECT code, expires_at, failed_attempts FROM verification_codes
                WHERE email = ? AND purpose = ?
                """, (rs, rowNum) -> new CodeRow(rs.getString("code"), rs.getLong("expires_at"),
                rs.getInt("failed_attempts")), email, purpose);
        if (rows.isEmpty()) return false;
        CodeRow row = rows.get(0);
        long now = Instant.now().getEpochSecond();
        if (now > row.expiresAt() || row.failedAttempts() >= MAX_CODE_ATTEMPTS) {
            jdbc.update("DELETE FROM verification_codes WHERE email = ?", email);
            return false;
        }
        boolean valid = constantTimeEquals(row.codeHash(), hashVerificationCode(email, purpose, code));
        if (!valid) {
            int attempts = row.failedAttempts() + 1;
            if (attempts >= MAX_CODE_ATTEMPTS) {
                jdbc.update("DELETE FROM verification_codes WHERE email = ?", email);
            } else {
                jdbc.update("UPDATE verification_codes SET failed_attempts = ? WHERE email = ?", attempts, email);
            }
            return false;
        }
        if (consume) jdbc.update("DELETE FROM verification_codes WHERE email = ?", email);
        return true;
    }

    @Transactional
    public void register(String username, String password, String rawEmail, String code) {
        String email = normalizeEmail(rawEmail);
        if (!verifyCode(email, code, "register", true)) {
            throw new IllegalArgumentException("Invalid or expired code");
        }
        try {
            jdbc.update("INSERT INTO users(public_id, username, email, password_hash) VALUES (?, ?, ?, ?)",
                    UUID.randomUUID().toString(), username.trim(), email, passwordEncoder.encode(password));
        } catch (DuplicateKeyException e) {
            throw new IllegalStateException("Username or email already exists");
        }
    }

    public LoginResult login(String username, String password) {
        return jdbc.query("SELECT public_id, username, password_hash, auth_version FROM users WHERE username = ?", rs -> {
            if (!rs.next()) throw new IllegalArgumentException("Invalid credentials");
            String hash = rs.getString("password_hash");
            boolean matches;
            try {
                matches = hash != null && hash.startsWith("$2") && passwordEncoder.matches(password, hash);
            } catch (RuntimeException ignored) {
                matches = false;
            }
            if (!matches) {
                if (hash != null && !hash.startsWith("$2")) {
                    throw new IllegalArgumentException("Password reset required for this migrated account");
                }
                throw new IllegalArgumentException("Invalid credentials");
            }
            String publicId = rs.getString("public_id");
            String storedUsername = rs.getString("username");
            long authVersion = rs.getLong("auth_version");
            return new LoginResult(jwtService.createAccessToken(publicId, storedUsername, authVersion),
                    jwtService.createRefreshToken(publicId, storedUsername, authVersion), storedUsername, publicId);
        }, username.trim());
    }

    @Transactional
    public LoginResult refresh(String refreshToken) {
        Claims claims = jwtService.parse(refreshToken);
        if (!jwtService.isType(claims, "refresh") || isBlocked(claims.getId())) {
            throw new IllegalArgumentException("Invalid refresh token");
        }
        AccountSession account = jdbc.query("SELECT username, auth_version FROM users WHERE public_id = ?",
                rs -> rs.next() ? new AccountSession(rs.getString("username"), rs.getLong("auth_version")) : null,
                claims.getSubject());
        if (account == null || account.authVersion() != jwtService.authVersion(claims)) {
            throw new IllegalArgumentException("Invalid refresh token");
        }
        block(claims);
        return new LoginResult(
                jwtService.createAccessToken(claims.getSubject(), account.username(), account.authVersion()),
                jwtService.createRefreshToken(claims.getSubject(), account.username(), account.authVersion()),
                account.username(), claims.getSubject());
    }

    @Transactional
    public void logout(String accessToken, String refreshToken) {
        block(jwtService.parse(accessToken));
        if (refreshToken != null && !refreshToken.isBlank()) {
            try { block(jwtService.parse(refreshToken)); } catch (RuntimeException ignored) { }
        }
        cleanupBlocklist();
    }

    public String findUsername(String email, String code) {
        if (!verifyCode(email, code, "recovery", false)) {
            throw new IllegalArgumentException("Invalid or expired code");
        }
        String username = jdbc.query("SELECT username FROM users WHERE email = ?",
                rs -> rs.next() ? rs.getString("username") : null, normalizeEmail(email));
        if (username == null) throw new IllegalArgumentException("Invalid or expired code");
        return username;
    }

    @Transactional
    public void resetPassword(String rawEmail, String code, String newPassword) {
        String email = normalizeEmail(rawEmail);
        if (!verifyCode(email, code, "recovery", true)) {
            throw new IllegalArgumentException("Invalid or expired code");
        }
        int updated = jdbc.update("UPDATE users SET password_hash = ?, auth_version = auth_version + 1 WHERE email = ?",
                passwordEncoder.encode(newPassword), email);
        if (updated == 0) throw new IllegalArgumentException("Invalid or expired code");
    }

    @Transactional
    public void changePassword(JwtPrincipal principal, String currentPassword, String newPassword) {
        String hash = jdbc.query("SELECT password_hash FROM users WHERE public_id = ?",
                rs -> rs.next() ? rs.getString("password_hash") : null, principal.publicId());
        if (hash == null || !passwordEncoder.matches(currentPassword, hash)) {
            throw new IllegalArgumentException("Current password is incorrect");
        }
        jdbc.update("UPDATE users SET password_hash = ?, auth_version = auth_version + 1 WHERE public_id = ?",
                passwordEncoder.encode(newPassword), principal.publicId());
    }

    private boolean emailExists(String email) {
        Integer count = jdbc.queryForObject("SELECT COUNT(*) FROM users WHERE email = ?", Integer.class, email);
        return count != null && count > 0;
    }

    private boolean isBlocked(String jti) {
        cleanupBlocklist();
        Integer count = jdbc.queryForObject("SELECT COUNT(*) FROM token_blocklist WHERE jti = ?", Integer.class, jti);
        return count != null && count > 0;
    }

    private void block(Claims claims) {
        long expiration = claims.getExpiration() == null ? Instant.now().getEpochSecond()
                : claims.getExpiration().toInstant().getEpochSecond();
        try {
            jdbc.update("INSERT INTO token_blocklist(jti, created_at, expires_at) VALUES (?, ?, ?)",
                    claims.getId(), Instant.now().getEpochSecond(), expiration);
        } catch (DuplicateKeyException alreadyBlocked) {
            // Idempotent logout/refresh invalidation.
        }
    }

    private void cleanupBlocklist() {
        jdbc.update("DELETE FROM token_blocklist WHERE expires_at IS NOT NULL AND expires_at < ?",
                Instant.now().getEpochSecond());
    }

    private static String normalizeEmail(String email) {
        return email.trim().toLowerCase(java.util.Locale.ROOT);
    }

    private static String normalizePurpose(String purpose) {
        String normalized = purpose == null ? "register" : purpose.trim().toLowerCase(java.util.Locale.ROOT);
        if (!normalized.equals("register") && !normalized.equals("recovery")) {
            throw new IllegalArgumentException("Invalid verification purpose");
        }
        return normalized;
    }

    private String hashVerificationCode(String email, String purpose, String value) {
        return jwtService.keyedHash("verification-code:" + email + ":" + purpose, value);
    }

    private static boolean constantTimeEquals(String left, String right) {
        return MessageDigest.isEqual(left.getBytes(StandardCharsets.UTF_8), right.getBytes(StandardCharsets.UTF_8));
    }

    private record CodeRow(String codeHash, long expiresAt, int failedAttempts) { }
    private record AccountSession(String username, long authVersion) { }
    public record CodeDelivery(boolean delivered, String developmentCode) { }
    public record LoginResult(String accessToken, String refreshToken, String username, String publicId) { }
}