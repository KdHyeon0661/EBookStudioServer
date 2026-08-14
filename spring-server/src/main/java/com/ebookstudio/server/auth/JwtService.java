package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Service;

import javax.crypto.Mac;
import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.Date;
import java.util.HexFormat;
import java.util.UUID;

@Service
public class JwtService {
    private final EBookStudioProperties properties;
    private final SecretKey key;

    public JwtService(EBookStudioProperties properties) {
        this.properties = properties;
        this.key = Keys.hmacShaKeyFor(sha256(resolveSecret(properties)));
    }

    public String createAccessToken(String publicId, String username, long authVersion) {
        return createToken(publicId, username, authVersion, "access", properties.accessTokenSeconds());
    }

    public String createRefreshToken(String publicId, String username, long authVersion) {
        return createToken(publicId, username, authVersion, "refresh", properties.refreshTokenSeconds());
    }

    public Claims parse(String token) {
        return Jwts.parser().verifyWith(key).build().parseSignedClaims(token).getPayload();
    }

    public boolean isType(Claims claims, String expected) {
        return expected.equals(claims.get("type", String.class));
    }

    public long authVersion(Claims claims) {
        Object value = claims.get("auth_version");
        return value instanceof Number number ? number.longValue() : -1;
    }

    public String keyedHash(String namespace, String value) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(key);
            byte[] payload = (namespace + "\u0000" + value).getBytes(StandardCharsets.UTF_8);
            return HexFormat.of().formatHex(mac.doFinal(payload));
        } catch (Exception e) {
            throw new IllegalStateException("Unable to calculate keyed hash", e);
        }
    }

    private String createToken(String publicId, String username, long authVersion,
                               String type, long lifetimeSeconds) {
        Instant now = Instant.now();
        return Jwts.builder()
                .subject(publicId)
                .id(UUID.randomUUID().toString())
                .claim("username", username)
                .claim("auth_version", authVersion)
                .claim("type", type)
                .issuedAt(Date.from(now))
                .expiration(Date.from(now.plusSeconds(lifetimeSeconds)))
                .signWith(key)
                .compact();
    }

    private static String resolveSecret(EBookStudioProperties properties) {
        if (properties.jwtSecret() != null && !properties.jwtSecret().isBlank()) {
            return properties.jwtSecret();
        }
        Path secretFile = Path.of(properties.storageRoot()).toAbsolutePath().normalize().resolve(".jwt-secret");
        try {
            Files.createDirectories(secretFile.getParent());
            if (Files.isRegularFile(secretFile)) return Files.readString(secretFile).trim();
            byte[] bytes = new byte[48];
            new SecureRandom().nextBytes(bytes);
            String generated = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
            try {
                Files.writeString(secretFile, generated, StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
                return generated;
            } catch (FileAlreadyExistsException ignored) {
                return Files.readString(secretFile).trim();
            }
        } catch (Exception e) {
            throw new IllegalStateException("SECRET_KEY is not set and a persistent JWT secret could not be created", e);
        }
    }

    private static byte[] sha256(String secret) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(secret.getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            throw new IllegalStateException("Unable to initialize JWT signing key", e);
        }
    }
}