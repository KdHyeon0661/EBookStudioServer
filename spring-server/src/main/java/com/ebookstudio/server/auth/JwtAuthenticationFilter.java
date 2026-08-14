package com.ebookstudio.server.auth;

import io.jsonwebtoken.Claims;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.time.Instant;
import java.util.List;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    private final JwtService jwtService;
    private final JdbcTemplate jdbc;

    public JwtAuthenticationFilter(JwtService jwtService, JdbcTemplate jdbc) {
        this.jwtService = jwtService;
        this.jdbc = jdbc;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        String header = request.getHeader("Authorization");
        if (header != null && header.startsWith("Bearer ")
                && SecurityContextHolder.getContext().getAuthentication() == null) {
            try {
                Claims claims = jwtService.parse(header.substring(7));
                if (jwtService.isType(claims, "access") && !isBlocked(claims.getId())
                        && accountVersionMatches(claims.getSubject(), jwtService.authVersion(claims))) {
                    JwtPrincipal principal = new JwtPrincipal(
                            claims.getSubject(), claims.get("username", String.class), claims.getId());
                    SecurityContextHolder.getContext().setAuthentication(
                            new UsernamePasswordAuthenticationToken(principal, null, List.of()));
                }
            } catch (Exception ignored) {
                SecurityContextHolder.clearContext();
            }
        }
        filterChain.doFilter(request, response);
    }

    private boolean isBlocked(String jti) {
        jdbc.update("DELETE FROM token_blocklist WHERE expires_at IS NOT NULL AND expires_at < ?",
                Instant.now().getEpochSecond());
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM token_blocklist WHERE jti = ?", Integer.class, jti);
        return count != null && count > 0;
    }

    private boolean accountVersionMatches(String publicId, long tokenVersion) {
        Long currentVersion = jdbc.query("SELECT auth_version FROM users WHERE public_id = ?",
                rs -> rs.next() ? rs.getLong("auth_version") : null, publicId);
        return currentVersion != null && currentVersion == tokenVersion;
    }
}