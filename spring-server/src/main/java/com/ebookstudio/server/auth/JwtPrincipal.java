package com.ebookstudio.server.auth;

public record JwtPrincipal(String publicId, String username, String jti) {
}
