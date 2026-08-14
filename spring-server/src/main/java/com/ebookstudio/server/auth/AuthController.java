package com.ebookstudio.server.auth;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ebookstudio.server.config.EBookStudioProperties;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class AuthController {
    private final AuthService authService;
    private final AccountService accountService;
    private final RateLimitService rateLimits;
    private final EBookStudioProperties properties;

    public AuthController(AuthService authService, AccountService accountService,
                          RateLimitService rateLimits, EBookStudioProperties properties) {
        this.authService = authService;
        this.accountService = accountService;
        this.rateLimits = rateLimits;
        this.properties = properties;
    }

    @PostMapping("/send_code")
    public Map<String, Object> sendCode(@Valid @RequestBody EmailRequest request,
                                        HttpServletRequest servletRequest) {
        rateLimits.requireAllowed("verification-ip", servletRequest.getRemoteAddr(),
                properties.verificationIpLimit(), properties.verificationRateWindowSeconds());
        rateLimits.requireAllowed("verification-email", request.email(),
                properties.verificationEmailLimit(), properties.verificationRateWindowSeconds());
        AuthService.CodeDelivery delivery = authService.sendCode(request.email(), request.purpose());
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("message", "If the address can be used, a code has been sent");
        response.put("delivered", delivery.delivered());
        if (delivery.developmentCode() != null) response.put("development_code", delivery.developmentCode());
        return response;
    }

    @PostMapping("/verify_code")
    public Map<String, String> verifyCode(@Valid @RequestBody VerifyRequest request) {
        if (!authService.verifyCode(request.email(), request.code(), request.purpose(), false)) {
            throw new IllegalArgumentException("Invalid or expired code");
        }
        return Map.of("message", "Verified");
    }

    @PostMapping("/register")
    @ResponseStatus(HttpStatus.CREATED)
    public Map<String, String> register(@Valid @RequestBody RegisterRequest request) {
        authService.register(request.username(), request.password(), request.email(), request.code());
        return Map.of("message", "Registered successfully");
    }

    @PostMapping("/login")
    public Map<String, String> login(@Valid @RequestBody LoginRequest request,
                                     HttpServletRequest servletRequest) {
        rateLimits.requireAllowed("login-ip", servletRequest.getRemoteAddr(),
                properties.loginIpLimit(), properties.loginRateWindowSeconds());
        rateLimits.requireAllowed("login-account", request.username(),
                properties.loginAccountLimit(), properties.loginRateWindowSeconds());
        AuthService.LoginResult result = authService.login(request.username(), request.password());
        rateLimits.clear("login-account", request.username());
        return loginResponse(result);
    }

    @PostMapping("/refresh")
    public Map<String, String> refresh(@RequestHeader("Authorization") String authorization) {
        return loginResponse(authService.refresh(bearerToken(authorization)));
    }

    @PostMapping("/logout")
    public Map<String, String> logout(@RequestHeader("Authorization") String authorization,
                                      @RequestBody(required = false) LogoutRequest request) {
        authService.logout(bearerToken(authorization), request == null ? null : request.refreshToken());
        return Map.of("message", "Successfully logged out");
    }

    @PostMapping("/find_id")
    public Map<String, String> findId(@Valid @RequestBody FindIdRequest request) {
        return Map.of("message", "Success", "username",
                authService.findUsername(request.email(), request.code()));
    }

    @PostMapping("/reset_password")
    public Map<String, String> resetPassword(@Valid @RequestBody ResetPasswordRequest request) {
        authService.resetPassword(request.email(), request.code(), request.newPassword());
        return Map.of("message", "Password changed successfully");
    }

    @PostMapping("/change_password")
    public Map<String, String> changePassword(@AuthenticationPrincipal JwtPrincipal principal,
                                               @Valid @RequestBody ChangePasswordRequest request) {
        authService.changePassword(principal, request.currentPassword(), request.newPassword());
        return Map.of("message", "Password changed successfully");
    }

    @DeleteMapping("/account")
    public Map<String, String> deleteAccount(@AuthenticationPrincipal JwtPrincipal principal) {
        accountService.delete(principal);
        return Map.of("message", "Account deleted");
    }

    private static Map<String, String> loginResponse(AuthService.LoginResult result) {
        return Map.of("access_token", result.accessToken(), "refresh_token", result.refreshToken(),
                "username", result.username(), "public_id", result.publicId());
    }

    private static String bearerToken(String authorization) {
        if (authorization == null || !authorization.startsWith("Bearer ")) {
            throw new IllegalArgumentException("Bearer token required");
        }
        return authorization.substring(7);
    }

    public record EmailRequest(@NotBlank @Email String email, String purpose) { }
    public record VerifyRequest(@NotBlank @Email String email, @NotBlank @Size(min = 6, max = 6) String code,
                                String purpose) { }
    public record RegisterRequest(@NotBlank @Size(max = 64) String username,
                                  @Size(min = 8, max = 128) String password,
                                  @NotBlank @Email String email,
                                  @NotBlank @Size(min = 6, max = 6) String code) { }
    public record LoginRequest(@NotBlank String username, @NotBlank String password) { }
    public record FindIdRequest(@NotBlank @Email String email,
                                @NotBlank @Size(min = 6, max = 6) String code) { }
    public record ResetPasswordRequest(@NotBlank @Email String email,
                                       @NotBlank @Size(min = 6, max = 6) String code,
                                       @JsonProperty("new_password") @Size(min = 8, max = 128) String newPassword) { }
    public record ChangePasswordRequest(@JsonProperty("current_password") @NotBlank String currentPassword,
                                        @JsonProperty("new_password") @Size(min = 8, max = 128) String newPassword) { }
    public record LogoutRequest(@JsonProperty("refresh_token") String refreshToken) { }
}