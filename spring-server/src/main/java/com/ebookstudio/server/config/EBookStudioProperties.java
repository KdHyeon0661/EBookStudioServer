package com.ebookstudio.server.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ebookstudio")
public record EBookStudioProperties(
        String storageRoot,
        String jwtSecret,
        long accessTokenSeconds,
        long refreshTokenSeconds,
        long verificationCodeSeconds,
        long verificationSendCooldownSeconds,
        int loginIpLimit,
        int loginAccountLimit,
        long loginRateWindowSeconds,
        int verificationIpLimit,
        int verificationEmailLimit,
        long verificationRateWindowSeconds,
        String emailFrom,
        boolean emailDeliveryEnabled,
        boolean emailExposeDevelopmentCode
) {
}
