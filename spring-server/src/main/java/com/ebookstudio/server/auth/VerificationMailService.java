package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class VerificationMailService {
    private final JavaMailSender mailSender;
    private final EBookStudioProperties properties;

    public VerificationMailService(JavaMailSender mailSender, EBookStudioProperties properties) {
        this.mailSender = mailSender;
        this.properties = properties;
    }

    public void send(String email, String code, String purpose) {
        if (!properties.emailDeliveryEnabled()) return;
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(properties.emailFrom());
        message.setTo(email);
        message.setSubject("EBookStudio verification code");
        String action = "recovery".equals(purpose) ? "account recovery" : "registration";
        message.setText("Your EBookStudio " + action + " code is " + code
                + ". It expires in " + properties.verificationCodeSeconds() + " seconds.");
        mailSender.send(message);
    }
}