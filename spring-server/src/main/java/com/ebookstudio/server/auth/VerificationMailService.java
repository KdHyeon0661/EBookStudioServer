package com.ebookstudio.server.auth;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.MailException;
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
        message.setSubject("[EBookStudio] 이메일 인증번호");
        String action = "recovery".equals(purpose) ? "계정 복구" : "회원가입";
        message.setText("EBookStudio " + action + " 인증번호는 " + code + "입니다.\n\n"
                + properties.verificationCodeSeconds() + "초 안에 입력해주세요.\n"
                + "본인이 요청하지 않았다면 이 메일을 무시하세요.");
        try {
            mailSender.send(message);
        } catch (MailException error) {
            throw new IllegalStateException("인증 메일을 보내지 못했습니다. SMTP 설정을 확인해주세요", error);
        }
    }
}
