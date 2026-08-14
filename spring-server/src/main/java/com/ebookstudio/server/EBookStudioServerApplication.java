package com.ebookstudio.server;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.security.autoconfigure.UserDetailsServiceAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

@SpringBootApplication(exclude = UserDetailsServiceAutoConfiguration.class)
@EnableConfigurationProperties(EBookStudioProperties.class)
public class EBookStudioServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(EBookStudioServerApplication.class, args);
    }
}
