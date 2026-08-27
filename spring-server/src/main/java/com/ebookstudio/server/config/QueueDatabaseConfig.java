package com.ebookstudio.server.config;

import org.flywaydb.core.Flyway;
import org.sqlite.SQLiteDataSource;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.jdbc.autoconfigure.DataSourceProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.transaction.PlatformTransactionManager;

import javax.sql.DataSource;
import java.nio.file.Files;
import java.nio.file.Path;

@Configuration
public class QueueDatabaseConfig {
    @Bean(name = "dataSource")
    @Primary
    DataSource dataSource(DataSourceProperties properties) {
        return properties.initializeDataSourceBuilder().build();
    }

    @Bean(name = "jdbcTemplate")
    @Primary
    JdbcTemplate jdbcTemplate(@Qualifier("dataSource") DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }


    @Bean(initMethod = "migrate")
    Flyway flyway(@Qualifier("dataSource") DataSource dataSource) {
        return Flyway.configure().dataSource(dataSource).load();
    }

    @Bean(name = "transactionManager")
    @Primary
    PlatformTransactionManager transactionManager(@Qualifier("dataSource") DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }

    @Bean(name = "queueDataSource")
    DataSource queueDataSource(@Value("${ebookstudio.queue-db-path}") String configuredPath) {
        Path path = Path.of(configuredPath).toAbsolutePath().normalize();
        try {
            if (path.getParent() != null) Files.createDirectories(path.getParent());
        } catch (Exception error) {
            throw new IllegalStateException("Unable to create queue database directory", error);
        }
        SQLiteDataSource dataSource = new SQLiteDataSource();
        dataSource.setUrl("jdbc:sqlite:" + path);
        return dataSource;
    }

    @Bean(name = "queueJdbcTemplate")
    JdbcTemplate queueJdbcTemplate(@Qualifier("queueDataSource") DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }

    @Bean(name = "queueTransactionManager")
    PlatformTransactionManager queueTransactionManager(@Qualifier("queueDataSource") DataSource dataSource) {
        return new DataSourceTransactionManager(dataSource);
    }
}
