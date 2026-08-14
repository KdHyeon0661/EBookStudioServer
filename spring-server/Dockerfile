FROM maven:3.9.11-eclipse-temurin-17-alpine AS build

WORKDIR /build
COPY pom.xml ./pom.xml
RUN mvn -B -ntp dependency:go-offline
COPY src ./src
RUN mvn -B -ntp -DskipTests package \
    && cp target/ebookstudio-server-1.0.0.jar /build/app.jar

FROM eclipse-temurin:17-jre-alpine

RUN addgroup -S -g 10001 ebookstudio \
    && adduser -S -D -H -u 10001 -G ebookstudio ebookstudio
WORKDIR /app
COPY --from=build --chown=10001:10001 /build/app.jar /app/app.jar

USER 10001:10001
EXPOSE 5000
ENTRYPOINT ["java", "-XX:MaxRAMPercentage=75.0", "-XX:+ExitOnOutOfMemoryError", "-jar", "/app/app.jar"]
