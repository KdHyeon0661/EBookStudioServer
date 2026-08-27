# EBookStudio Spring Server

인증, 사용자·책 영속 데이터, PDF 업로드, Job 상태, 파일 제공과 사용량 집계를 담당합니다.
무거운 PDF 및 음악 처리는 형제 폴더 `../python-worker/`의 Worker가 수행합니다.

## 데이터 저장

- PostgreSQL: `users`, 인증 관련 테이블, `books`, `usage_events`
- SQLite `/data/jobs.db`: `jobs`, `worker_nodes`, `music_prompt_cache`
- 파일 저장소: PDF, 표지, 분석 JSON, 공용 음악

Spring의 기본 `JdbcTemplate`은 PostgreSQL을 사용하고 `queueJdbcTemplate`만 SQLite를
사용합니다. PostgreSQL 스키마는 `src/main/resources/db/migration`의 Flyway migration으로
관리합니다. MyBatis와 JPA는 사용하지 않습니다.

## 실행

로컬에서는 PostgreSQL을 먼저 실행하고 다음 값을 설정합니다.

```powershell
$env:SPRING_DATASOURCE_URL='jdbc:postgresql://localhost:5432/ebookstudio'
$env:SPRING_DATASOURCE_USERNAME='ebookstudio'
$env:SPRING_DATASOURCE_PASSWORD='ebookstudio-local'
$env:EBOOK_QUEUE_DB_PATH='C:\ebookstudio\data\jobs.db'
.\mvnw.cmd spring-boot:run
```

가장 간단한 실행 방법은 상위 폴더에서 `docker compose up --build -d`를 사용하는 것입니다.
기본 포트는 5000입니다.

## 주요 API

- 인증: `/send_code`, `/verify_code`, `/register`, `/login`, `/refresh`, `/logout`
- 계정: `/find_id`, `/reset_password`, `/change_password`, `DELETE /account`
- 도서: `/upload_book`, `/check_status/{jobId}`, `/my_books`, `/delete_server_book`
- 파일: `/list_music_files/...`, `/files/...`
- 사용량: `POST /usage/events`, `GET /usage/summary`

업로드 시 PostgreSQL에 책 처리 상태를 만들고 SQLite에 분석 Job을 등록합니다. 분석 완료
결과는 Job 조회 또는 내 서재 조회 시 PostgreSQL 책 카탈로그에 멱등 반영됩니다.
