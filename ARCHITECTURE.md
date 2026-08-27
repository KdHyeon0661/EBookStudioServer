# EBookStudio 아키텍처

## 목표와 경계

EBookStudio는 온라인에서 PDF를 책 데이터로 변환하고 배경 음악을 준비한 뒤,
다운로드가 끝난 책은 Windows 클라이언트에서 오프라인으로 읽는 서비스입니다.
HTTP API와 업무 규칙은 Spring Boot가 소유하고, Python은 PDF 분석과 MusicGen 추론만
수행합니다. WPF 클라이언트는 서버 장애나 네트워크 단절이 독서 자체를 막지 않도록
완성된 책과 사용자 데이터를 로컬에 보관합니다.

## 구성 요소와 데이터 소유권

| 구성 요소 | 책임 | 상태 저장 |
|---|---|---|
| WPF 클라이언트 | 로그인, 업로드, 작업 추적, 다운로드, 오프라인 독서, 사용량 outbox | `%LOCALAPPDATA%/EBookStudio/DownloadCache` |
| Spring Boot API | 인증·인가, 사용자·책 카탈로그, 업로드, 작업 등록·조회·취소, 파일 제공, 사용량 집계 | PostgreSQL, SQLite Job 큐, 파일 저장소 |
| PostgreSQL | 사용자·인증·책·사용량·처리 이력·산출물·음악 자산 관계의 영속 저장 | `postgres-data` named volume |
| SQLite Job 큐 | 분석·음악 생성 Job, Worker heartbeat, 음악 프롬프트 캐시 | `/data/jobs.db` |
| 분석 Worker | PDF 검증 이후 표지·본문 JSON 생성, 기존 음악 매핑 | SQLite Job 큐와 파일 저장소 |
| MusicGen Worker | 프롬프트 서명 기반 음악 재사용 또는 생성, 책 JSON 원자적 갱신 | SQLite 음악 캐시와 공용 음악 폴더 |
| Docker Compose | 단일 호스트 배포, 볼륨·헬스체크·자원 제한·프로필 관리 | `postgres-data`, `app-data`, `model-cache` |

PostgreSQL과 SQLite의 분리는 기술을 늘리기 위한 구성이 아닙니다. 계정·책·사용량·처리 이력·공용 음악 자산처럼
장기 보존되고 관계 무결성이 필요한 데이터는 PostgreSQL이 소유하고, Python Worker가
직접 선점해야 하는 단일 호스트 작업 상태만 SQLite에 둡니다. Job의 `user_uuid`와
`book_id`는 다른 저장소의 식별자를 전달하는 메시지 필드이므로 외래키를 두지 않고,
Spring이 인증 사용자와 비교해 소유권을 검증합니다. PostgreSQL의 전체 PK·FK 관계와 실제 JOIN은 [DATABASE.md](DATABASE.md)에 정리되어 있습니다.

## 주요 처리 흐름

1. WPF가 UUID `request_id`와 PDF를 Spring의 `/upload_book`에 보냅니다.
2. Spring은 PostgreSQL에 `books` 처리 상태를 기록하고 SQLite에 `analyze` Job을 멱등 등록합니다.
3. 분석 Worker가 Job을 원자적으로 선점하고 heartbeat를 기록하면서 표지와 JSON을 만듭니다.
4. 분석 완료와 결정적 `music_generation` 자식 Job 등록은 SQLite 한 트랜잭션으로 반영됩니다.
5. Spring은 Job 결과를 PostgreSQL 책 카탈로그에 projection하고 WPF에 결과를 제공합니다.
6. WPF는 분석 결과를 원자적으로 로컬 캐시에 반영하므로 즉시 오프라인 독서가 가능합니다.
7. 음악 Worker는 프롬프트 서명을 확인해 기존 WAV를 재사용하거나 새로 생성합니다.
8. 음악 작업이 끝나면 Spring은 음악 자산과 세그먼트 매핑을 PostgreSQL에 projection합니다.
9. WPF는 갱신된 JSON과 필요한 WAV만 다시 동기화하고, 책 상세에서 처리 이력과 음악 재사용 정보를 조회합니다.

PostgreSQL과 SQLite를 가로지르는 분산 트랜잭션은 사용하지 않습니다. 업로드 중 Job 등록이
실패하면 Spring이 책 행과 파일을 보상 삭제하고, 완료 상태는 Job 조회·내 서재 조회 시
멱등 projection합니다. 계정 삭제는 실행 중 Job의 종료를 확인한 뒤 큐 데이터, 서비스
데이터, 사용자 파일 순으로 정리합니다.

## 신뢰성과 중복 방지

- 업로드 `request_id`는 재전송되어도 같은 분석 Job을 반환합니다.
- Job 선점은 SQLite 조건부 갱신으로 한 Worker만 성공합니다.
- 실행 중 Job은 heartbeat, 제한 재시도, 지수 backoff를 사용합니다.
- 오래된 Job만 재기동 시 복구하며 정상 Worker의 작업은 빼앗지 않습니다.
- 취소는 `queued` Job을 즉시 종료하고 `running` Job에는 취소 요청을 전달합니다.
- 책 JSON과 WPF 로컬 JSON·다운로드 파일은 임시 파일 작성 후 교체합니다.
- 음악 프롬프트 서명과 파일 잠금이 같은 조건의 동시 생성을 직렬화합니다.
- 사용량 이벤트는 PostgreSQL의 `(user_uuid, event_id)` 유니크 제약으로 멱등 저장됩니다.
- 서비스 DB 스키마는 Flyway로 버전 관리합니다.

## 보안과 개인정보

- 비밀번호는 BCrypt로 해시하며 API는 짧은 액세스 토큰과 회전형 리프레시 토큰을 사용합니다.
- PostgreSQL 외래키와 `ON DELETE CASCADE`가 사용자·책·사용량의 참조 무결성을 보조합니다.
- 로그아웃된 토큰은 blocklist에 기록되고 보호 API는 사용자 소유권을 다시 확인합니다.
- 인증번호는 서버 비밀키 기반 HMAC으로 저장하며 로그인·인증번호 요청에 제한을 적용합니다.
- 파일 경로는 안전한 단일 경로 조각으로 제한하고 책 JSON이 참조한 음악만 제공합니다.
- 사용량에는 책 원문, 메모, 하이라이트, 페이지별 열람 이력을 넣지 않습니다.
- 운영에서는 TLS 종단 프록시 뒤에 API를 두고 개발용 인증번호 노출을 비활성화해야 합니다.

## 배포 프로필과 확장 기준

- 기본: PostgreSQL + Spring API + PDF 분석 Worker
- `cpu-music`: CPU MusicGen Worker를 추가하는 기능 확인용 구성
- `gpu`: NVIDIA GPU MusicGen Worker를 추가하는 실제 생성용 구성

컨테이너는 가능한 범위에서 UID 10001, read-only root filesystem, capability 제거,
`no-new-privileges`와 로그 회전을 사용합니다. PostgreSQL은 전용 `postgres-data`, SQLite
Job 큐와 산출물은 `app-data`, 모델 캐시는 `model-cache`에 보존합니다.

SQLite Job 큐는 개인 프로젝트와 단일 Docker 호스트 운영을 위한 의도적인 선택입니다.
API·Worker를 여러 호스트로 확장하거나 우선순위·dead-letter·독립 큐 운영이 필요해지면
RabbitMQ 같은 브로커로 교체합니다. 산출물이 단일 호스트 백업 범위를 넘으면 S3 호환
스토리지로 옮깁니다. PostgreSQL 서비스 데이터는 이 확장과 무관하게 유지할 수 있습니다.

## 검증 명령

```powershell
cd spring-server
.\mvnw.cmd clean test

cd ..\python-worker
pip install -r requirements-test.txt
python -m pytest -q

cd ..
docker compose --env-file .env.example config --quiet
```

실제 컨테이너 빌드와 헬스체크는 Docker 엔진이 실행 중인 환경에서
`docker compose up --build -d`와 `docker compose ps`로 확인합니다.
