# EBookStudio Server

WPF 클라이언트는 `spring-server/`의 Spring Boot API로 PDF를 업로드하고 분석 결과를 받습니다.
`python-worker/`는 SQLite 큐의 PDF 분석과 MusicGen 작업을 처리합니다. 내려받은 책은
WPF에서 오프라인으로 읽을 수 있습니다.

## 저장소 구조

```text
EBookStudioServer/
├─ spring-server/       # Spring Boot API와 업무 규칙
├─ python-worker/       # PDF 분석과 MusicGen 작업 처리
├─ compose.yaml         # 소스 빌드용 로컬 구성
├─ compose.deploy.yaml  # GHCR 이미지 운영 배포 구성
├─ scripts/             # 실행 준비 확인·DB 이전·안전 초기화
└─ README.md
```

## 처리 흐름

1. WPF가 Spring에 PDF를 업로드합니다.
2. Spring은 소유권과 파일을 검증하고 PostgreSQL에 책 상태, SQLite에 `analyze` 작업을 기록합니다.
3. Python 분석 워커가 표지와 구조화 JSON을 생성합니다. 분석 완료 처리와 음악 하위 작업 등록은 같은 SQLite 트랜잭션에서 진행합니다.
4. 사용자는 포함된 `default_ambient.wav` 또는 기존 음악으로 즉시 읽을 수 있습니다.
5. MusicGen 워커가 세그먼트별 음악을 생성하고 현재 책 JSON을 원자적으로 갱신합니다.
6. WPF는 `music_job_id`를 추적해 갱신된 JSON과 음악을 다시 내려받습니다.

두 DB는 각자 트랜잭션을 사용합니다. 업로드 중 큐 등록에 실패하면 Spring은 먼저 저장한 책 행과 파일을 정리합니다.
Spring은 작업 상태 조회 또는 내 서재 조회 시 큐의 결과를 PostgreSQL 책 카탈로그에 중복 없이 반영합니다.

작업은 heartbeat, 제한 재시도, 지수 backoff와 결정적 하위 작업 ID를 사용합니다.
오래된 `running` 작업만 복구하므로 정상 워커의 작업을 다시 가져가지 않습니다.
클라이언트가 생성한 `request_id`를 분석 작업 ID로 사용하므로 업로드 응답 유실이나
재전송에도 같은 작업이 중복 등록되지 않습니다. `DELETE /jobs/{jobId}`는 대기 작업을
즉시 `cancelled`로 만들고 실행 중 작업은 `cancel_requested`로 전환합니다. 워커는
분석 단계 사이와 MusicGen 청크 사이에서 이를 확인하며, 워커가 죽은 경우에도 다음
기동 시 취소와 부분 파일 정리를 마무리합니다.

음악 중복 판정은 `music_prompt_cache`에 기록된 최종 프롬프트, 장르, BPM,
키워드, 생성 길이와 `MUSIC_GENERATOR_VERSION`을 SHA-256으로 묶어 수행합니다.
동일 키의 생성은 공용 음악 폴더의 프롬프트별 잠금으로 직렬화되며, DB가 `ready`여도
실제 WAV가 없으면 캐시를 무효화하고 다시 생성합니다. `reuse_count`, 최근 사용 시각,
실패 원인과 생성 작업 ID도 함께 남으므로 중복 방지와 운영 진단에 사용할 수 있습니다.
모델이나 생성 설정을 바꿨다면 `MUSIC_GENERATOR_VERSION`도 변경해야 기존 음악과
새 결과가 잘못 섞이지 않습니다.

WPF 사용량은 오프라인 우선 outbox를 통해 `POST /usage/events`로 배치 전송됩니다.
`usage_events`의 `(user_uuid, event_id)` 복합 키가 재전송을 멱등 처리하며,
`GET /usage/summary`는 앱 활성 시간, 독서 시간, 세션·페이지 이동·책·활동 일수만
집계합니다. 책 원문, 메모, 하이라이트와 페이지별 열람 이력은 수집하지 않습니다.
회원 탈퇴 시 서비스 데이터와 함께 사용량을 삭제하고, SQLite 작업과 격리한 사용자 파일은 바로 정리하거나 실패 시 재시도합니다.

## 데이터베이스 구조

| 저장소 | 주요 데이터 |
|---|---|
| PostgreSQL | 계정·인증, 책, 처리 이력·산출물 메타데이터, 공용 음악·세그먼트 매핑, 사용량 |
| SQLite `jobs.db` | `jobs`, 워커 heartbeat(`worker_nodes`), 음악 프롬프트 캐시(`music_prompt_cache`), 탈퇴 정리 재시도(`account_cleanup_tasks`) |
| 파일 저장소 | PDF, 표지, 책 JSON, WAV. 처리 산출물의 경로·크기·체크섬은 PostgreSQL에 기록합니다. |

PostgreSQL에서 `users`는 `books`, `usage_events`에 연결됩니다. 책은 `processing_runs`로
분석·생성 이력을 보관하고, 각 실행은 `book_artifacts`로 산출물을 연결합니다.
`book_music_bindings`의 `(book_id, segment_key)`가 책의 세그먼트를 공용 `music_assets`에 연결합니다.
스키마 변경은 [Flyway 마이그레이션](spring-server/src/main/resources/db/migration)에 새 버전 SQL을 추가합니다.

`usage_events.book_id`는 클라이언트의 책 폴더 식별자입니다. 서버에서 책을 삭제한 뒤에도
오프라인 사용량을 뒤늦게 받을 수 있도록 이 필드에는 책 FK를 두지 않습니다. 일별 집계의
날짜 경계는 `USAGE_TIMEZONE`(기본 `Asia/Seoul`)을 따릅니다. SQLite Job에 들어가는 사용자·책
식별자는 DB 사이의 물리적 FK 없이 전달되며 Spring이 소유권을 검증합니다.

## 화면과 API 연결

| WPF 화면/동작 | Spring API | 핵심 데이터 |
|---|---|---|
| 로그인·계정 관리 | `/login`, `/refresh`, `/change_password`, `/account` | 사용자, 폐기 토큰, 인증 보조 데이터 |
| 서재 업로드·작업 추적 | `/upload_book`, `/check_status/{jobId}`, `/jobs/{jobId}` | 책 카탈로그, 처리 실행, SQLite Job |
| 내 서재 | `/my_books`, `/delete_server_book` | 사용자 소유 책 카탈로그 |
| 책 상세 | `/api/books/{bookFolder}/processing-history`, `/api/books/{bookFolder}/music-tracks` | 처리 이력·산출물·세그먼트별 음악과 재사용 횟수 |
| 마이페이지 | `/usage/summary`, `/usage/books`, `/usage/daily` | 전체·책별·최근 7일 사용량 집계 |
| 오프라인 활동 동기화 | `POST /usage/events` | 이벤트 UUID 기반 멱등 사용량 저장 |
| 운영 상태 | `/health` | DB 연결, 큐 적체, 워커 heartbeat |

## 요구 사항

- Java 17 이상
- Python 3.11 또는 3.12 권장
- WPF 클라이언트 빌드용 .NET 10 SDK (Windows)
- PostgreSQL 15 이상 또는 Docker
- MusicGen 실행용 GPU 권장

현재 시스템 Python이 3.14라면 별도의 3.12 가상환경을 사용하십시오.

## Python 워커 설치

```powershell
cd python-worker
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-worker.txt
```

CUDA 13.0 전용 PyTorch wheel을 사용할 때는 다음 파일을 사용합니다.

```powershell
pip install -r requirements-worker-cu130.txt
```

최초 MusicGen 실행에는 Hugging Face 모델 다운로드가 필요합니다.

## 로컬 실행

Spring은 PostgreSQL 접속 정보가 필요하고, Spring과 Python Worker는 같은
`EBOOK_QUEUE_DB_PATH`와 `EBOOK_STORAGE_ROOT`를 사용해야 합니다. 기본 queue 파일은
저장소 루트의 `jobs.db`입니다. 전체 구성은 [Docker Compose](#docker-compose)에서 실행할 수 있습니다.

직접 실행할 때는 PostgreSQL을 먼저 준비하고 `SPRING_DATASOURCE_URL`,
`SPRING_DATASOURCE_USERNAME`, `SPRING_DATASOURCE_PASSWORD`를 설정한 뒤 Spring과 Worker를
각각 실행합니다. 실행 위치와 명령은 [Spring 서버 README](spring-server/README.md)와
[Python 워커 README](python-worker/README.md)에 있습니다.

워커 하나만 직접 실행할 때는 역할을 지정할 수 있습니다.

```powershell
python spring_worker.py --role analyze
python spring_worker.py --role music_generation
```

## Docker Compose

Docker Desktop을 시작한 뒤 저장소 루트에서 실행하면 PostgreSQL, Spring API,
PDF 분석 워커, CPU AI 음악 생성 워커가 함께 시작됩니다.

```powershell
.\scripts\docker-up.ps1
```

스크립트는 `.env`가 없으면 예제에서 만들고 JWT 비밀키를 준비한 뒤 API와 워커 준비 상태를 기다립니다.
첫 실행에는 이미지 빌드가 필요하며, 첫 음악 작업은 MusicGen 모델을 내려받을 수 있습니다.
NVIDIA GPU 워커로 시작하려면 `-MusicProfile gpu`, 음악 워커 기동을 생략하려면 `-MusicProfile none`을 사용합니다.

`.env.example`은 `127.0.0.1`에만 바인딩되는 로컬 Compose에서 계정 흐름을 바로
확인할 수 있도록 인증번호를 WPF 클라이언트에 표시합니다. 다른 컴퓨터에서 API에
접속할 수 있게 만들기 전에는 반드시 `EMAIL_EXPOSE_DEVELOPMENT_CODE=false`로 바꾸고
SMTP를 설정하십시오. `/health`의 `email` 값으로 `smtp`, `development-code`,
`disabled` 상태를 확인할 수 있습니다.

```powershell
Copy-Item .env.example .env
docker compose --profile cpu-music up --build -d
docker compose --profile cpu-music ps
Invoke-RestMethod http://127.0.0.1:5000/health
```

음악 워커는 하드웨어에 따라 하나의 프로필만 선택합니다.

```powershell
# CPU 전용 PyTorch. 기능 확인용이며 생성 속도는 느립니다.
docker compose --profile cpu-music up --build -d

# NVIDIA Container Toolkit과 호환 GPU가 있는 환경
docker compose --profile gpu up --build -d
```

`data-init`은 named volume의 소유권을 1회 맞춘 뒤 종료합니다. 이후 API와 Worker는
UID 10001, read-only root filesystem, capability 제거, `no-new-privileges` 상태로
실행됩니다. PostgreSQL은 `postgres-data`, `/data/jobs.db`와 사용자 산출물은 `app-data`,
모델 캐시는 `model-cache`에 유지됩니다. `/health`는 PostgreSQL 연결, queue 적체와 Worker
heartbeat를 함께 보여주며 로그는 컨테이너별 10MB × 3개로 회전합니다.

```powershell
docker compose logs -f api analysis-worker
docker compose down
```

`docker compose down`은 named volume을 보존하지만 `docker compose down -v`는
PostgreSQL, Job, 책과 모델 캐시를 모두 삭제하므로 운영 데이터가 있을 때 사용하면 안 됩니다.

### 백업

PostgreSQL dump와 `app-data` 볼륨을 함께 보관합니다. 아래 명령의 `ebookstudio` DB 이름과
사용자명은 로컬 기본값이므로 변경한 경우 환경설정에 맞춥니다.

```powershell
docker compose stop api analysis-worker music-worker music-worker-gpu
New-Item -ItemType Directory -Force backups | Out-Null
docker compose exec -T postgres pg_dump -U ebookstudio -d ebookstudio -Fc -f /tmp/ebookstudio-postgres.dump
docker compose cp postgres:/tmp/ebookstudio-postgres.dump .\backups\ebookstudio-postgres.dump
docker run --rm -v ebookstudio_app-data:/data:ro -v "${PWD}/backups:/backup" `
  alpine:3.22 sh -c "tar czf /backup/ebookstudio-data.tgz -C /data ."
docker compose start api analysis-worker
```

### Docker와 무관한 DB 안전 초기화

`scripts/reset-databases.py`는 PostgreSQL 서비스 데이터와 SQLite 작업 큐만 비우고
테이블 구조, Flyway 이력, 책·음악 파일은 보존합니다. 기본 실행은 대상과 행 개수만
보여주는 dry-run입니다. API와 Worker를 먼저 중지해야 하며, 최근 120초 안에 heartbeat가
있는 작업·Worker 또는 대상 PostgreSQL에 연결된 다른 클라이언트가 발견되면 실행을
거부합니다.

```powershell
pip install -r .\scripts\requirements-migration.txt
$env:DATABASE_URL='postgresql://ebookstudio:비밀번호@127.0.0.1:5432/ebookstudio'
$env:EBOOK_QUEUE_DB_PATH='C:\ebookstudio\data\jobs.db'

# 1. 삭제될 행과 정확한 DB 대상을 확인합니다. 아무것도 변경하지 않습니다.
python .\scripts\reset-databases.py

# 2. 자동 백업 후 실제 초기화합니다.
python .\scripts\reset-databases.py `
  --backup-dir .\backups\manual-reset `
  --execute --confirm "RESET EBOOKSTUDIO DATABASES"
```

실행 전에 `pg_dump`가 PATH에 있어야 합니다. PostgreSQL dump, SQLite `jobs.db`, 실행 전
건수와 결과가 기록된 `manifest.json`이 모두 생성되지 않으면 초기화하지 않습니다.
실제 비밀번호가 명령 기록에 남지 않도록 PostgreSQL URL은 `DATABASE_URL` 환경변수로
전달하는 것을 권장합니다. 파일 저장소까지 폐기하려면 이 명령과 별개로 백업 후 명시적인
파일 정리가 필요합니다.

초기화를 되돌릴 때는 서비스를 중지하고 `manifest.json`에서 대상 DB와 실행 상태를 확인합니다.
PostgreSQL dump는 `pg_restore`로 복원하고, 백업한 `jobs.db`는 원래 경로에 되돌립니다.

PostgreSQL 서비스 DB와 SQLite Job 큐는 이미 물리적으로 분리되어 있습니다. 현재 SQLite
queue는 단일 Docker 호스트와 로컬 named volume을 전제로 하며 여러 호스트에서 공유하지
않습니다. API·Worker의 다중 호스트 확장, 우선순위나 dead-letter 운영이 필요해지면
RabbitMQ 같은 브로커로 교체하고, 산출물이 단일 호스트 범위를 넘으면 S3 호환 저장소로
옮기는 것이 전환 기준입니다.

SMTP 없이 UI를 로컬 시연할 때만 Spring 실행 전에 아래 값을 설정하면 인증번호가
API 응답으로 전달됩니다. 운영 환경에서는 절대 활성화하지 마십시오.

```powershell
$env:EMAIL_EXPOSE_DEVELOPMENT_CODE='true'
```

## 기존 통합 SQLite 데이터 이전

1. 기존 서버를 종료해 SQLite 쓰기를 끝낸 뒤 `users.db`, `users/`, `defaults/music/`을 별도 백업합니다. 전환을 검증할 때까지 백업과 기존 서버를 보관합니다.
2. 로컬 Compose에서 PostgreSQL과 API를 시작해 Flyway 스키마를 만든 뒤 API를 중지합니다. 로컬 PostgreSQL 포트는 `127.0.0.1`에 바인딩됩니다.
3. 새 PostgreSQL과 큐 전용 `jobs.db`로 데이터를 이전합니다. 이전 도구는 기존 큐 출력 파일의 덮어쓰기를 거부합니다.

```powershell
docker compose up --build -d postgres api
Invoke-RestMethod http://127.0.0.1:5000/health
docker compose stop api analysis-worker music-worker music-worker-gpu

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\scripts\requirements-migration.txt

python .\scripts\migrate-sqlite-to-postgres.py `
  --sqlite C:\backup\ebookstudio\users.db `
  --queue-output C:\backup\ebookstudio\jobs.db `
  --postgres-url "postgresql://ebookstudio:비밀번호@127.0.0.1:5432/ebookstudio"
```

새 설치용 `app-data` 볼륨에 사용자 파일이 없고 초기 `jobs.db`에 보존할 작업이 없는지
확인한 뒤 이전한 큐와 파일을 복사합니다.

```powershell
docker run --rm `
  --mount "type=bind,source=C:\backup\ebookstudio,target=/legacy,readonly" `
  --mount "type=volume,source=ebookstudio_app-data,target=/data" `
  alpine:3.22 sh -c "cp /legacy/jobs.db /data/jobs.db && cp -a /legacy/users /data/users && mkdir -p /data/defaults && cp -a /legacy/defaults/music /data/defaults/music && chown -R 10001:10001 /data"

docker compose up -d api analysis-worker
Invoke-RestMethod http://127.0.0.1:5000/health
docker compose logs --tail 100 api analysis-worker
```

`/health`에서 PostgreSQL, SQLite 큐, 분석 워커를 확인한 뒤 기존 계정으로 로그인합니다.
내 서재, PDF 업로드·분석·오프라인 다운로드, 책 상세와 마이페이지 통계까지 확인합니다.

## GHCR 배포

GitHub Release를 발행하면 Actions가 API 및 워커 이미지를 GHCR에 게시합니다.
배포 전에 `.env.docker.example`을 복사하고 이미지 경로, `POSTGRES_PASSWORD`, `SECRET_KEY`, SMTP를 설정합니다.

```powershell
Copy-Item .env.docker.example .env.docker
# 편집기로 .env.docker 설정을 채운 뒤 실행합니다.
docker login ghcr.io
docker compose --env-file .env.docker -f compose.deploy.yaml pull
docker compose --env-file .env.docker -f compose.deploy.yaml up -d
```

배포용 PostgreSQL은 Compose 내부 네트워크에서 API와 통신합니다. 외부 DB를 쓸 때는
별도 Compose override에서 `SPRING_DATASOURCE_*`를 지정합니다. 외부 API 접속은 TLS 프록시를 사용합니다.

## 이메일과 보안

운영 환경에서는 `.env.example`을 참고해 SMTP를 설정하고
`EMAIL_DELIVERY_ENABLED=true`, `EMAIL_EXPOSE_DEVELOPMENT_CODE=false`를 사용합니다.
`SECRET_KEY`가 없으면 Spring은 저장소에 `.jwt-secret`을 생성하여 재시작 후에도
토큰 서명을 유지합니다. 리프레시 토큰은 사용할 때마다 회전하며 로그아웃 시
액세스 토큰과 함께 폐기됩니다. 인증번호는 원문 대신 서버 비밀키 기반 HMAC으로
저장합니다. 로그인은 IP와 계정, 인증번호 발송은 IP와 이메일 기준으로 제한하며
한도 초과 시 `429 Too Many Requests`와 `Retry-After`를 반환합니다. 한도와 시간
창은 `.env.example`의 `LOGIN_*`, `VERIFICATION_*` 값으로 조정할 수 있습니다.

## WPF 서버 주소

기본 주소는 `http://127.0.0.1:5000`입니다. 다른 서버는 WPF 실행 전에 설정합니다.

```powershell
$env:EBOOK_API_BASE_URL='https://api.example.com'
```

## 테스트

```powershell
cd spring-server
.\mvnw.cmd test

cd ..\python-worker
pip install -r requirements-test.txt
python -m pytest -q

cd ..
python -m unittest discover -s scripts\tests -v
docker compose --env-file .env.example config --quiet
```

Spring 테스트는 회원가입, 로그인, 업로드 큐 등록, 비밀번호 변경, 로그아웃,
계정 복구와 회원 탈퇴뿐 아니라 인증번호 HMAC 저장, 요청 제한, 안전한 파일 삭제,
사용량 배치 멱등성과 계정 삭제 연동을 실제 HTTP로 검증합니다. Python 테스트는 작업 선점,
heartbeat 복구, 재시도, 프롬프트 카탈로그 중복 방지, 파일 유실 감지와
책 JSON 음악 교체를 검증합니다. 서비스 흐름 테스트에는 처리 이력·음악 상세·책별 및 일별 사용량 API의 인증과 소유권 검사도 포함됩니다. `scripts/tests`는 기존 통합 SQLite를 분리할 때 PostgreSQL 소유 테이블이 새 queue DB에 남지 않고 기존 출력 파일을 덮어쓰지 않는지 검증합니다.
