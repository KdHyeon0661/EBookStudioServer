# 기존 서버 교체와 Docker 배포

이 저장소에서는 Spring Boot가 기존 Flask HTTP 서버를 완전히 대체합니다. Python은
`python-worker/`에서 PDF 분석과 MusicGen 작업만 수행하며 외부 포트를 열지 않습니다.
서비스 데이터는 PostgreSQL, 단일 호스트 AI Job은 `/data/jobs.db`에 저장합니다.

## 1. 새 설치

Docker Desktop을 시작한 뒤 저장소 루트에서 실행합니다.

```powershell
.\scripts\docker-up.ps1
```

스크립트는 실행 위치와 무관하게 저장소의 Compose 파일을 선택하고, `.env.example`을 복사해 JWT 비밀키를 생성한 뒤 최대 5분 동안 컨테이너 healthcheck를 기다립니다. 성공 메시지가 출력되면 PostgreSQL·Spring·분석 Worker가 모두 준비된 상태입니다. `.env`의
`POSTGRES_PASSWORD`는 외부 배포 전에 반드시 별도 강력한 값으로 바꾸십시오.

```powershell
# CPU MusicGen 포함
.\scripts\docker-up.ps1 -MusicProfile cpu-music

# NVIDIA GPU MusicGen 포함
.\scripts\docker-up.ps1 -MusicProfile gpu
```

## 2. 기존 통합 SQLite 데이터 이전

기존 Flask/Spring 프로세스를 정상 종료해 SQLite WAL 쓰기를 끝낸 뒤 `users.db`, `users/`,
`defaults/music/`을 별도 백업합니다. 원본은 이전 검증이 끝날 때까지 수정하거나 삭제하지
마십시오.

먼저 새 PostgreSQL과 API를 한 번 실행해 Flyway 스키마를 생성한 다음 API를 중지합니다.
로컬 Compose의 PostgreSQL 포트는 보안을 위해 `127.0.0.1`에만 바인딩됩니다.

```powershell
docker compose --env-file .env up -d postgres api
Invoke-RestMethod http://127.0.0.1:5000/health
docker compose stop api
```

일회성 migration 환경을 만들고, 기존 SQLite를 PostgreSQL 서비스 데이터와 queue 전용
`jobs.db`로 나눕니다. `jobs.db`가 이미 있으면 도구가 덮어쓰지 않고 중단합니다.

```powershell
py -3.12 -m venv .migration-venv
.\.migration-venv\Scripts\Activate.ps1
pip install -r .\scripts\requirements-migration.txt

python .\scripts\migrate-sqlite-to-postgres.py `
  --sqlite C:\backup\ebookstudio\users.db `
  --queue-output C:\backup\ebookstudio\jobs.db `
  --postgres-url "postgresql://ebookstudio:비밀번호@127.0.0.1:5432/ebookstudio"
```

빈 `app-data` volume인지 먼저 확인한 후 생성된 `jobs.db`와 사용자 산출물을 복사합니다.

```powershell
docker compose stop analysis-worker music-worker music-worker-gpu
docker volume create ebookstudio_app-data
docker run --rm `
  --mount "type=bind,source=C:\backup\ebookstudio,target=/legacy,readonly" `
  --mount "type=volume,source=ebookstudio_app-data,target=/data" `
  alpine:3.22 sh -c "cp /legacy/jobs.db /data/jobs.db && cp -a /legacy/users /data/users && mkdir -p /data/defaults && cp -a /legacy/defaults/music /data/defaults/music && chown -R 10001:10001 /data"
```

## 3. 전환 확인

```powershell
docker compose up -d api analysis-worker
docker compose ps
Invoke-RestMethod http://127.0.0.1:5000/health
docker compose logs --tail 100 api analysis-worker
```

`/health`에서 `persistence=postgresql`, `queue.backend=sqlite`, 분석 Worker heartbeat를 확인합니다. 그다음 기존 계정 로그인, 내 서재, PDF 한 권 업로드, 분석 완료와 오프라인 다운로드를 순서대로 확인합니다. 책 상세의 처리 이력·음악 목록과 마이페이지의 전체·책별·일별 통계까지 조회되면 현재 DB/API 전환 범위가 모두 확인된 것입니다.

## 4. 백업과 롤백

PostgreSQL과 파일/Job volume을 별도로 백업해야 합니다.

```powershell
docker compose exec -T postgres pg_dump -U ebookstudio -d ebookstudio -Fc > .\backups\ebookstudio-postgres.dump
docker run --rm -v ebookstudio_app-data:/data:ro -v "${PWD}/backups:/backup" `
  alpine:3.22 sh -c "tar czf /backup/ebookstudio-app-data.tgz -C /data ."
```

`docker compose down`은 named volume을 보존하지만 `docker compose down -v`는 PostgreSQL,
Job, 책, 음악과 모델 캐시를 삭제합니다. 기존 백업과 서버는 전환 검증 전까지 보존합니다.

## 5. GHCR 배포

GitHub Release를 발행하면 Actions가 API와 Worker 이미지를 게시합니다. 배포 서버에서는
`.env.docker.example`을 복사해 이미지 경로, JWT 비밀키, PostgreSQL 비밀번호와 SMTP를
설정합니다. 예제의 `POSTGRES_PASSWORD`와 `SECRET_KEY`는 의도적으로 비어 있어 값을 채우지 않으면 Compose가 시작 전에 실패합니다.

```powershell
Copy-Item .env.docker.example .env.docker
docker login ghcr.io
docker compose --env-file .env.docker -f compose.deploy.yaml pull
docker compose --env-file .env.docker -f compose.deploy.yaml up -d
```

배포용 PostgreSQL은 호스트 포트를 공개하지 않고 Compose backend network에서만 API와
통신합니다. 외부 DB를 사용할 경우 `compose.deploy.yaml`을 수정하지 말고 별도 Compose override에서 `SPRING_DATASOURCE_*` 환경을 덮어쓰십시오. 테이블 관계와 Flyway 관리 원칙은 [DATABASE.md](DATABASE.md)를 참고하십시오.
