# 기존 서버 교체와 Docker 배포

이 저장소에서는 Spring Boot가 기존 Flask HTTP 서버를 완전히 대체합니다. Python은
`python-worker/`에서 PDF 분석과 MusicGen 작업만 수행하며 외부 포트를 열지 않습니다.

## 1. 새 설치에서 바로 실행

Docker Desktop을 시작한 뒤 저장소 루트에서 실행합니다.

```powershell
.\scripts\docker-up.ps1
```

최초 실행 시 `.env.example`을 복사해 `.env`를 만들고 안전한 무작위
`SECRET_KEY`를 자동으로 기록합니다. 기본 구성은 Spring API와 분석 Worker입니다.

```powershell
# CPU MusicGen 포함
.\scripts\docker-up.ps1 -MusicProfile cpu-music

# NVIDIA GPU MusicGen 포함
.\scripts\docker-up.ps1 -MusicProfile gpu
```

API는 기본적으로 `http://127.0.0.1:5000`에만 바인딩됩니다. 같은 네트워크의 다른
PC에서 WPF가 접속해야 한다면 `.env`의 `API_BIND_ADDRESS=0.0.0.0`으로 변경하고
방화벽 및 TLS 종단 프록시를 별도로 설정합니다.

## 2. 기존 Flask 서버의 데이터 이전

기존 서버를 먼저 중지해 SQLite 쓰기를 완전히 멈춘 다음 다음 항목을 함께 백업합니다.

```text
users.db
users/
defaults/music/
defaults/music_index.json
```

`users.db-wal`이나 `users.db-shm`이 남아 있다면 DB 파일만 임의로 복사하지 말고,
기존 프로세스를 정상 종료한 뒤 SQLite 백업을 생성해야 합니다. 가장 안전한 전환은
기존 데이터 디렉터리 전체를 별도 위치에 복사하고 원본은 보존하는 것입니다.

현재 Compose는 `ebookstudio_app-data` named volume을 사용합니다. 백업 디렉터리가
`C:\backup\ebookstudio`라면 빈 volume에 다음처럼 넣습니다.

```powershell
docker volume create ebookstudio_app-data
docker run --rm `
  --mount "type=bind,source=C:\backup\ebookstudio,target=/legacy,readonly" `
  --mount "type=volume,source=ebookstudio_app-data,target=/data" `
  alpine:3.22 sh -c "cp -a /legacy/. /data/ && chown -R 10001:10001 /data"
```

이미 데이터가 들어 있는 volume에 이 명령을 실행하면 파일이 덮어써질 수 있으므로,
반드시 새 volume인지 `docker run --rm -v ebookstudio_app-data:/data alpine:3.22 ls -la /data`
로 먼저 확인합니다.

## 3. 전환 확인

```powershell
docker compose ps
Invoke-RestMethod http://127.0.0.1:5000/health
docker compose logs --tail 100 api analysis-worker
```

기존 계정 로그인, 내 서재 목록, PDF 한 권 업로드, 작업 완료 후 오프라인 다운로드를
순서대로 확인합니다. 그 뒤 WPF의 `EBOOK_API_BASE_URL` 또는 배포 설정을 새 Spring
주소로 바꿉니다. 같은 호스트와 포트 `5000`을 유지한다면 WPF 변경 없이 교체할 수
있습니다.

기존 Flask 실행 파일과 데이터 백업은 즉시 삭제하지 말고 전환 확인이 끝날 때까지
보존합니다. 문제가 생기면 새 Compose를 중지하고 기존 데이터와 Flask 프로세스를
원래 포트로 되돌리면 됩니다.

```powershell
docker compose down
```

`docker compose down -v`는 DB와 책 파일이 든 volume까지 삭제하므로 롤백 과정에서
사용하면 안 됩니다.

## 4. GitHub Container Registry 이미지로 배포

GitHub에서 Release를 발행하면 `.github/workflows/publish-images.yml`이 네 이미지를
GHCR에 게시합니다. 저장소의 Actions 설정에서 workflow의 패키지 쓰기 권한이 허용돼
있어야 합니다.

배포 서버에서는 예제 환경 파일을 복사하고 GitHub 사용자 또는 조직명을 지정합니다.

```powershell
Copy-Item .env.docker.example .env.docker
# IMAGE_PREFIX, IMAGE_TAG, SECRET_KEY와 SMTP 값을 편집

docker login ghcr.io
docker compose --env-file .env.docker -f compose.deploy.yaml pull
docker compose --env-file .env.docker -f compose.deploy.yaml up -d
```

MusicGen 이미지는 필요한 프로필 하나만 선택합니다.

```powershell
docker compose --env-file .env.docker -f compose.deploy.yaml `
  --profile gpu up -d
```

공개 패키지는 로그인 없이 pull할 수 있습니다. 비공개 패키지는 `read:packages` 권한이
있는 GitHub PAT로 `docker login ghcr.io`를 수행해야 합니다.

