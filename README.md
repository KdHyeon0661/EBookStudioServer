# EBookStudio Server

EBookStudio의 유일한 API 진입점은 `spring-server/`의 Spring Boot 서버입니다.
Python은 `python-worker/spring_worker.py`를 통해 PDF 분석과 MusicGen 추론만 담당하며 HTTP API를
제공하지 않습니다. 구성 요소의 책임, 장애 복구와 확장 기준은
[ARCHITECTURE.md](ARCHITECTURE.md)에 정리되어 있습니다.

## 저장소 구조

```text
EBookStudioServer-master/
├─ spring-server/       # Spring Boot API와 업무 규칙
├─ python-worker/       # PDF 분석과 MusicGen 작업 처리
├─ compose.yaml         # API와 Worker 통합 실행
├─ README.md
└─ ARCHITECTURE.md
```

두 실행 모듈은 프레임워크 이름이 아니라 배포 역할로 구분합니다. Python 구성 요소는
HTTP API를 제공하는 Flask 서버가 아니므로 `flask-server`가 아닌 `python-worker`로
명명합니다.

## 처리 흐름

1. WPF가 Spring에 PDF를 업로드합니다.
2. Spring은 소유권과 파일을 검증하고 `analyze` 작업을 등록합니다.
3. Python 분석 워커가 표지와 구조화 JSON을 생성합니다.
4. 사용자는 포함된 `default_ambient.wav` 또는 기존 음악으로 즉시 읽을 수 있습니다.
5. MusicGen 워커가 세그먼트별 음악을 생성하고 현재 책 JSON을 원자적으로 갱신합니다.
6. WPF는 `music_job_id`를 추적해 갱신된 JSON과 음악을 다시 내려받습니다.

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
회원 탈퇴 시 이 집계 데이터도 같은 트랜잭션에서 삭제됩니다.

## 요구 사항

- Java 17 이상
- Python 3.11 또는 3.12 권장
- .NET 9 Windows SDK
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

두 프로세스가 같은 `EBOOK_DB_PATH`와 `EBOOK_STORAGE_ROOT`를 사용해야 합니다.
기본값은 저장소 루트의 `users.db`, `users/`, `defaults/`입니다.

```powershell
cd spring-server
.\mvnw.cmd spring-boot:run
```

다른 터미널에서 저장소 루트를 기준으로:

`powershell
cd python-worker
python spring_worker.py
```

워커 하나만 직접 실행할 때는 역할을 지정할 수 있습니다.

```powershell
python spring_worker.py --role analyze
python spring_worker.py --role music_generation
```

## CI와 기존 서버 교체

GitHub Actions는 Spring·Python 테스트, Compose 검증과 핵심 이미지 빌드를 수행합니다.
GitHub Release를 발행하면 GHCR 배포 이미지도 생성됩니다. 기존 Flask 서버의 데이터
백업, Spring 전환, 검증과 롤백 절차는 [DEPLOYMENT.md](DEPLOYMENT.md)를 따릅니다.
## Docker Compose

WPF는 Windows 네이티브 오프라인 클라이언트이므로 컨테이너 대상이 아니며,
Spring API와 Python 워커만 Docker로 실행합니다. 기본 구성은 API와 가벼운 PDF
분석 워커를 실행합니다.

`.env.example`은 인증번호를 보내지도 응답에 노출하지도 않는 안전한 기본값입니다.
로컬에서 계정 흐름을 시연할 때만 복사한 `.env`의
`EMAIL_EXPOSE_DEVELOPMENT_CODE=true`를 설정하십시오. 외부에 공개되는 환경에서는
반드시 `false`로 유지하고 SMTP를 설정합니다.

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://127.0.0.1:5000/health
```

음악 워커는 하드웨어에 따라 하나의 프로필만 선택합니다.

```powershell
# CPU 전용 PyTorch. 기능 확인용이며 생성 속도는 느립니다.
docker compose --profile cpu-music up --build -d

# NVIDIA Container Toolkit과 호환 GPU가 있는 환경
docker compose --profile gpu up --build -d
```

`data-init`은 named volume의 소유권을 1회 맞춘 뒤 종료합니다. 이후 API와 워커는
UID 10001, read-only root filesystem, capability 제거, `no-new-privileges` 상태로
실행됩니다. `/data`에는 SQLite와 사용자 산출물, `/models`에는 Hugging Face/PyTorch
모델 캐시가 유지됩니다. 상태 확인은 API `/health`와 각 워커의 SQLite heartbeat를
사용하며 로그는 컨테이너별 10MB × 3개로 회전합니다.

```powershell
docker compose logs -f api analysis-worker
docker compose down
```

`docker compose down`은 named volume을 보존하지만 `docker compose down -v`는 DB,
책과 모델 캐시를 삭제하므로 운영 데이터가 있을 때 사용하면 안 됩니다. 백업은 쓰기를
멈춘 뒤 volume 전체를 보관하는 방식이 가장 단순합니다.

```powershell
docker compose stop api analysis-worker music-worker music-worker-gpu
New-Item -ItemType Directory -Force backups | Out-Null
docker run --rm -v ebookstudio_app-data:/data:ro -v "${PWD}/backups:/backup" `
  alpine:3.22 sh -c "tar czf /backup/ebookstudio-data.tgz -C /data ."
docker compose start api analysis-worker
```

현재 SQLite 큐는 단일 Docker 호스트와 로컬 named volume을 전제로 합니다. API를 여러
인스턴스로 수평 확장하거나 여러 서버가 네트워크 파일시스템의 같은 DB를 공유하는
구성은 지원하지 않습니다. 그 단계에서 PostgreSQL을 영속 DB로, RabbitMQ 같은
브로커를 작업 전달 계층으로, S3 호환 저장소를 산출물 계층으로 분리하는 것이 전환
기준입니다. 현재 개인 프로젝트 규모에서는 이 구성이 운영 복잡도보다 명확한 이점을
줍니다.

SMTP 없이 UI를 로컬 시연할 때만 Spring 실행 전에 아래 값을 설정하면 인증번호가
API 응답으로 전달됩니다. 운영 환경에서는 절대 활성화하지 마십시오.

```powershell
$env:EMAIL_EXPOSE_DEVELOPMENT_CODE='true'
```

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
```

Spring 테스트는 회원가입, 로그인, 업로드 큐 등록, 비밀번호 변경, 로그아웃,
계정 복구와 회원 탈퇴뿐 아니라 인증번호 HMAC 저장, 요청 제한, 안전한 파일 삭제,
사용량 배치 멱등성과 계정 삭제 연동을 실제 HTTP로 검증합니다. Python 테스트는 작업 선점,
heartbeat 복구, 재시도, 프롬프트 카탈로그 중복 방지, 파일 유실 감지와
책 JSON 음악 교체를 검증합니다.