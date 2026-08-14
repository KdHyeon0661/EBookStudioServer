# EBookStudio 아키텍처

## 목표와 경계

EBookStudio는 온라인에서 PDF를 책 데이터로 변환하고 배경 음악을 준비한 뒤,
다운로드가 끝난 책은 Windows 클라이언트에서 오프라인으로 읽는 서비스입니다.
HTTP API와 업무 규칙은 Spring Boot가 소유하고, Python은 PDF 분석과 MusicGen 추론만
수행합니다. WPF 클라이언트는 서버 장애나 네트워크 단절이 독서 자체를 막지 않도록
완성된 책과 사용자 데이터를 로컬에 보관합니다.

## 구성 요소

| 구성 요소 | 책임 | 상태 저장 |
|---|---|---|
| WPF 클라이언트 | 로그인, 업로드, 작업 추적, 다운로드, 오프라인 독서, 사용량 outbox | `%LOCALAPPDATA%/EBookStudio/DownloadCache` |
| Spring Boot API | 인증·인가, 소유권 검증, 업로드, 작업 등록·조회·취소, 파일 제공, 사용량 집계 | SQLite와 파일 저장소 |
| 분석 워커 | PDF 검증 이후 표지·본문 JSON 생성, 기존 음악 매핑 | Spring과 공유하는 SQLite·파일 저장소 |
| MusicGen 워커 | 프롬프트 서명 기반 음악 재사용 또는 생성, 책 JSON 원자적 갱신 | 공용 음악 폴더와 `music_prompt_cache` |
| Docker Compose | 단일 호스트 배포, 볼륨·헬스체크·자원 제한·프로필 관리 | `app-data`, `model-cache` named volume |

## 주요 처리 흐름

1. WPF가 UUID `request_id`와 PDF를 Spring의 `/upload_book`에 보냅니다.
2. Spring이 PDF 헤더·사용자 소유권·경로를 검증하고 `analyze` 작업을 멱등 등록합니다.
3. 분석 워커가 작업을 원자적으로 선점하고 heartbeat를 기록하면서 표지와 JSON을 만듭니다.
4. 분석 완료 시 결정적 하위 작업 ID로 `music_generation` 작업을 등록합니다.
5. WPF는 분석 결과를 내려받아 원자적으로 로컬 캐시에 반영하므로 즉시 오프라인 독서가 가능합니다.
6. 음악 워커는 프롬프트 서명을 확인해 기존 WAV를 재사용하거나 새로 생성합니다.
7. 음악 작업이 끝나면 WPF가 갱신된 JSON과 필요한 WAV만 다시 동기화합니다.

## 신뢰성과 중복 방지

- 업로드 `request_id`는 재전송되어도 같은 분석 작업을 반환합니다.
- 작업 선점은 SQLite 조건부 갱신으로 한 워커만 성공합니다.
- 실행 중 작업은 heartbeat, 제한 재시도, 지수 backoff를 사용합니다.
- 오래된 작업만 재기동 시 복구하며 정상 워커의 작업은 빼앗지 않습니다.
- 취소는 `queued` 작업을 즉시 종료하고 `running` 작업에는 취소 요청을 전달합니다.
- 책 JSON과 WPF 로컬 JSON·다운로드 파일은 임시 파일 작성 후 교체합니다.
- 음악 프롬프트 서명과 파일 잠금이 같은 조건의 동시 생성을 직렬화합니다.
- 사용량 이벤트는 `(user_uuid, event_id)`로 멱등 저장됩니다.

## 보안과 개인정보

- 비밀번호는 BCrypt로 해시하며 API는 짧은 액세스 토큰과 회전형 리프레시 토큰을 사용합니다.
- 로그아웃된 토큰은 blocklist에 기록되고 보호 API는 사용자 소유권을 다시 확인합니다.
- 인증번호는 서버 비밀키 기반 HMAC으로 저장하며 로그인·인증번호 요청에 제한을 적용합니다.
- 파일 경로는 안전한 단일 경로 조각으로 제한하고 책 JSON이 참조한 음악만 제공합니다.
- 사용량에는 책 원문, 메모, 하이라이트, 페이지별 열람 이력을 넣지 않습니다.
- 운영에서는 TLS 종단 프록시 뒤에 API를 두고 개발용 인증번호 노출을 비활성화해야 합니다.

## 배포 프로필

- 기본: Spring API + PDF 분석 워커
- `cpu-music`: CPU MusicGen 워커를 추가하는 기능 확인용 구성
- `gpu`: NVIDIA GPU MusicGen 워커를 추가하는 실제 생성용 구성

컨테이너는 UID 10001, read-only root filesystem, capability 제거,
`no-new-privileges`와 로그 회전을 사용합니다. SQLite와 산출물은 같은 `app-data`
볼륨을 공유하므로 현재 배포 단위는 단일 Docker 호스트입니다.

## 확장 기준과 의도적인 제한

현재 SQLite 큐는 개인 프로젝트와 단일 호스트 운영에서 단순성과 관찰 가능성을
우선한 선택입니다. 다음 조건 중 하나가 생기면 저장·작업 계층을 분리합니다.

- API 또는 워커를 여러 호스트로 수평 확장해야 할 때
- 큐 적체, 우선순위, 지연 작업과 dead-letter 운영이 필요할 때
- 산출물 용량이 단일 호스트 백업 범위를 넘을 때

그 시점의 권장 경계는 PostgreSQL(영속 데이터), RabbitMQ(작업 전달), S3 호환
스토리지(책·음악 산출물)입니다. 현재 구성은 이 전환을 전제로 API, 워커, 파일
저장 책임을 이미 분리해 두었습니다.

## 검증 명령

```powershell
# Spring
cd spring-server
.\mvnw.cmd clean test

# Python 경량 테스트
cd ..\python-worker
pip install -r requirements-test.txt
python -m pytest -q

# Docker Compose 정적 검증
docker compose config --quiet

# WPF (클라이언트 저장소에서)
dotnet build EBookStudio\EBookStudio.csproj -c Release
```

실제 컨테이너 빌드와 헬스체크는 Docker 엔진이 실행 중인 환경에서
`docker compose up --build -d`와 `docker compose ps`로 확인합니다.
