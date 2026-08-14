# EBookStudio Spring Server

인증, 사용자/도서 소유권, PDF 업로드, 작업 상태, 파일 제공을 담당합니다.
무거운 PDF 및 음악 처리는 형제 폴더 `../python-worker/`의 Worker가 수행합니다.
전체 구성과 운영 경계는 상위 폴더의 `ARCHITECTURE.md`를 참고하십시오.

## 실행

```powershell
.\mvnw.cmd spring-boot:run
```

기본 포트는 5000입니다. 환경 변수 전체 예시는 상위 폴더의 `.env.example`과
`README.md`를 참고하십시오. 컨테이너 실행은 상위 폴더에서 `docker compose up --build -d`,
음악 생성은 `--profile cpu-music` 또는 `--profile gpu` 중 하나를 추가합니다.

## 주요 API

- 인증: `/send_code`, `/verify_code`, `/register`, `/login`, `/refresh`, `/logout`
- 계정: `/find_id`, `/reset_password`, `/change_password`, `DELETE /account`
- 도서: `/upload_book`, `/check_status/{jobId}`, `/my_books`, `/delete_server_book`
- 파일: `/list_music_files/...`, `/files/...`
- 사용량: `POST /usage/events`, `GET /usage/summary`

`/find_id`와 `/reset_password`는 복구용 이메일 인증번호가 필요합니다.
업로드 완료 응답은 `music_job_id`를 포함할 수 있으며, 클라이언트는 해당 작업이
완료되면 JSON과 음악 파일을 다시 동기화해야 합니다.

Spring과 Python 워커는 같은 SQLite의 `music_prompt_cache`를 사용합니다. Spring은
테이블을 초기화하고, 워커는 프롬프트 서명별 생성·완료·재사용·실패 상태를 기록합니다.

사용량 이벤트는 인증된 본인 계정에만 기록되며 `(user_uuid, event_id)`로 중복을
제거합니다. 서버는 콘텐츠나 메모를 받지 않고 집계에 필요한 최소 필드만 저장합니다.