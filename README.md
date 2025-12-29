# EbookStudioServer (Flask)

WPF 클라이언트(EBookStudio)에서 PDF를 업로드하면, 서버가 PDF를 분석해 **표지(PNG)** 와 **전체 텍스트 JSON(_full.json)** 을 생성하고, JSON의 각 세그먼트에 **프리셋 음악(또는 AI 생성 음악)** 을 매핑하는 Flask 서버입니다.

> 현재 구조는 **Flat 저장 방식**입니다. (books/texts/covers 폴더로 나누지 않고 `users/<uuid>/<book>/` 루트에 PNG/JSON/PDF를 저장)

---

## 핵심 동작 요약

1. 클라이언트가 `/upload_book` 로 PDF 업로드
2. 백그라운드 큐에 `analyze` 작업 등록
3. `analyzer.py`가 결과 생성
   - `users/<uuid>/<book>/<book>.png`
   - `users/<uuid>/<book>/<book>_full.json`
4. JSON 내부 `music_path`는 `"music/<filename>"` 형태로 기록  
   - 책 폴더 안에 `music/` 폴더를 만들지 않습니다.
   - 음악 실파일은 `defaults/music/` 또는 `defaults/music/storage_xxx/`에 존재합니다.
5. 클라이언트는 `/list_music_files/<username>/<book>`로 필요한 음악 파일명을 얻고,
   `/files/<username>/<book>/music/<filename>`로 다운로드합니다.

---

## 폴더 구조

프로젝트 루트 기준:

```
EbookStudioServer/
  server.py
  analyzer.py
  background_music_jobs.py
  indexer.py

  defaults/
    default.png
    emotions_20.py
    music_genres_200.py
    genre_bpm_connector.py
    music_index.json
    music/                 # 프리셋/AI 생성 음악 저장소(공용)
      storage_001/         # (자동 생성될 수 있음)

  users/                   # 런타임 생성 (업로드/결과 저장소)
    <uuid>/<BookTitle>/
      <BookTitle>.pdf
      <BookTitle>.png
      <BookTitle>_full.json
    _bg_jobs.json          # 런타임 큐 파일

  users.db                 # SQLite DB(런타임)
```

---

## 요구 사항

- Python 3.10+ 권장
- 주요 라이브러리
  - flask, flask_sqlalchemy, flask_jwt_extended, werkzeug
  - PyMuPDF (`fitz`)
  - nltk (wordnet 자동 다운로드 시도)
  - **AI 음악 생성 파이프라인 의존성(현재 import가 강제됨)**  
    torch, transformers, keybert, scipy, numpy

> ⚠️ `background_music_jobs.py`는 import 시점에 torch/transformers 등을 바로 import 합니다.  
> “프리셋 음악만” 쓰고 싶다면 해당 파일을 옵션화/지연 import로 리팩터링해야 합니다.

---

## 실행 방법(개발)

### 1) 가상환경 & 설치

```bash
cd EbookStudioServer
python -m venv .venv

# Windows
# .venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -U pip
pip install flask flask_sqlalchemy flask_jwt_extended werkzeug
pip install PyMuPDF nltk
pip install torch transformers keybert scipy numpy
```

### 2) 서버 실행

```bash
python server.py
```

- 기본 주소: `http://0.0.0.0:5000`
- 헬스체크: `GET /health`

---

## 환경변수

- `SECRET_KEY` : JWT 서명 키  
  (미설정 시 매 실행마다 랜덤 생성 → 서버 재시작하면 기존 토큰이 사실상 무효가 될 수 있음)
- `FLASK_DEBUG` : `1`이면 debug 모드 (기본 `1`)
- `ENABLE_PERIODIC_EXECUTE` : `1`이면 백그라운드 큐 실행 (기본 `1`)
- `EXECUTE_INTERVAL_SECONDS` : 설정은 존재하나 현재 루프는 2초 sleep을 사용
- `MAX_JOBS_PER_RUN` : 한 번에 처리할 큐 작업 수 (기본 `5`)
- `KEYWORDS_HISTORY_PATH` : 키워드 히스토리 저장 경로 (기본 `defaults/keywords_history.json`)

---

## API 요약

### Auth / 계정
- `POST /register` `{ username, email, password }`
- `POST /login` `{ username, password }` → access/refresh 반환
- `POST /refresh` (refresh 토큰 필요)
- `POST /logout` (토큰 blocklist 등록)

### 이메일 인증(현재 콘솔 출력)
- `POST /send_code` `{ email }`
- `POST /verify_code` `{ email, code }`

### 책/라이브러리
- `POST /upload_book` (multipart: `file=PDF`) → 202 + `job_id`, `book_title`
- `POST /sync_library` `{ username, book_title }` → book_data(JSON) 반환
- `POST /get_toc` `{ username, filename }` → 목차 리스트 반환
- `GET  /list_music_files/<username>/<book_title>` → JSON에서 사용된 음악 파일명 목록
- `POST /my_books` → 서버에 저장된 내 책 목록
- `POST /delete_server_book` `{ book_title }` → 서버 책 폴더 삭제(음악은 보존)

### 파일 서빙
- `GET /files/<username>/<book_folder>/<filename>`  
  → `users/<uuid>/<book_folder>/<filename>` 에서 제공 (**사실상 인증 필요**)
- `GET /files/<username>/<book_folder>/music/<filename>`  
  → `defaults/music/` 또는 `defaults/music/storage_xxx/`에서 찾아 제공

> `<username>` 자리에는 실제로 `username` 또는 `uuid`를 넣어도 동작할 수 있으며, 최종 권한은 JWT identity(uuid)로 검증합니다.

---

## 개발 메모

- `users/`, `users.db`, `__pycache__/`, `defaults/keywords_history.json` 는 런타임 산출물이므로 커밋하지 않는 것을 권장합니다.
- 음악 파일(wav 등)은 용량이 커질 수 있어 Git LFS를 고려하거나 `.gitignore`로 제외하는 것을 추천합니다.
