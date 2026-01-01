# EbookStudio Server (Flask)

**EBookStudio** 클라이언트(WPF)를 위한 백엔드 서버입니다.
사용자가 PDF를 업로드하면 서버는 이를 분석하여 **표지(PNG)** 및 **구조화된 텍스트(JSON)** 를 생성하고, 각 문단(Segment)의 감정을 분석하여 **배경 음악(AI 생성 또는 프리셋)** 을 자동으로 매핑합니다.

> **저장 구조:** `users/<uuid>/<book>/` 경로에 PDF, PNG, JSON 파일을 모두 저장하는 **Flat 구조**를 따릅니다.

---

## 핵심 기능 및 동작 원리

1. **PDF 업로드 및 분석 요청**:
* 클라이언트가 `/upload_book`으로 PDF를 전송하면 서버는 파일을 저장하고 백그라운드 작업 큐(`analyze`)에 등록합니다.


2. **비동기 분석 (Analyzer)**:
* `analyzer.py`가 PDF를 파싱하여 표지 이미지와 텍스트 데이터를 추출합니다.
* 텍스트의 감정/분위기를 분석하여 적절한 음악을 매칭하거나 생성합니다.


3. **결과물 생성**:
* `users/<uuid>/<book>/<book>.png` (표지)
* `users/<uuid>/<book>/<book>_full.json` (메타데이터 및 본문)


4. **음악 매핑 및 제공**:
* JSON 내 `music_path`는 `"music/<filename>"` 형식으로 저장됩니다.
* 실제 음악 파일은 각 사용자 폴더가 아닌 **공용 스토리지(`defaults/music/`)** 에 저장되어 관리됩니다.
* 클라이언트는 `/files/.../music/<filename>` 엔드포인트를 통해 중앙 저장소의 음악을 다운로드합니다.



---

## 폴더 구조

```bash
EbookStudioServer/
├── server.py                 # 메인 Flask 앱 및 라우트 핸들러
├── analyzer.py               # PDF 분석 및 텍스트 마이닝 로직
├── background_music_jobs.py  # AI 음악 생성 및 백그라운드 작업 관리
├── indexer.py                # 음악 파일 인덱싱 및 DB화
├── pytest.ini                # 테스트 설정 파일
├── requirements.txt          # (권장) 의존성 패키지 목록
│
├── defaults/                 # 서버 기본 리소스
│   ├── default.png
│   ├── music_index.json      # 음악 파일 인덱스 데이터
│   └── music/                # [중요] 음악 파일 저장소 (AI 생성/프리셋)
│       └── storage_xxx/      # 파일 개수 제한에 따른 하위 폴더
│
├── users/                    # [런타임 생성] 사용자 데이터 저장소
│   ├── _bg_jobs.json         # 백그라운드 작업 큐 상태
│   └── <uuid>/               # 사용자별 격리 공간
│       └── <BookTitle>/      # 책 단위 폴더
│
├── tests/                    # 단위 및 통합 테스트
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_logic.py
│   └── ...
│
└── users.db                  # [런타임 생성] 사용자/인증 정보 SQLite DB

```

---

## 설치 및 실행 방법

### 1. 필수 요건 (Prerequisites)

* **Python 3.10 이상**
* **FFmpeg** (오디오 처리를 위해 설치 권장, Windows/Linux 환경에 따라 필요할 수 있음)

### 2. 가상환경 설정 및 패키지 설치

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 필수 패키지 설치
pip install -U pip
pip install -r requirements.txt

```

> **참고:** `requirements.txt`가 없다면 다음 명령어로 주요 라이브러리를 직접 설치하세요.
> ```bash
> pip install flask flask_sqlalchemy flask_jwt_extended werkzeug
> pip install PyMuPDF nltk librosa
> pip install torch transformers keybert scipy numpy pytest pytest-flask pytest-mock requests-mock
> 
> ```
> 
> 

### 3. 서버 실행

```bash
python server.py

```

* 서버는 기본적으로 `http://0.0.0.0:5000`에서 실행됩니다.
* 최초 실행 시 NLTK 데이터 다운로드 및 음악 인덱싱이 수행될 수 있습니다.

---

## 테스트 실행

프로젝트의 안정성을 검증하기 위해 `pytest`를 사용합니다.

```bash
# 전체 테스트 실행
pytest

# 상세 결과 확인
pytest -v

```

---

## 환경 변수 설정

`.env` 파일 또는 시스템 환경 변수로 설정 가능합니다.

| 변수명 | 설명 | 기본값 |
| --- | --- | --- |
| `SECRET_KEY` | JWT 서명 및 보안 키 (프로덕션 필수 설정) | (Random UUID) |
| `FLASK_DEBUG` | 디버그 모드 활성화 여부 (`1`: On) | `1` |
| `ENABLE_PERIODIC_EXECUTE` | 백그라운드 작업 큐 자동 실행 여부 | `1` |
| `EXECUTE_INTERVAL_SECONDS` | 작업 큐 폴링 간격 (초) | `2` |
| `MAX_JOBS_PER_RUN` | 한 번에 처리할 최대 작업 수 | `5` |
| `KEYWORDS_HISTORY_PATH` | 키워드 히스토리 파일 경로 | `defaults/keywords_history.json` |

---

## API 명세 요약

모든 보안 엔드포인트는 헤더에 `Authorization: Bearer <access_token>`이 필요합니다.

### 1. 인증 (Auth)

* `POST /register`: 회원가입 (`username`, `email`, `password`, `code`)
* `POST /login`: 로그인 (`access_token`, `refresh_token` 발급)
* `POST /refresh`: 액세스 토큰 갱신
* `POST /logout`: 로그아웃 (토큰 만료 처리)
* `POST /send_code` & `/verify_code`: 이메일 인증 (현재 콘솔 출력으로 코드 확인)

### 2. 도서 관리 (Library)

* `POST /upload_book`: PDF 파일 업로드 및 분석 요청 (Multipart)
* 응답: `202 Accepted`, `job_id` 반환


* `POST /my_books`: 내 서재 목록 조회 (표지 URL 포함)
* `POST /get_toc`: 특정 도서의 목차(Table of Contents) 조회
* `POST /delete_server_book`: 서버에서 도서 데이터 삭제

### 3. 파일 및 리소스 (Files)

* `GET /list_music_files/<username>/<book_title>`: 해당 도서에 매핑된 음악 파일명 목록 조회
* `GET /files/<username>/<book_folder>/<filename>`: 도서 관련 파일(이미지, JSON 등) 다운로드
* `GET /files/<username>/<book_folder>/music/<filename>`: **음악 파일 다운로드** (공용 스토리지에서 서빙)
