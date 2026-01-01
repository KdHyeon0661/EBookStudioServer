# EbookStudioServer (Flask)

**EBookStudio 클라이언트(WPF)**를 위한 백엔드 서버입니다.  
사용자가 PDF를 업로드하면 서버는 이를 분석하여 **표지(PNG)** 및 **구조화된 텍스트(JSON)** 를 생성하고, 각 세그먼트에 맞는 **배경 음악(프리셋 매칭 / AI 생성)** 을 제공합니다.

---

## 동작 흐름(High-level)

1. 클라이언트가 `/upload_book`으로 PDF 업로드 (JWT 필요)
2. 서버는 `users/<uuid>/<BookTitle>/`에 PDF를 저장하고 **분석 Job(analyze)** 를 큐에 등록
3. `analyzer.py`가 PDF를 파싱해
   - 표지: `<BookTitle>.png`
   - 본문: `<BookTitle>_full.json` (챕터/세그먼트/페이지 구조)
   - 세그먼트별 `music_path = "music/<filename>"` 형태로 경로를 기록
4. 이어서 **음악 Job(music)** 이 실행되어 세그먼트를 재분석하고(키워드/감정) 음악을 생성/재사용합니다.
5. 클라이언트는 `/files/<username>/<book>/...` 및 `/files/<username>/<book>/music/...`로 파일을 다운로드합니다.

> ⚠️ 현재 `background_music_jobs.py`는 세그먼트에 이미 `music_filename`이 있어도 **재생성/재매핑**하도록 되어 있습니다.  
> 프리셋 우선/미생성 구간만 생성하려면 해당 로직을 다시 “스킵”하도록 변경하는 것이 좋습니다.

---

## 저장 구조

책 데이터는 **Flat 구조**로 저장합니다.

```text
users/
└─ <uuid>/
   └─ <BookTitle>/
      ├─ <BookTitle>.pdf
      ├─ <BookTitle>.png
      └─ <BookTitle>_full.json
```

음악 파일은 사용자 폴더가 아니라 **공용 저장소**에 관리됩니다.

```text
defaults/
└─ music/
   └─ storage_001/
      ├─ <hash>.wav
      └─ <hash>.json    # (선택) 프롬프트/장르/BPM 등 메타데이터 사이드카
```

---

## 요구 사항
- **Python 3.10+**
- (권장) **FFmpeg**: 오디오 관련 처리/디코딩 보조
- AI 음악 생성을 쓰는 경우:
  - `torch`, `transformers` 등 대형 의존성 + (권장) GPU 환경

> `requirements.txt`에 `torch==2.9.1+cu130` 처럼 CUDA 버전이 고정되어 있을 수 있습니다.  
> 환경에 맞는 PyTorch 설치 가이드를 따라 `torch` 버전을 조정하세요.

---

## 설치 & 실행

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

python server.py
```

- 기본 실행 주소: `http://0.0.0.0:5000`
- 최초 실행 시:
  - NLTK WordNet 데이터 다운로드가 발생할 수 있음
  - 음악 인덱서(`indexer.create_music_index`)가 실행되어 `defaults/music_index.json`을 생성/갱신

---

## 환경 변수

| 변수명 | 설명 | 기본값(코드 기준) |
| --- | --- | --- |
| `SECRET_KEY` | JWT 서명 키 | `my-super-secret-fixed-key` |
| `FLASK_DEBUG` | Flask 디버그 모드 (`1`=ON) | `1` |
| `ENABLE_PERIODIC_EXECUTE` | 백그라운드 큐 자동 실행 | `1` |
| `EXECUTE_INTERVAL_SECONDS` | 큐 폴링 간격(초) | `2` |
| `MAX_JOBS_PER_RUN` | 1회 처리 최대 Job 수 | `5` |
| `KEYWORDS_HISTORY_PATH` | 키워드 히스토리 파일 경로 | `defaults/keywords_history.json` |

---

## API 요약

### Auth / Account
- `POST /send_code` : 이메일 인증 코드 발급 (현재는 콘솔 출력)
- `POST /verify_code` : 코드 검증
- `POST /register` : 회원가입
- `POST /login` : 로그인(JWT 발급)
- `POST /refresh` : 토큰 갱신(Refresh 토큰 필요)
- `POST /logout` : 로그아웃(토큰 블랙리스트)

### Library
- `POST /upload_book` : PDF 업로드 + 분석 요청 (JWT 필요, multipart)
- `POST /get_toc` : 목차 조회(JWT 필요)
- `POST /my_books` : 내 서재 목록(JWT 필요)
- `POST /delete_server_book` : 서버 책 삭제(JWT 필요)

### Files
- `GET /files/<username>/<book_folder>/<filename>` : PNG/JSON 다운로드
- `GET /files/<username>/<book_folder>/music/<filename>` : 음악 다운로드(공용 defaults/music에서 서빙)
- `GET /list_music_files/<username>/<book_title>` : 해당 책 JSON에 참조된 음악 파일명 목록

---

## 테스트

```bash
pytest
pytest -v
```

---

## 보안/운영 메모
- 인증 코드는 메모리(`VERIFICATION_CODES`)에 저장되어 **서버 재시작 시 사라집니다.**
- 실제 이메일 발송(SMTP)과 운영 환경의 시크릿 관리(SECRET_KEY), 파일 업로드 보안(확장자/경로 검증 강화) 등은 프로덕션에서 필수로 보강해야 합니다.
