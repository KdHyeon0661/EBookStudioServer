# EBookStudio Python Worker

Spring Boot가 SQLite Job 큐에 등록한 `analyze`와 `music_generation` Job을 처리합니다.
HTTP API나 Flask 서버는 포함하지 않습니다. 사용자·인증·사용량 PostgreSQL에는 직접
접속하지 않으며, Worker의 책임은 SQLite Job 상태와 파일 산출물에 한정됩니다.

## 실행

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-worker.txt
$env:EBOOK_QUEUE_DB_PATH='C:\ebookstudio\data\jobs.db'
$env:EBOOK_STORAGE_ROOT='C:\ebookstudio\data'
python spring_worker.py
```

역할을 분리해 실행할 수도 있습니다.

```powershell
python spring_worker.py --role analyze
python spring_worker.py --role music_generation
```

기본 Job DB는 저장소 상위 폴더의 `jobs.db`입니다. 공식 환경변수는
`EBOOK_QUEUE_DB_PATH`이며, 이전 실행 환경과의 호환을 위해 `EBOOK_DB_PATH`도 fallback으로
읽습니다.

## 테스트

```powershell
pip install -r requirements-test.txt
python -m pytest -q
```
