# EBookStudio Python Worker

Spring Boot가 SQLite 작업 큐에 등록한 `analyze`와 `music_generation` 작업을
처리합니다. HTTP API나 Flask 서버는 포함하지 않습니다.

## 실행

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-worker.txt
python spring_worker.py
```

역할을 분리해 실행할 수도 있습니다.

```powershell
python spring_worker.py --role analyze
python spring_worker.py --role music_generation
```

기본 데이터 경로는 저장소 상위 폴더의 `users.db`와 `defaults/`, `users/`이며,
Spring과 동일하게 `EBOOK_DB_PATH`, `EBOOK_STORAGE_ROOT`로 재정의할 수 있습니다.

## 테스트

```powershell
pip install -r requirements-test.txt
python -m pytest -q
```
