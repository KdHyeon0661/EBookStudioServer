import os
import secrets
import time
import json
import random
import glob
import threading
import shutil
import uuid
from datetime import timedelta, datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# [라이브러리 사용] JWT 매니저
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)

# [사용자 정의 모듈]
from background_music_jobs import BackgroundMusicJobRunner
from indexer import create_music_index

app = Flask(__name__)

# ==========================================
# 1. 설정 및 초기화
# ==========================================
basedir = os.path.abspath(os.path.dirname(__file__))

# JWT 설정
app.config['JWT_SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_hex(24)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=7)

# [핵심] 폴더 설정
USERS_BASE_FOLDER = os.path.join(basedir, 'users')
DEFAULTS_FOLDER = os.path.join(basedir, 'defaults')
# [추가] 공용 음악 폴더 경로 변수
DEFAULTS_MUSIC_FOLDER = os.path.join(DEFAULTS_FOLDER, 'music')

# 폴더 자동 생성
if not os.path.exists(USERS_BASE_FOLDER): os.makedirs(USERS_BASE_FOLDER)
if not os.path.exists(DEFAULTS_FOLDER): os.makedirs(DEFAULTS_FOLDER)
if not os.path.exists(DEFAULTS_MUSIC_FOLDER): os.makedirs(DEFAULTS_MUSIC_FOLDER)

app.config['USERS_FOLDER'] = USERS_BASE_FOLDER
app.config['DEFAULTS_FOLDER'] = DEFAULTS_FOLDER
app.config['DEFAULTS_MUSIC_FOLDER'] = DEFAULTS_MUSIC_FOLDER

# DB 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
jwt = JWTManager(app)

# 백그라운드 작업 실행기
bg_runner = BackgroundMusicJobRunner(app.config['USERS_FOLDER'])

# 이메일 인증 코드 저장소
VERIFICATION_CODES = {}
VERIFICATION_LOCK = threading.Lock()


# ==========================================
# 2. DB 모델
# ==========================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class TokenBlocklist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False)


@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload["jti"]
    token = db.session.query(TokenBlocklist.id).filter_by(jti=jti).scalar()
    return token is not None


with app.app_context():
    db.create_all()


# ==========================================
# 3. 유틸리티
# ==========================================
def _is_safe_segment(seg: str) -> bool:
    if not seg or seg in {".", ".."}: return False
    if "/" in seg or "\\" in seg: return False
    return True


def _safe_join(base: str, *parts: str) -> str:
    path = os.path.abspath(os.path.join(base, *parts))
    base_abs = os.path.abspath(base)
    if not (path == base_abs or path.startswith(base_abs + os.sep)):
        raise ValueError("Unsafe path")
    return path


def _resolve_user_uuid(username_or_uuid: str) -> str | None:
    try:
        current_uuid = get_jwt_identity()
    except Exception:
        return None
    if not current_uuid: return None
    if username_or_uuid == current_uuid: return current_uuid
    try:
        u = User.query.filter_by(public_id=current_uuid).first()
        if u and username_or_uuid == u.username: return current_uuid
    except Exception:
        pass
    return None


def _resolve_client_username(current_uuid: str) -> str:
    try:
        u = User.query.filter_by(public_id=current_uuid).first()
        if u and u.username: return u.username
    except Exception:
        pass
    return current_uuid


ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==========================================
# 4. 이메일 인증 API
# ==========================================
@app.route('/send_code', methods=['POST'])
def send_code():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    if not email: return jsonify({'message': 'Email is required'}), 400
    code = str(random.randint(100000, 999999))
    with VERIFICATION_LOCK:
        VERIFICATION_CODES[email] = {'code': code, 'timestamp': time.time() + 300}
    print(f"📧 [Email Verification] To: {email}, Code: {code}")
    return jsonify({'message': 'Code sent'}), 200


@app.route('/verify_code', methods=['POST'])
def verify_code():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    code = data.get('code')
    if not email or not code: return jsonify({'message': 'Email and code required'}), 400
    with VERIFICATION_LOCK:
        stored = VERIFICATION_CODES.get(email)
    if not stored: return jsonify({'message': 'Request code first'}), 400
    if time.time() > stored['timestamp']:
        with VERIFICATION_LOCK: VERIFICATION_CODES.pop(email, None)
        return jsonify({'message': 'Code expired'}), 400
    if stored['code'] == str(code):
        with VERIFICATION_LOCK: VERIFICATION_CODES.pop(email, None)
        return jsonify({'message': 'Verified'}), 200
    return jsonify({'message': 'Invalid code'}), 400


# ==========================================
# 5. 인증 API
# ==========================================
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password: return jsonify({'message': 'Missing fields'}), 400
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'message': 'User already exists'}), 400
    new_uuid = str(uuid.uuid4())
    new_user = User(username=username, email=email, public_id=new_uuid)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    user_folder = os.path.join(USERS_BASE_FOLDER, new_uuid)
    os.makedirs(user_folder, exist_ok=True)
    return jsonify({'message': 'Registered successfully', 'user_id': new_uuid}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'message': 'Invalid credentials'}), 401
    access_token = create_access_token(identity=user.public_id, additional_claims={"username": user.username})
    refresh_token = create_refresh_token(identity=user.public_id)
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'username': user.username,
        'public_id': user.public_id
    }), 200


@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify({'token': new_access_token}), 200


@app.route('/logout', methods=['POST'])
@jwt_required(verify_type=False)
def logout():
    jti = get_jwt()["jti"]
    now = datetime.now(timezone.utc)
    db.session.add(TokenBlocklist(jti=jti, created_at=now))
    db.session.commit()
    return jsonify({"message": "Successfully logged out"}), 200


# ==========================================
# 6. 파일 서빙 및 업로드 (핵심 로직 수정됨)
# ==========================================

# (1) 음악 파일 라우팅: 클라이언트가 "music/song.wav"를 요청하면 공용 폴더에서 꺼내줌
@app.route('/files/<username>/<book_folder>/music/<filename>')
@jwt_required(optional=True)
def serve_music_file(username, book_folder, filename):
    try:
        # 안전한 파일명인지 체크
        if not _is_safe_segment(filename): return "Access Denied", 403

        base_music_dir = app.config['DEFAULTS_MUSIC_FOLDER']  # defaults/music

        # [수정된 핵심 로직]
        # 파일이 base_music_dir 루트에 있을 수도 있고 (Preset),
        # storage_xxx 하위 폴더에 있을 수도 있음 (AI Gen).
        # 따라서 찾아내야 함.

        # 1. 루트 검사 (Preset 파일 등)
        if os.path.exists(os.path.join(base_music_dir, filename)):
            return send_from_directory(base_music_dir, filename)

        # 2. 하위 폴더(storage_xxx) 검사
        found_dir = None
        for entry in os.scandir(base_music_dir):
            if entry.is_dir():
                target_path = os.path.join(entry.path, filename)
                if os.path.exists(target_path):
                    found_dir = entry.path
                    break

        if found_dir:
            return send_from_directory(found_dir, filename)
        else:
            return "File Not Found", 404

    except Exception as e:
        print(f"Error serving music: {e}")
        return "File Not Found", 404


# (2) 일반 파일(JSON, 이미지, PDF) 서빙: 사용자의 책 폴더 루트에서 꺼내줌 (Flat 구조)
@app.route('/files/<username>/<book_folder>/<filename>')
@jwt_required(optional=True)
def serve_root_file(username, book_folder, filename):
    # 1. UUID 확인
    user_uuid = _resolve_user_uuid(username)
    if not user_uuid: return "Unauthorized", 403

    # 2. 경로 안전성 검사
    if not (_is_safe_segment(book_folder) and _is_safe_segment(filename)):
        return "Access Denied", 403

    # 3. 사용자 책 폴더 경로: users/{uuid}/{book_title}/
    try:
        user_book_dir = _safe_join(app.config['USERS_FOLDER'], user_uuid, book_folder)
        return send_from_directory(user_book_dir, filename)
    except Exception:
        return "File Not Found", 404


@app.route('/upload_book', methods=['POST'])
@jwt_required()
def upload_book():
    current_user_uuid = get_jwt_identity()
    if 'file' not in request.files: return jsonify({'message': 'No file part'}), 400
    file = request.files['file']

    if file and file.filename.lower().endswith('.pdf'):
        filename_safe = secure_filename(file.filename)
        book_folder_name = os.path.splitext(filename_safe)[0]

        # [핵심] texts, covers 폴더 없이 "users/uuid/BookName/"에 모두 저장 (Flat)
        save_dir = os.path.join(USERS_BASE_FOLDER, current_user_uuid, book_folder_name)

        # 공용 음악 폴더 경로 (defaults/music) -> 인자로 넘기지만 analyzer는 쓰지 않음
        music_folder = app.config['DEFAULTS_MUSIC_FOLDER']

        # 기존 폴더 초기화 (덮어쓰기)
        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # PDF 저장
        pdf_path = os.path.join(save_dir, filename_safe)
        file.save(pdf_path)

        client_username = _resolve_client_username(current_user_uuid)
        # Web Path Prefix: /files/user/BookName
        web_path_prefix = f"/files/{client_username}/{book_folder_name}"

        # 인덱스 최신화
        try:
            create_music_index()
        except:
            pass

        # 백그라운드 작업 등록
        # analyzer.py가 JSON과 이미지를 Flat하게 저장하도록 세팅됨
        job_id = bg_runner.enqueue(
            job_type='analyze',
            username=current_user_uuid,
            book_id=book_folder_name,
            pdf_path=pdf_path,
            book_root_folder=save_dir,  # 루트 폴더 전달
            music_folder=music_folder,  # 공용 음악 폴더 전달 (참조용)
            web_path_prefix=web_path_prefix
        )

        return jsonify({
            'message': 'Upload successful. Processing started.',
            'job_id': job_id,
            'book_title': book_folder_name
        }), 202

    return jsonify({'message': 'Invalid file type'}), 400


# ==========================================
# 7. 기타 조회 API
# ==========================================
@app.route('/get_toc', methods=['POST'])
@jwt_required()
def get_toc():
    current_user = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    user_uuid = _resolve_user_uuid(username) if username else None
    if not user_uuid: return jsonify({'message': 'Unauthorized'}), 403

    filename = data.get('filename')
    # book.pdf -> book_full.json
    json_filename = os.path.splitext(filename)[0] + "_full.json"

    try:
        user_base = _safe_join(app.config['USERS_FOLDER'], user_uuid)
        # [핵심] 책 폴더 바로 아래에서 json 찾기
        patterns = [os.path.join(user_base, '*', json_filename)]
        found_files = []
        for pat in patterns:
            found_files.extend(glob.glob(pat))

        if not found_files: return jsonify({'toc': []}), 404

        with open(found_files[0], 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        toc = [ch.get('title') for ch in book_data.get('chapters', []) if isinstance(ch, dict)]
        return jsonify({'toc': toc}), 200
    except Exception:
        return jsonify({'toc': []}), 500


@app.route('/sync_library', methods=['POST'])
@jwt_required()
def sync_library():
    current_user = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    user_uuid = _resolve_user_uuid(username) if username else None
    if not user_uuid: return jsonify({"message": "Unauthorized"}), 403
    book_title = data.get('book_title')

    try:
        book_dir = _safe_join(app.config['USERS_FOLDER'], user_uuid, book_title)
        if not os.path.exists(book_dir): return jsonify({"message": "Book folder not found"}), 404

        files = os.listdir(book_dir)
        # [핵심] _full.json으로 끝나는 파일 검색 (Flat 구조)
        json_file = next((f for f in files if f.endswith('_full.json')), None)

        if json_file:
            with open(os.path.join(book_dir, json_file), 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            return jsonify({"message": "Success", "book_data": book_data, "json_filename": json_file}), 200
        else:
            return jsonify({"message": "JSON File not found"}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route('/list_music_files/<username>/<book_title>', methods=['GET'])
@jwt_required()
def list_music_files(username, book_title):
    user_uuid = _resolve_user_uuid(username)
    if not user_uuid: return jsonify({"message": "Unauthorized"}), 403

    book_dir = _safe_join(app.config['USERS_FOLDER'], user_uuid, book_title)

    json_file = None
    if os.path.isdir(book_dir):
        for f in os.listdir(book_dir):
            if f.endswith('_full.json'):
                json_file = os.path.join(book_dir, f)
                break

    files = set()
    if json_file and os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                book_data = json.load(f)
            # JSON 파싱하여 음악 파일명만 추출
            for ch in book_data.get('chapters', []) or []:
                for seg in ch.get('segments', []) or []:
                    fn = seg.get('music_filename')
                    if fn: files.add(fn.strip())
        except:
            pass
    # 정렬하여 리턴 (클라이언트는 이 목록으로 공용 폴더 다운로드 진행)
    return jsonify({'files': sorted(files)}), 200


@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"}), 200


@app.route('/my_books', methods=['POST'])
@jwt_required()
def get_my_books():
    """사용자의 서버 저장소에 있는 책 목록 반환"""
    current_uuid = get_jwt_identity()
    user_dir = os.path.join(app.config['USERS_FOLDER'], current_uuid)

    books = []
    if os.path.exists(user_dir):
        # 유저 폴더 내의 하위 폴더(책 제목)들을 스캔
        for folder_name in os.listdir(user_dir):
            book_path = os.path.join(user_dir, folder_name)
            if os.path.isdir(book_path):
                # 표지 이미지 확인 (커버가 있으면 URL 생성)
                cover_url = ""
                if os.path.exists(os.path.join(book_path, f"{folder_name}.png")):
                    # 클라이언트가 가져갈 수 있는 웹 경로
                    client_username = _resolve_client_username(current_uuid)
                    cover_url = f"/files/{client_username}/{folder_name}/{folder_name}.png"

                books.append({
                    "title": folder_name,
                    "cover_url": cover_url
                })

    return jsonify({"books": books}), 200


@app.route('/delete_server_book', methods=['POST'])
@jwt_required()
def delete_server_book():
    """서버에서 책 데이터 삭제 (음악은 보존)"""
    current_uuid = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    book_title = data.get('book_title')

    if not book_title: return jsonify({'message': 'Book title required'}), 400

    target_dir = os.path.join(app.config['USERS_FOLDER'], current_uuid, book_title)

    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)  # 폴더 통째로 삭제

            # (선택) 인덱스 정리 로직이 필요하다면 여기에 추가
            # 하지만 음악 파일은 안 지웠으므로 인덱스는 유지하는 것이 맞음.

            return jsonify({'message': 'Deleted successfully'}), 200
        except Exception as e:
            return jsonify({'message': f'Error: {str(e)}'}), 500
    else:
        return jsonify({'message': 'Book not found'}), 404


@app.route('/find_id', methods=['POST'])
def find_id():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    user = User.query.filter_by(email=email).first()
    if user: return jsonify({'message': 'Success', 'username': user.username}), 200
    return jsonify({'message': 'Not found'}), 404


@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    new_password = data.get('new_password')
    user = User.query.filter_by(email=email).first()
    if user:
        user.set_password(new_password)
        db.session.commit()
        return jsonify({'message': 'Password reset successfully'}), 200
    return jsonify({'message': 'User not found'}), 404


def _start_periodic_execute():
    if os.environ.get("ENABLE_PERIODIC_EXECUTE", "1") != "1": return
    interval = int(os.environ.get("EXECUTE_INTERVAL_SECONDS", "3600"))
    max_jobs = int(os.environ.get("MAX_JOBS_PER_RUN", "5"))

    def _loop():
        while True:
            try:
                # 큐에 쌓인 분석/생성 작업 처리
                result = bg_runner.execute(max_jobs=max_jobs)
                # 작업이 있었을 때만 로그 출력
                if result.get("ran", 0) > 0:
                    print(f"🕒 [JobRunner] {result}")
            except Exception as e:
                print(f"❌ execute() loop error: {e}")
            time.sleep(2)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


print("🔥 서버 초기화 중... 인덱싱을 강제로 시작합니다.")
try:
    create_music_index()
except Exception as e:
    print(f"❌ 초기 인덱싱 실패: {e}")

# 백그라운드 워커 시작
_start_periodic_execute()

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug)