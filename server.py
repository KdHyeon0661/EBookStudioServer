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

from background_music_jobs import BackgroundMusicJobRunner
from indexer import create_music_index

app = Flask(__name__)

# ==========================================
# 1. 설정 및 초기화
# ==========================================
basedir = os.path.abspath(os.path.dirname(__file__))

# JWT 설정 (서버 재시작 시 로그아웃 방지를 위해 고정 키 사용)
app.config['JWT_SECRET_KEY'] = os.environ.get('SECRET_KEY') or "my-super-secret-fixed-key"
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=7)

# [핵심] 폴더 설정
USERS_BASE_FOLDER = os.path.join(basedir, 'users')
DEFAULTS_FOLDER = os.path.join(basedir, 'defaults')
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

# [수정] 인증 코드 저장소 통일 (email_codes 삭제함)
VERIFICATION_CODES = {}
VERIFICATION_LOCK = threading.Lock()

# 백그라운드 작업 실행기
bg_runner = BackgroundMusicJobRunner(app.config['USERS_FOLDER'])


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

    # [수정] VERIFICATION_CODES 사용으로 통일
    with VERIFICATION_LOCK:
        VERIFICATION_CODES[email] = {'code': code, 'timestamp': time.time() + 300}  # 5분 유효

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
        # 여기서는 삭제하지 않음 (회원가입/비번변경 때 한 번 더 확인하거나 그때 삭제)
        return jsonify({'message': 'Verified'}), 200
    return jsonify({'message': 'Invalid code'}), 400


# ==========================================
# 5. 인증 API (Register / Login)
# ==========================================
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    code = data.get('code')

    if not all([username, password, email, code]):
        return jsonify({'message': '필수 정보가 누락되었습니다.'}), 400

    # 1. [수정] 이메일 인증 코드 검증 (VERIFICATION_CODES 사용)
    with VERIFICATION_LOCK:
        server_data = VERIFICATION_CODES.get(email)

    if not server_data or str(server_data['code']) != str(code):
        return jsonify({'message': '인증 코드가 일치하지 않거나 만료되었습니다.'}), 400

    # 2. DB 중복 검사
    if User.query.filter_by(username=username).first():
        return jsonify({'message': '이미 사용 중인 아이디입니다.'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'message': '이미 가입된 이메일입니다.'}), 409

    try:
        # 3. 사용자 생성
        new_user = User(username=username, email=email, public_id=str(uuid.uuid4()))
        new_user.set_password(password)
        db.session.add(new_user)

        # 4. 폴더 생성
        user_folder = os.path.join(USERS_BASE_FOLDER, new_user.public_id)  # 보안상 public_id(UUID) 사용 권장
        os.makedirs(user_folder, exist_ok=True)

        db.session.commit()

        # 5. 사용된 코드 삭제
        with VERIFICATION_LOCK:
            if email in VERIFICATION_CODES: del VERIFICATION_CODES[email]

        return jsonify({'message': '회원가입 성공'}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Server Error: {str(e)}'}), 500


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
# 6. 파일 서빙 및 업로드
# ==========================================

@app.route('/files/<username>/<book_folder>/music/<filename>')
@jwt_required(optional=True)
def serve_music_file(username, book_folder, filename):
    try:
        if not _is_safe_segment(filename): return "Access Denied", 403
        base_music_dir = app.config['DEFAULTS_MUSIC_FOLDER']

        if os.path.exists(os.path.join(base_music_dir, filename)):
            return send_from_directory(base_music_dir, filename)

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
        return "File Not Found", 404


@app.route('/files/<username>/<book_folder>/<filename>')
@jwt_required(optional=True)
def serve_root_file(username, book_folder, filename):
    user_uuid = _resolve_user_uuid(username)
    if not user_uuid: return "Unauthorized", 403

    if not (_is_safe_segment(book_folder) and _is_safe_segment(filename)):
        return "Access Denied", 403

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

        # 저장 경로
        save_dir = os.path.join(USERS_BASE_FOLDER, current_user_uuid, book_folder_name)
        music_folder = app.config['DEFAULTS_MUSIC_FOLDER']

        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        pdf_path = os.path.join(save_dir, filename_safe)
        file.save(pdf_path)

        client_username = _resolve_client_username(current_user_uuid)
        web_path_prefix = f"/files/{client_username}/{book_folder_name}"

        # ---------------------------------------------------------
        # [삭제됨] 여기서 create_music_index() 호출하던 것 제거!
        # 책(PDF)만 올렸는데 음악 인덱싱을 돌리는 건 자원 낭비이자 논리 오류임.
        # ---------------------------------------------------------

        # 작업 등록 (여기서 분석 -> 음악생성 -> 인덱싱 순으로 처리됨)
        job_id = bg_runner.enqueue(
            job_type='analyze',
            username=current_user_uuid,
            book_id=book_folder_name,
            pdf_path=pdf_path,
            book_root_folder=save_dir,
            music_folder=music_folder,
            web_path_prefix=web_path_prefix
        )

        return jsonify({
            'message': 'Upload successful. Processing started.',
            'job_id': job_id,
            'book_title': book_folder_name
        }), 202

    return jsonify({'message': 'Invalid file type'}), 400


# ==========================================
# 7. 기타 조회 및 관리 API
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
    json_filename = os.path.splitext(filename)[0] + "_full.json"

    try:
        user_base = _safe_join(app.config['USERS_FOLDER'], user_uuid)
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
            for ch in book_data.get('chapters', []) or []:
                for seg in ch.get('segments', []) or []:
                    fn = seg.get('music_filename')
                    if fn: files.add(fn.strip())
        except:
            pass
    return jsonify({'files': sorted(files)}), 200


@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"}), 200


@app.route('/my_books', methods=['POST'])
@jwt_required()
def get_my_books():
    current_uuid = get_jwt_identity()
    user_dir = os.path.join(app.config['USERS_FOLDER'], current_uuid)
    books = []
    if os.path.exists(user_dir):
        for folder_name in os.listdir(user_dir):
            book_path = os.path.join(user_dir, folder_name)
            if os.path.isdir(book_path):
                cover_url = ""
                if os.path.exists(os.path.join(book_path, f"{folder_name}.png")):
                    client_username = _resolve_client_username(current_uuid)
                    cover_url = f"/files/{client_username}/{folder_name}/{folder_name}.png"
                books.append({"title": folder_name, "cover_url": cover_url})
    return jsonify({"books": books}), 200


@app.route('/delete_server_book', methods=['POST'])
@jwt_required()
def delete_server_book():
    current_uuid = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    book_title = data.get('book_title')
    if not book_title: return jsonify({'message': 'Book title required'}), 400
    target_dir = os.path.join(app.config['USERS_FOLDER'], current_uuid, book_title)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
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
    """
    [수정 완료] DB 처리 로직을 SQLAlchemy 방식에 맞게 수정 + 인증 코드 검증 강화
    """
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')
    new_password = data.get('new_password')

    if not all([email, code, new_password]):
        return jsonify({'message': '필수 정보가 누락되었습니다.'}), 400

    # 1. 인증 코드 검증 (VERIFICATION_CODES 사용)
    with VERIFICATION_LOCK:
        server_data = VERIFICATION_CODES.get(email)

    if not server_data or str(server_data['code']) != str(code):
        return jsonify({'message': '인증 코드가 일치하지 않거나 만료되었습니다.'}), 400

    if len(new_password) < 8:
        return jsonify({'message': '비밀번호는 최소 8자 이상이어야 합니다.'}), 400

    # 2. [수정] DB에서 사용자 찾기 (딕셔너리 순회 X -> DB 쿼리 O)
    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({'message': '가입된 이메일이 아닙니다.'}), 404

    # 3. 비밀번호 변경 및 저장
    try:
        user.set_password(new_password)
        db.session.commit()

        # 사용된 코드 삭제
        with VERIFICATION_LOCK:
            if email in VERIFICATION_CODES: del VERIFICATION_CODES[email]

        return jsonify({'message': '비밀번호가 성공적으로 변경되었습니다.'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Database Error: {str(e)}'}), 500


def _start_periodic_execute():
    if os.environ.get("ENABLE_PERIODIC_EXECUTE", "1") != "1": return
    interval = int(os.environ.get("EXECUTE_INTERVAL_SECONDS", "2"))
    max_jobs = int(os.environ.get("MAX_JOBS_PER_RUN", "5"))

    def _loop():
        # [로그 추가] 시작됨을 알림
        print(f"🚀 [JobRunner] 백그라운드 워커 가동됨 (Interval: {interval}s)")
        while True:
            try:
                result = bg_runner.execute(max_jobs=max_jobs)
                if result.get("ran", 0) > 0:
                    print(f"🕒 [JobRunner] 작업 처리됨: {result}")
            except Exception as e:
                print(f"❌ execute() loop error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


print("🔥 서버 초기화 중...")
try:
    create_music_index()
except Exception as e:
    print(f"❌ 초기 인덱싱 실패: {e}")

_start_periodic_execute()

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug)