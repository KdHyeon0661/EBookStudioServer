import os
import time
import json
import random
import glob
import shutil
import uuid
from datetime import timedelta, datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

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

app.config['JWT_SECRET_KEY'] = os.environ.get('SECRET_KEY') or "my-super-secret-fixed-key"
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=7)

USERS_BASE_FOLDER = os.path.join(basedir, 'users')
DEFAULTS_FOLDER = os.path.join(basedir, 'defaults')
DEFAULTS_MUSIC_FOLDER = os.path.join(DEFAULTS_FOLDER, 'music')

if not os.path.exists(USERS_BASE_FOLDER): os.makedirs(USERS_BASE_FOLDER)
if not os.path.exists(DEFAULTS_FOLDER): os.makedirs(DEFAULTS_FOLDER)
if not os.path.exists(DEFAULTS_MUSIC_FOLDER): os.makedirs(DEFAULTS_MUSIC_FOLDER)

app.config['USERS_FOLDER'] = USERS_BASE_FOLDER
app.config['DEFAULTS_FOLDER'] = DEFAULTS_FOLDER
app.config['DEFAULTS_MUSIC_FOLDER'] = DEFAULTS_MUSIC_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
jwt = JWTManager(app)


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


class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.String(36), primary_key=True)
    type = db.Column(db.String(50), nullable=False)
    user_uuid = db.Column(db.String(50), index=True)
    book_id = db.Column(db.String(100))
    status = db.Column(db.String(20), default='queued', index=True)
    created_at = db.Column(db.Integer, default=lambda: int(time.time()))
    started_at = db.Column(db.Integer, nullable=True)
    finished_at = db.Column(db.Integer, nullable=True)
    error = db.Column(db.Text, nullable=True)
    json_path = db.Column(db.String(255), nullable=True)
    music_folder = db.Column(db.String(255), nullable=True)
    web_path_prefix = db.Column(db.String(255), nullable=True)
    pdf_path = db.Column(db.String(255), nullable=True)
    book_root_folder = db.Column(db.String(255), nullable=True)


class VerificationCode(db.Model):
    __tablename__ = 'verification_codes'
    email = db.Column(db.String(120), primary_key=True)
    code = db.Column(db.String(10), nullable=False)
    expires_at = db.Column(db.Float, nullable=False)


@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload["jti"]
    token = db.session.query(TokenBlocklist.id).filter_by(jti=jti).scalar()
    return token is not None


bg_runner = BackgroundMusicJobRunner(app.config['USERS_FOLDER'], db_instance=db, job_model=Job)

with app.app_context():
    db.create_all()
    # 서버 프로세스에서는 복구 로직 실행 X (worker가 수행)
    pass


# ==========================================
# 2. 유틸리티 함수
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
        if current_uuid and username_or_uuid == current_uuid:
            return current_uuid
        u = User.query.filter_by(username=username_or_uuid).first()
        if u: return u.public_id
        u_by_id = User.query.filter_by(public_id=username_or_uuid).first()
        if u_by_id: return u_by_id.public_id
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


def _verify_code_logic(email, code_input, delete_on_success=False):
    vc = VerificationCode.query.get(email)
    if not vc:
        return False, 'Request verification code first.'
    if time.time() > vc.expires_at:
        return False, 'Code expired.'
    if vc.code != str(code_input):
        return False, 'Invalid code.'
    if delete_on_success:
        try:
            db.session.delete(vc)
            db.session.commit()
        except:
            db.session.rollback()
    return True, None


def _safely_cleanup_old_versions(user_uuid, filename_base):
    user_base_path = os.path.join(app.config['USERS_FOLDER'], user_uuid)
    if not os.path.exists(user_base_path): return
    candidates = []
    try:
        for entry in os.scandir(user_base_path):
            if entry.is_dir() and entry.name.startswith(filename_base + "_"):
                candidates.append(entry.name)
    except Exception:
        return
    if not candidates: return

    active_jobs = Job.query.filter(
        Job.user_uuid == user_uuid,
        Job.book_id.in_(candidates),
        Job.status.in_(['queued', 'running'])
    ).all()
    active_folders = {job.book_id for job in active_jobs}

    for folder_name in candidates:
        if folder_name not in active_folders:
            try:
                target_path = os.path.join(user_base_path, folder_name)
                shutil.rmtree(target_path)
                print(f"[Cleanup] Deleted old version: {folder_name}")
            except Exception:
                pass


ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==========================================
# 3. 인증 API
# ==========================================

@app.route('/send_code', methods=['POST'])
def send_code():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    if not email: return jsonify({'message': 'Email is required'}), 400
    code = str(random.randint(100000, 999999))
    expires_at = time.time() + 300
    try:
        vc = VerificationCode.query.get(email)
        if vc:
            vc.code = code
            vc.expires_at = expires_at
        else:
            vc = VerificationCode(email=email, code=code, expires_at=expires_at)
            db.session.add(vc)
        db.session.commit()
        print(f"[Auth] Code sent to {email}: {code}")
        return jsonify({'message': 'Code sent'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'DB Error: {str(e)}'}), 500


@app.route('/verify_code', methods=['POST'])
def verify_code():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    code = data.get('code')
    if not email or not code: return jsonify({'message': 'Email and code required'}), 400
    success, msg = _verify_code_logic(email, code, delete_on_success=False)
    if success: return jsonify({'message': 'Verified'}), 200
    return jsonify({'message': msg}), 400


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    code = data.get('code')
    if not all([username, password, email, code]):
        return jsonify({'message': 'Missing required fields'}), 400
    success, msg = _verify_code_logic(email, code, delete_on_success=True)
    if not success: return jsonify({'message': msg}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'message': 'Email already exists'}), 409
    try:
        new_user = User(username=username, email=email, public_id=str(uuid.uuid4()))
        new_user.set_password(password)
        db.session.add(new_user)
        user_folder = os.path.join(USERS_BASE_FOLDER, new_user.public_id)
        os.makedirs(user_folder, exist_ok=True)
        db.session.commit()
        return jsonify({'message': 'Registered successfully'}), 201
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


@app.route('/find_id', methods=['POST'])
def find_id():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    user = User.query.filter_by(email=email).first()
    if user: return jsonify({'message': 'Success', 'username': user.username}), 200
    return jsonify({'message': 'Not found'}), 404


@app.route('/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    code = data.get('code')
    new_password = data.get('new_password')
    if not all([email, code, new_password]):
        return jsonify({'message': 'Missing required fields'}), 400
    success, msg = _verify_code_logic(email, code, delete_on_success=True)
    if not success: return jsonify({'message': msg}), 400
    if len(new_password) < 8:
        return jsonify({'message': 'Password too short'}), 400
    user = User.query.filter_by(email=email).first()
    if not user: return jsonify({'message': 'Email not found'}), 404
    try:
        user.set_password(new_password)
        db.session.commit()
        return jsonify({'message': 'Password changed successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'DB Error: {str(e)}'}), 500


# ==========================================
# 4. 파일 및 도서 API
# ==========================================

@app.route('/upload_book', methods=['POST'])
@jwt_required()
def upload_book():
    current_user_uuid = get_jwt_identity()
    if 'file' not in request.files: return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file and file.filename.lower().endswith('.pdf'):
        filename_safe = secure_filename(file.filename)
        filename_base = os.path.splitext(filename_safe)[0]

        upload_uuid = str(uuid.uuid4())[:8]
        book_folder_name = f"{filename_base}_{upload_uuid}"

        try:
            _safely_cleanup_old_versions(current_user_uuid, filename_base)
        except Exception as e:
            print(f"[Upload] Cleanup warning: {e}")

        save_dir = os.path.join(USERS_BASE_FOLDER, current_user_uuid, book_folder_name)
        music_folder = app.config['DEFAULTS_MUSIC_FOLDER']
        os.makedirs(save_dir, exist_ok=True)

        pdf_path = os.path.join(save_dir, filename_safe)
        file.save(pdf_path)

        client_username = _resolve_client_username(current_user_uuid)
        web_path_prefix = f"/files/{client_username}/{book_folder_name}"

        job_id = bg_runner.enqueue(
            job_type='analyze',
            user_uuid=current_user_uuid,
            book_id=book_folder_name,
            pdf_path=pdf_path,
            book_root_folder=save_dir,
            music_folder=music_folder,
            web_path_prefix=web_path_prefix
        )

        return jsonify({
            'message': 'Upload successful. Processing started.',
            'job_id': job_id,
            'book_title': filename_base,
            'book_folder': book_folder_name
        }), 202
    return jsonify({'message': 'Invalid file type'}), 400


@app.route('/files/<username>/<book_folder>/music/<filename>')
@jwt_required()
def serve_music_file(username, book_folder, filename):
    claims = get_jwt()
    if claims.get("username") != username:
        return jsonify({'message': 'Access Denied'}), 403
    try:
        if not _is_safe_segment(filename): return "Access Denied", 403
        base_music_dir = app.config['DEFAULTS_MUSIC_FOLDER']
        if os.path.exists(os.path.join(base_music_dir, filename)):
            return send_from_directory(base_music_dir, filename)
        for entry in os.scandir(base_music_dir):
            if entry.is_dir():
                target_path = os.path.join(entry.path, filename)
                if os.path.exists(target_path):
                    return send_from_directory(entry.path, filename)
        return "File Not Found", 404
    except Exception:
        return "File Not Found", 404


@app.route('/files/<username>/<book_folder>/<filename>')
@jwt_required()
def serve_root_file(username, book_folder, filename):
    claims = get_jwt()
    token_username = claims.get("username")
    user_uuid = get_jwt_identity()
    if token_username != username:
        return jsonify({'message': 'Access Denied'}), 403
    if not (_is_safe_segment(book_folder) and _is_safe_segment(filename)):
        return jsonify({'message': 'Invalid path'}), 400
    try:
        user_book_dir = _safe_join(app.config['USERS_FOLDER'], user_uuid, book_folder)
        return send_from_directory(user_book_dir, filename)
    except Exception:
        return jsonify({'message': 'File Not Found'}), 404


@app.route('/get_toc', methods=['POST'])
@jwt_required()
def get_toc():
    current_user = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    book_folder = data.get('book_folder')
    user_uuid = _resolve_user_uuid(username) if username else None
    if not user_uuid: return jsonify({'message': 'Unauthorized'}), 403
    try:
        user_base = _safe_join(app.config['USERS_FOLDER'], user_uuid)
        target_json_path = None
        if book_folder:
            folder_path = _safe_join(user_base, book_folder)
            candidates = glob.glob(os.path.join(folder_path, "*_full.json"))
            if candidates: target_json_path = candidates[0]
        else:
            filename = data.get('filename')
            if filename:
                base = os.path.splitext(filename)[0]
                patterns = [os.path.join(user_base, f"{base}_*", "*_full.json")]
                found = []
                for p in patterns: found.extend(glob.glob(p))
                if found:
                    found.sort(key=os.path.getmtime, reverse=True)
                    target_json_path = found[0]
        if not target_json_path or not os.path.exists(target_json_path):
            return jsonify({'toc': []}), 404
        with open(target_json_path, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        toc = [ch.get('title') for ch in book_data.get('chapters', []) if isinstance(ch, dict)]
        return jsonify({'toc': toc}), 200
    except Exception:
        return jsonify({'toc': []}), 500


@app.route('/list_music_files/<username>/<book_folder>', methods=['GET'])
@jwt_required()
def list_music_files(username, book_folder):
    user_uuid = _resolve_user_uuid(username)
    if not user_uuid: return jsonify({"message": "Unauthorized"}), 403
    book_dir = _safe_join(app.config['USERS_FOLDER'], user_uuid, book_folder)
    json_file = None
    if os.path.isdir(book_dir):
        candidates = glob.glob(os.path.join(book_dir, "*_full.json"))
        if candidates: json_file = candidates[0]
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


@app.route('/my_books', methods=['POST'])
@jwt_required()
def get_my_books():
    current_uuid = get_jwt_identity()
    user_dir = os.path.join(app.config['USERS_FOLDER'], current_uuid)
    books = []
    if os.path.exists(user_dir):
        all_folders = [
            os.path.join(user_dir, d) for d in os.listdir(user_dir)
            if os.path.isdir(os.path.join(user_dir, d))
        ]
        all_folders.sort(key=os.path.getmtime, reverse=True)
        client_username = _resolve_client_username(current_uuid)
        for book_path in all_folders:
            folder_name = os.path.basename(book_path)
            display_title = folder_name
            if "_" in folder_name:
                parts = folder_name.rsplit("_", 1)
                if len(parts[1]) >= 8:
                    display_title = parts[0]
            cover_url = ""
            covers = glob.glob(os.path.join(book_path, "*.png"))
            if covers:
                cover_name = os.path.basename(covers[0])
                cover_url = f"/files/{client_username}/{folder_name}/{cover_name}"
            books.append({
                "title": display_title,
                "folder": folder_name,
                "cover_url": cover_url
            })
    return jsonify({"books": books}), 200


@app.route('/delete_server_book', methods=['POST'])
@jwt_required()
def delete_server_book():
    current_uuid = get_jwt_identity()
    data = request.get_json(silent=True) or {}
    book_folder = data.get('book_folder')
    if not book_folder: return jsonify({'message': 'book_folder required'}), 400
    target_dir = os.path.join(app.config['USERS_FOLDER'], current_uuid, book_folder)
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            return jsonify({'message': 'Deleted successfully'}), 200
        except Exception as e:
            return jsonify({'message': f'Error: {str(e)}'}), 500
    else:
        return jsonify({'message': 'Book not found'}), 404


@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"}), 200


# ==========================================
# 5. 서버 실행
# ==========================================

print("[Init] Checking music index...")
try:
    create_music_index()
    print("[Init] Music index ready.")
except Exception as e:
    print(f"[Init] Indexing failed: {e}")

if __name__ == '__main__':
    print("[Main] Web Server Started")
    app.run(host='0.0.0.0', port=5000)