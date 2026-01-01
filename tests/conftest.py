import pytest
import os
import sys
import tempfile

# 이제 server.py를 안전하게 임포트할 수 있습니다.
from server import app, db, VERIFICATION_CODES


@pytest.fixture
def client():
    db_fd, db_path = tempfile.mkstemp()
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
    app.config['TESTING'] = True
    app.config['JWT_SECRET_KEY'] = 'test-secret'

    with tempfile.TemporaryDirectory() as temp_users_dir:
        app.config['USERS_FOLDER'] = temp_users_dir

        with app.app_context():
            db.create_all()
            yield app.test_client()
            db.session.remove()
            db.drop_all()

    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def auth_headers(client):
    email = "test@example.com"
    client.post('/send_code', json={'email': email})
    code = VERIFICATION_CODES[email]['code']

    client.post('/register', json={
        'username': 'testuser',
        'password': 'password123',
        'email': email,
        'code': code
    })

    res = client.post('/login', json={
        'username': 'testuser',
        'password': 'password123'
    })
    token = res.get_json()['access_token']
    return {'Authorization': f'Bearer {token}'}