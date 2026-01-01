from server import VERIFICATION_CODES


def test_send_verification_code(client):
    """이메일 인증 코드 발송 테스트"""
    response = client.post('/send_code', json={'email': 'new@test.com'})
    assert response.status_code == 200
    assert 'new@test.com' in VERIFICATION_CODES


def test_register_flow(client):
    """인증 코드 검증 후 회원가입 성공 시나리오"""
    email = "register@test.com"

    # 1. 코드 요청
    client.post('/send_code', json={'email': email})
    code = VERIFICATION_CODES[email]['code']

    # 2. 틀린 코드로 가입 시도
    bad_res = client.post('/register', json={
        'username': 'user1',
        'password': 'pw',
        'email': email,
        'code': '000000'
    })
    assert bad_res.status_code == 400

    # 3. 올바른 코드로 가입
    good_res = client.post('/register', json={
        'username': 'user1',
        'password': 'pw',
        'email': email,
        'code': code
    })
    assert good_res.status_code == 201


def test_login_and_protected_route(client):
    """로그인 및 JWT 보호 라우트 접근 테스트"""
    email = "login@test.com"
    client.post('/send_code', json={'email': email})
    code = VERIFICATION_CODES[email]['code']

    client.post('/register', json={
        'username': 'loginuser',
        'password': 'validpass',
        'email': email,
        'code': code
    })

    # 로그인
    login_res = client.post('/login', json={
        'username': 'loginuser',
        'password': 'validpass'
    })
    assert login_res.status_code == 200
    token = login_res.get_json()['access_token']

    # [수정됨] 토큰 없이 접근 (GET -> POST로 변경)
    # /my_books는 POST만 허용하므로, GET을 보내면 405가 뜸.
    # POST를 보내야 토큰 검사를 수행하고 401을 뱉음.
    assert client.post('/my_books').status_code == 401

    # 토큰 포함 접근 -> 200 OK
    headers = {'Authorization': f'Bearer {token}'}
    assert client.post('/my_books', headers=headers).status_code == 200