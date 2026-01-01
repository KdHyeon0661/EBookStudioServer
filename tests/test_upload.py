import io
from unittest.mock import patch


def test_upload_book(client, auth_headers):
    """PDF 파일 업로드 및 Job Enqueue 확인"""

    # 더미 PDF 파일 생성
    data = {
        'file': (io.BytesIO(b"%PDF-1.4...dummy content..."), 'test_book.pdf')
    }

    # BackgroundMusicJobRunner.enqueue 메소드를 Mocking하여 실제 작업 실행 방지
    with patch('server.bg_runner.enqueue') as mock_enqueue:
        mock_enqueue.return_value = "test-job-id"

        response = client.post(
            '/upload_book',
            headers=auth_headers,
            data=data,
            content_type='multipart/form-data'
        )

        assert response.status_code == 202
        assert response.get_json()['job_id'] == "test-job-id"

        # enqueue가 올바른 인자로 호출되었는지 확인
        mock_enqueue.assert_called_once()
        args, kwargs = mock_enqueue.call_args
        assert kwargs['job_type'] == 'analyze'
        assert kwargs['book_id'] == 'test_book'