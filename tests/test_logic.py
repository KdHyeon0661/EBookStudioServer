import pytest
from unittest.mock import patch

# conftest.py 설정 덕분에 루트 경로에서 바로 import 가능
from defaults.genre_bpm_connector import pick_bpm, get_bpm_range
from background_music_jobs import create_dynamic_music_prompt, estimate_bpm_from_analysis


# --------------------------
# 1. BPM Connector Tests
# --------------------------
def test_pick_bpm():
    # 정상 범위
    bpm = pick_bpm('pop', default=100)
    assert 100 <= bpm <= 130

    # 정의되지 않은 장르
    bpm = pick_bpm('unknown_genre', default=999)
    assert bpm == 999


def test_get_bpm_range():
    assert get_bpm_range('chillout') == (60, 110)
    assert get_bpm_range('non_existent') is None


# --------------------------
# 2. Prompt Generation Tests
# --------------------------
def test_estimate_bpm_from_analysis():
    # 빠른 템포 키워드
    analysis_fast = {'emotion': 'anger', 'keywords': ['run', 'fast', 'speed']}
    bpm = estimate_bpm_from_analysis(analysis_fast)
    assert bpm > 140

    # 느린 템포 키워드
    analysis_slow = {'emotion': 'neutral', 'keywords': ['sleep', 'calm']}
    bpm = estimate_bpm_from_analysis(analysis_slow)
    assert bpm < 80


# [수정됨] 데코레이터 대신 with 구문 사용 (Pytest 충돌 방지)
def test_create_dynamic_music_prompt():
    analysis = {
        'emotion': 'joy',
        'keywords': ['sun', 'beach', 'dance']
    }

    # 모든 외부 의존성을 내부에서 안전하게 Mocking
    with patch('background_music_jobs.MUSIC_GENRES_200', {'pop': 'pop description'}), \
            patch('background_music_jobs.GENRE_BPM_CONNECTOR', {'pop': (100, 120)}), \
            patch('background_music_jobs.select_genre_dynamically', return_value='pop'), \
            patch('background_music_jobs.pick_bpm_from_genre_bpm_connector', return_value=110):
        prompt, genre, bpm = create_dynamic_music_prompt(analysis)

        assert genre == 'pop'
        assert bpm == 110
        assert "Mood:" in prompt
        assert "Genre: pop" in prompt