import os
import json
import librosa
import numpy as np
import warnings
import hashlib

# 경고 무시
warnings.filterwarnings('ignore')

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS_DIR = os.path.join(BASE_DIR, 'defaults')
MUSIC_FOLDER = os.path.join(DEFAULTS_DIR, 'music')
INDEX_FILE = os.path.join(DEFAULTS_DIR, 'music_index.json')


def analyze_audio(file_path):
    """
    오디오 전체 길이 및 BPM 측정 (최적화)
    """
    try:
        # 1. 길이 측정
        try:
            duration = librosa.get_duration(path=file_path)
        except:
            y_temp, sr_temp = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y_temp, sr=sr_temp)

        # 2. BPM 측정 (60초 샘플링)
        y, sr = librosa.load(file_path, sr=None, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        if isinstance(tempo, np.ndarray):
            tempo = tempo[0] if len(tempo) > 0 else 0.0

        return duration, float(tempo)
    except Exception as e:
        print(f"   ⚠️ 분석 실패 ({os.path.basename(file_path)}): {e}")
        return 0.0, 0.0


def create_music_index():
    print(f"🚀 Indexer 시작")
    print(f"📂 스캔 대상: {MUSIC_FOLDER} (하위 폴더 포함)")

    if not os.path.exists(MUSIC_FOLDER):
        os.makedirs(MUSIC_FOLDER, exist_ok=True)
        return

    # =========================================================
    # [수정됨] 하위 폴더(storage_xxx)까지 재귀 탐색 (os.walk)
    # =========================================================
    valid_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')

    # disk_files_map: { "파일명": "전체경로" }
    disk_files_map = {}

    for root, dirs, files in os.walk(MUSIC_FOLDER):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # 파일명 중복 방지를 위해 덮어쓰거나,
                # 해시 기반이라 중복이 없다고 가정
                full_path = os.path.join(root, file)
                disk_files_map[file] = full_path

    print(f"🔎 전체 오디오 파일 발견: {len(disk_files_map)}개")

    # 3. 기존 인덱스 로드
    index_data = {}
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        except:
            index_data = {}

    changed = False

    # 4. [정리] 디스크에서 삭제된 파일 제거
    ids_to_remove = []
    for key, val in index_data.items():
        # 인덱스에 있는 파일명이 현재 디스크 맵에 없으면 삭제 대상
        if val.get('filename') not in disk_files_map:
            ids_to_remove.append(key)

    if ids_to_remove:
        print(f"🧹 삭제된 파일 정리 중... ({len(ids_to_remove)}개 제거)")
        for key in ids_to_remove:
            del index_data[key]
        changed = True

    # 5. [추가] 신규 파일 인덱싱
    registered_filenames = {v['filename'] for v in index_data.values()}

    # 아직 등록되지 않은 파일명만 골라냄
    new_filenames = [f for f in disk_files_map.keys() if f not in registered_filenames]

    if new_filenames:
        print(f"🆕 신규 파일 {len(new_filenames)}개 분석 시작...")

    for i, filename in enumerate(new_filenames):
        print(f"   [{i + 1}/{len(new_filenames)}] {filename} ... ", end='', flush=True)

        # 전체 경로를 맵에서 가져옴 (하위 폴더 어딘가에 있음)
        file_path = disk_files_map[filename]
        json_path = os.path.splitext(file_path)[0] + ".json"

        # 메타데이터 기본값
        genre = "unknown"
        prompt = ""
        bpm = 0

        # 사이드카 JSON 확인 (생성 시 같이 만들어진 json)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    meta = json.load(jf)
                    genre = meta.get('genre') or meta.get('emotion') or genre
                    prompt = meta.get('original_prompt') or meta.get('prompt') or prompt
                    bpm = meta.get('bpm', 0)
            except:
                pass

        if genre == "unknown":
            parts = filename.split('_')
            if len(parts) > 1 and parts[1].isalpha():
                genre = parts[1]

        # 오디오 분석
        dur, detected_bpm = analyze_audio(file_path)
        if not bpm: bpm = detected_bpm

        # ID 생성
        file_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16) % 10000000
        while str(file_hash) in index_data:
            file_hash += 1

        index_data[str(file_hash)] = {
            "id": file_hash,
            "genre": genre,
            "filename": filename,  # 파일명만 저장 (서버가 검색해서 서빙함)
            "duration": round(dur, 2),
            "bpm": int(round(bpm)),
            "prompt": prompt if prompt else f"{genre} mood music"
        }
        changed = True
        print("Done.")

    # 6. 저장
    if changed:
        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 인덱싱 완료! (총 {len(index_data)}개)")
    else:
        print("✅ 변경 사항 없음.")


if __name__ == "__main__":
    create_music_index()