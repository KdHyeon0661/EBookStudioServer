import time
import multiprocessing  # 프로세스 기반 병렬 처리를 위해 필수
import sys
import os

# Flask 앱 환경을 프로세스마다 불러오기 위해 필요
from server import app, db, Job
from background_music_jobs import BackgroundMusicJobRunner


def run_analysis_process():
    """
    [Fast Track] CPU 전용 워커
    - PDF 분석 및 기존 음악 매핑만 담당
    - 절대 GPU 모델(MusicGen)을 로드하지 않음
    """
    print(f"[Worker-CPU] 분석 전용 프로세스 시작 (PID: {os.getpid()})")

    # 프로세스별 DB 세션 격리
    with app.app_context():
        runner = BackgroundMusicJobRunner(app.config['USERS_FOLDER'], db, Job)

        while True:
            try:
                # 'analyze' 타입의 작업만 가져옴
                result = runner.execute(job_types=['analyze'], max_jobs=1)
                if result['ran'] == 0:
                    time.sleep(1)  # 대기
            except Exception as e:
                print(f"[Worker-CPU] Error: {e}")
                time.sleep(5)


def run_generation_process():
    """
    [Slow Track] GPU 전용 워커
    - 새로운 음악 생성 및 라이브러리 확장 담당
    - 분석 작업을 방해하지 않고 뒷단에서 천천히 동작
    """
    print(f"[Worker-GPU] 음악 생성 프로세스 시작 (PID: {os.getpid()})")

    with app.app_context():
        runner = BackgroundMusicJobRunner(app.config['USERS_FOLDER'], db, Job)

        # Stuck Jobs 복구는 한 곳에서만 수행
        runner.recover_stuck_jobs()

        while True:
            try:
                # 'music_generation' 타입의 작업만 가져옴
                result = runner.execute(job_types=['music_generation'], max_jobs=1)
                if result['ran'] == 0:
                    time.sleep(2)
            except Exception as e:
                print(f"[Worker-GPU] Error: {e}")
                time.sleep(5)


if __name__ == '__main__':
    # 멀티프로세싱 시작
    p_analyze = multiprocessing.Process(target=run_analysis_process, name="Analyzer")
    p_music = multiprocessing.Process(target=run_generation_process, name="Generator")

    p_analyze.start()
    p_music.start()

    p_analyze.join()
    p_music.join()