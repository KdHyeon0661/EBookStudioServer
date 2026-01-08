import os
import time
from server import app, bg_runner


def run_worker():
    print("[Worker] Process started")

    with app.app_context():
        bg_runner.recover_stuck_jobs()

        while True:
            try:
                result = bg_runner.execute(max_jobs=1)

                if result.get("ran", 0) > 0:
                    print(f"[Worker] Job processed: {result}")
                else:
                    time.sleep(2)

            except Exception as e:
                print(f"[Worker] Error during execution: {e}")
                time.sleep(5)


if __name__ == "__main__":
    run_worker()