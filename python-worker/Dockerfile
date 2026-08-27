FROM python:3.12-slim-bookworm

ARG REQUIREMENTS_FILE=requirements-worker.txt
ARG INSTALL_AUDIO_TOOLS=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/nltk_data

RUN set -eux; \
    if [ "${INSTALL_AUDIO_TOOLS}" = "true" ]; then \
        apt-get update; \
        apt-get install -y --no-install-recommends ffmpeg libgomp1 libsndfile1; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    groupadd --gid 10001 ebookstudio; \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin ebookstudio

WORKDIR /app
COPY requirements-analyzer.txt requirements-worker.txt requirements-worker-cpu.txt requirements-worker-cu130.txt ./
RUN python -m pip install --no-cache-dir -r "${REQUIREMENTS_FILE}" \
    && python -m nltk.downloader -d /opt/nltk_data wordnet omw-1.4 \
    && chown -R 10001:10001 /opt/nltk_data

COPY --chown=10001:10001 *.py ./
COPY --chown=10001:10001 defaults ./defaults

USER 10001:10001
ENTRYPOINT ["python", "/app/spring_worker.py"]
