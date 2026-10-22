# Code sandbox image for the example agents: one throwaway container per tool
# call (see xeon_runtime/sandbox.py). Plain Python with the analysis and
# document libraries the demos need; no network at run time, so everything
# must be installed here.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir \
      numpy==2.3.3 pandas==2.3.3 pypdf==6.21.1 pytest==9.1.1 \
    && useradd --create-home --uid 10002 sandbox
WORKDIR /work
USER sandbox
CMD ["python3", "--version"]
