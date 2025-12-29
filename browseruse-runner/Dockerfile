FROM mcr.microsoft.com/playwright/python:v1.49.1-noble

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl ca-certificates xvfb x11vnc novnc websockify fluxbox fonts-unifont fonts-ubuntu fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY server.py entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/profiles", "/app/artifacts"]

EXPOSE 8000 5900 7900

CMD ["/app/entrypoint.sh"]
