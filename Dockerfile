FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8004 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" -r requirements.txt

COPY . .

EXPOSE 8004

CMD ["streamlit", "run", "app/app.py", "--server.port=8004", "--server.address=0.0.0.0"]
