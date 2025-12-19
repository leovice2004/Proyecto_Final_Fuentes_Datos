
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst6 \
    libxi6 \
    xvfb \
    x11vnc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY NBA.csv .
COPY student_habits.xlsx .
COPY main.py .
COPY demograficos_2025.csv .

EXPOSE 5900

CMD sh -c "Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1 && x11vnc -display :1 -nopw -forever & python main.py"

