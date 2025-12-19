
FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    python3-tk \
    tk-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst6 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
COPY NBA.csv .
COPY student_habits.xlsx .
COPY main.py .
COPY demograficos_2025.csv .


RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "main.py"]
