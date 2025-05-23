# Используем официальный Python образ с поддержкой ARM64 для Apple Silicon
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем основной код
COPY main.py .

# Создаем директории для данных
RUN mkdir -p /app/input /app/output

# Устанавливаем переменные среды
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Создаем точку входа
ENTRYPOINT ["python", "main.py"]

# По умолчанию показываем справку
CMD ["--help"] 