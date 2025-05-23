#!/bin/bash

echo "🎵 Настройка Whisper + PyAnnote Audio Pipeline"

# Проверяем версию Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Версия Python: $python_version"

# Создаем виртуальное окружение
echo "📦 Создаем виртуальное окружение..."
python3 -m venv venv

# Активируем виртуальное окружение
echo "⚡ Активируем виртуальное окружение..."
source venv/bin/activate

# Обновляем pip
echo "🔧 Обновляем pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo "📥 Устанавливаем зависимости..."
pip install -r requirements.txt

# Создаем необходимые директории
echo "📁 Создаем директории..."
mkdir -p input output

# Копируем пример .env файла
if [ ! -f .env ]; then
    echo "📝 Создаем .env файл..."
    cp env.example .env
    echo "⚠️  Не забудьте добавить ваш HuggingFace токен в файл .env"
fi

echo "✅ Установка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Активируйте виртуальное окружение: source venv/bin/activate"
echo "2. Добавьте HuggingFace токен в файл .env"
echo "3. Поместите аудиофайлы в папку input/"
echo "4. Запустите: python main.py input/your_audio.wav"
echo ""
echo "🐳 Для использования Docker:"
echo "1. Добавьте HuggingFace токен в файл .env"
echo "2. Соберите образ: docker-compose build"
echo "3. Запустите: docker-compose run whisper-diarization /app/input/your_audio.wav" 