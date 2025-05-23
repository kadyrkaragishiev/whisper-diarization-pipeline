#!/bin/bash
# Скрипт для активации виртуального окружения и проверки зависимостей

echo "🔧 Настройка окружения для whisper-diarization-pipeline"

# Проверяем версию Python
PYTHON_VERSION=$(python3 --version 2>&1 | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+")
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || [ "$MAJOR_VERSION" -eq 3 -a "$MINOR_VERSION" -lt 9 ]; then
    echo "❌ Требуется Python 3.9+, найден: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python версия: $PYTHON_VERSION"

# Создаем виртуальное окружение если его нет
if [ ! -d "venv_py39" ]; then
    echo "📦 Создаем виртуальное окружение..."
    python3 -m venv venv_py39
fi

# Активируем окружение
echo "🔄 Активируем виртуальное окружение..."
source venv_py39/bin/activate

# Обновляем pip
echo "📥 Обновляем pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo "📥 Устанавливаем зависимости..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper pyannote.audio transformers librosa soundfile pydub python-dotenv click tqdm matplotlib pandas numpy scipy

echo "✅ Окружение готово!"
echo ""
echo "Для активации окружения в будущем используйте:"
echo "source venv_py39/bin/activate"
echo ""
echo "Для запуска пайплайна:"
echo "python main.py input/audio.wav --model tiny" 