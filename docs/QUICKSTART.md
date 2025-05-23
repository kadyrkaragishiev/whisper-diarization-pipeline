# 🚀 Быстрый старт

## Установка за 3 шага

### 1. Автоматическая установка
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Настройка (выберите способ)

#### 🔒 Способ A: Без токена (рекомендуется)
```bash
# Получите токен ТОЛЬКО для скачивания моделей
# https://huggingface.co/settings/tokens
# Примите лицензию: https://huggingface.co/pyannote/speaker-diarization-3.1

export HUGGINGFACE_TOKEN=your_token_here
python download_models.py

# Теперь токен больше не нужен!
unset HUGGINGFACE_TOKEN
```

#### 🌐 Способ B: С токеном (онлайн)
```bash
cp env.example .env
# Отредактируйте .env и добавьте ваш токен
```

### 3. Запуск
```bash
source venv/bin/activate

# Без токена (офлайн, быстрее и безопаснее)
python main.py input/your_audio.wav --local-models models/

# Или с токеном (онлайн)
python main.py input/your_audio.wav
```

## Тестирование

```bash
# Активируйте виртуальное окружение
source venv/bin/activate

# Запустите тест
python test_pipeline.py

# Если тест прошел успешно, запустите на тестовом файле
python main.py input/test_audio.wav --model tiny --local-models models/
```

## Преимущества офлайн режима 🔒

✅ **Быстрее** - модели загружаются мгновенно  
✅ **Безопаснее** - токены не нужны в продакшене  
✅ **Надежнее** - работает без интернета  
✅ **Приватнее** - данные не покидают машину  

## Docker (альтернативный способ)

```bash
# Настройте .env файл с токеном (только для скачивания)
cp env.example .env

# Соберите и запустите
docker-compose build
docker-compose run --rm whisper-diarization /app/input/your_audio.wav --local-models /app/models
```

## Результаты

Результаты будут в папке `output/`:
- `*_result.json` - полные данные
- `*_segments.csv` - таблица сегментов
- `*_transcript.txt` - читаемый текст

## Поддерживаемые форматы

WAV, MP3, M4A, FLAC, OGG, WMA

## Модели Whisper

- `tiny` - быстро, базовое качество
- `base` - сбалансированно (рекомендуется)
- `small` - хорошее качество
- `medium` - высокое качество
- `large` - максимальное качество

## Полезные команды

```bash
# Скачать только нужные модели
python download_models.py --whisper-models "base,small"

# Использовать переменную среды
export LOCAL_MODELS_DIR=models
python main.py input/audio.wav

# Быстрая обработка
python main.py input/audio.wav --model tiny --local-models models/

# Высокое качество
python main.py input/audio.wav --model large --local-models models/
```

---

💡 **Проблемы?** Смотрите полную документацию в [README.md](README.md)  
🔒 **Офлайн режим**: [OFFLINE_USAGE.md](OFFLINE_USAGE.md) 