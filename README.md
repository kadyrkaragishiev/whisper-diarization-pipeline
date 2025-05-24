# 🎤 Whisper Diarization Pipeline

**Коробочное решение** для автоматической транскрипции и диаризации аудиофайлов с поддержкой NVIDIA GPU.

> 🚀 **Один скрипт устанавливает всё!** Просто запустите `./setup-complete.sh` и система готова к работе.

## 🚀 Быстрый старт (Коробочное решение)

### Требования
- Docker Desktop (Windows/macOS) или Docker Engine (Linux)
- NVIDIA GPU (опционально, для ускорения - поддерживается RTX 3080!)
- Windows 10+, macOS 10.14+, или Linux

### Установка и запуск

1. **Клонируйте репозиторий:**
```bash
git clone <repository-url>
cd whisper-diarization-pipeline
```

2. **Запустите универсальный скрипт:**

**Linux/macOS:**
```bash
./setup-complete.sh  # Полная установка
./run.sh             # Показать справку
```

**Windows (PowerShell как администратор):**
```powershell
.\setup-complete.ps1  # Полная установка  
.\run.ps1             # Показать справку
```

Скрипт автоматически:
- Определит наличие NVIDIA GPU
- Установит необходимые зависимости
- Соберет Docker образ
- Покажет справку по использованию

### Использование

1. **Поместите аудиофайл в директорию `input/`:**
```bash
cp your_audio.wav input/
```

2. **Запустите обработку:**
```bash
# Базовая обработка
./run.sh your_audio.wav

# С дополнительными параметрами
./run.sh your_audio.wav --model large --max-speakers 5

# Кастомная модель
./run.sh your_audio.wav --custom-model path/to/model
```

3. **Результаты будут в директории `output/`:**
- `filename_transcript.txt` - полная транскрипция
- `filename_segments.csv` - сегменты с временными метками и спикерами
- `filename_result.json` - полные результаты в JSON формате

## 🔧 Конфигурация

### Переменные окружения (.env файл)

```bash
# HuggingFace токен для PyAnnote модели диаризации
HUGGINGFACE_TOKEN=your_token_here

# Директория для локальных моделей
LOCAL_MODELS_DIR=/app/models
```

### Поддерживаемые параметры

- `--model` - Модель Whisper (tiny, base, small, medium, large)
- `--custom-model` - Путь к кастомной модели или HuggingFace model ID
- `--output` - Директория для результатов
- `--device` - Устройство (cpu, cuda, mps)
- `--min-speakers` - Минимальное количество спикеров
- `--max-speakers` - Максимальное количество спикеров
- `--min-segment` - Минимальная длительность сегмента (сек)
- `--alignment-strategy` - Стратегия совмещения (strict, smart, aggressive)
- `--time-limit` - Ограничение времени транскрипции (сек)

## 🐳 Docker варианты

### GPU версия (рекомендуется)
```bash
docker-compose --profile gpu up whisper-diarization-gpu
```

### CPU версия
```bash
docker-compose --profile cpu up whisper-diarization-cpu
```

### Автоматическое определение
```bash
docker-compose up whisper-diarization
```

## 🛠️ Ручная установка (без Docker)

### Требования
- Python 3.10+
- FFmpeg
- CUDA 12.1+ (для GPU)

### Установка зависимостей

**Для NVIDIA GPU:**
```bash
pip install -r requirements.txt
```

**Для CPU:**
```bash
pip install -r requirements-cpu.txt
```

### Запуск
```bash
python main.py input/audio.wav --model large
```

## 📁 Структура проекта

```
whisper-diarization-pipeline/
├── main.py                 # Основной скрипт
├── run.sh                  # Универсальный лаунчер
├── requirements.txt        # Зависимости для GPU
├── requirements-cpu.txt    # Зависимости для CPU
├── Dockerfile             # Docker образ для GPU
├── Dockerfile.cpu         # Docker образ для CPU
├── docker-compose.yml     # Docker Compose конфигурация
├── download_models.py     # Скрипт загрузки моделей
├── setup_russian_model.py # Настройка русской модели
├── model-converter.py     # Конвертер моделей
├── env.example           # Пример .env файла
├── input/                # Входные аудиофайлы
├── output/               # Результаты обработки
└── models/               # Локальные модели
```

## 🔍 Поддерживаемые форматы

**Аудио:** WAV, MP3, M4A, FLAC, OGG, WMA
**Видео:** MP4, AVI, MKV, MOV (извлекается аудио)

## ⚡ Производительность

- **GPU (NVIDIA):** ~10-20x быстрее CPU
- **CPU:** Медленнее, но работает везде
- **Память:** 4-8GB RAM, 2-4GB VRAM (GPU)

## 🐛 Устранение неполадок

### NVIDIA GPU не обнаружена
1. Установите NVIDIA драйверы
2. Установите NVIDIA Docker runtime:
```bash
# Ubuntu/Debian
sudo apt install nvidia-docker2
sudo systemctl restart docker
```

### Ошибки с моделями
1. Проверьте HuggingFace токен в .env
2. Скачайте модели локально:
```bash
python download_models.py
```

### Проблемы с памятью
- Используйте меньшую модель Whisper (base вместо large)
- Уменьшите max-speakers
- Добавьте time-limit для длинных файлов

## 📝 Лицензия

MIT License

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Commit изменения
4. Push в branch
5. Создайте Pull Request

## 📚 Документация

- **[📖 QUICKSTART.md](QUICKSTART.md)** - Быстрый старт за 2 минуты
- **[🛠️ INSTALLATION.md](INSTALLATION.md)** - Подробная установка для разных ОС
- **[🪟 WINDOWS.md](WINDOWS.md)** - Специальное руководство для Windows с RTX 3080
- **[📝 PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Технические детали и архитектура

## 📞 Поддержка

- GitHub Issues для багов и предложений
- Документация в папке `docs/`
- Примеры в `examples/`