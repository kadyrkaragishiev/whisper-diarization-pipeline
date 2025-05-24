# 🚀 Быстрый старт

## Автоматическая установка (один скрипт)

### Linux/macOS:
```bash
# Клонируйте репозиторий
git clone <repository-url>
cd whisper-diarization-pipeline

# Запустите автоматическую установку
./setup-complete.sh
```

### Windows:
```powershell
# Клонируйте репозиторий
git clone <repository-url>
cd whisper-diarization-pipeline

# Запустите автоматическую установку (PowerShell как администратор)
.\setup-complete.ps1
```

Скрипт автоматически:
- ✅ Установит Docker и Docker Compose
- ✅ Настроит NVIDIA GPU поддержку (если доступно)
- ✅ Соберет Docker образы
- ✅ Создаст необходимые директории
- ✅ Настроит окружение

## Использование

### Linux/macOS:
```bash
# 1. Поместите аудиофайл в input/
cp your_audio.wav input/

# 2. Запустите обработку
./run.sh your_audio.wav

# 3. Результаты в output/
ls output/
```

### Windows:
```powershell
# 1. Поместите аудиофайл в input/
copy your_audio.wav input\

# 2. Запустите обработку
.\run.ps1 your_audio.wav

# 3. Результаты в output/
ls output\
```

## Примеры команд

### Linux/macOS:
```bash
# Базовая обработка
./run.sh audio.wav

# Большая модель для лучшего качества
./run.sh audio.wav --model large

# Ограничить количество спикеров
./run.sh audio.wav --max-speakers 3

# Кастомная модель
./run.sh audio.wav --custom-model path/to/model
```

### Windows:
```powershell
# Базовая обработка
.\run.ps1 audio.wav

# Большая модель для лучшего качества
.\run.ps1 audio.wav -Model large

# Ограничить количество спикеров
.\run.ps1 audio.wav -MaxSpeakers 3

# Кастомная модель
.\run.ps1 audio.wav -CustomModel path\to\model
```

## Что получите

- `filename_transcript.txt` - полная транскрипция
- `filename_segments.csv` - сегменты с временными метками
- `filename_result.json` - полные результаты в JSON

## Поддерживаемые форматы

**Аудио:** WAV, MP3, M4A, FLAC, OGG, WMA  
**Видео:** MP4, AVI, MKV, MOV (извлекается аудио)

## Проблемы?

1. **Docker не работает:** Перезагрузитесь или выполните `sudo usermod -aG docker $USER`
2. **GPU не обнаружена:** Установите NVIDIA драйверы
3. **Ошибки с моделями:** Добавьте HUGGINGFACE_TOKEN в .env файл

Подробная документация: [README.md](README.md) 