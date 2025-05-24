# 🪟 Windows Support

## 🚀 Быстрый старт для Windows с RTX 3080

### Автоматическая установка (один скрипт):

```powershell
# 1. Откройте PowerShell как администратор
# 2. Клонируйте репозиторий
git clone <repository-url>
cd whisper-diarization-pipeline

# 3. Запустите автоустановку
.\setup-complete.ps1

# 4. Используйте
copy your_audio.wav input\
.\run.ps1 your_audio.wav
```

**Готово!** RTX 3080 будет использована автоматически для ускорения.

## 🔧 Что установит скрипт

- ✅ **WSL2** (для лучшей производительности Docker)
- ✅ **Docker Desktop** (через winget или скачивание)
- ✅ **NVIDIA GPU поддержка** (автоматическое определение RTX 3080)
- ✅ **Все зависимости** и настройки

## 💻 Варианты запуска

### 1. PowerShell (рекомендуется)
```powershell
.\run.ps1 audio.wav -Model large -MaxSpeakers 3
```

### 2. WSL2 (максимальная производительность)
```bash
# В WSL2 терминале
./run.sh audio.wav --model large --max-speakers 3
```

### 3. Git Bash
```bash
# В Git Bash
./run.sh audio.wav --model large --max-speakers 3
```

### 4. Простой Batch файл (для новичков)
```batch
REM Просто двойной клик по run.bat или:
run.bat audio.wav --model large --max-speakers 3
```

## 🎮 NVIDIA RTX 3080 оптимизация

### Автоматическое определение GPU:
- ✅ Скрипт автоматически обнаружит RTX 3080
- ✅ Соберет GPU-оптимизированный Docker образ
- ✅ Использует CUDA 12.1 для максимальной производительности

### Производительность:
- **CPU версия:** ~30-60 минут для 1 часа аудио
- **RTX 3080:** ~3-5 минут для 1 часа аудио (10-20x ускорение!)

### Модели для RTX 3080:
```powershell
# Быстрая обработка
.\run.ps1 audio.wav -Model base

# Максимальное качество (рекомендуется для RTX 3080)
.\run.ps1 audio.wav -Model large

# Кастомная модель
.\run.ps1 audio.wav -CustomModel "openai/whisper-large-v3"
```

## 🐛 Решение проблем

### Docker не запускается:
1. Убедитесь что Docker Desktop запущен
2. Проверьте WSL2: `wsl --status`
3. Перезапустите Docker Desktop

### NVIDIA GPU не обнаружена:
1. Обновите драйверы NVIDIA (версия 472.12+)
2. Включите WSL2 backend в Docker Desktop
3. Проверьте: `nvidia-smi`

### Медленная работа:
1. Используйте WSL2 вместо обычного PowerShell
2. Убедитесь что используется GPU версия
3. Увеличьте RAM для WSL2 в `.wslconfig`

### Ошибки прав доступа:
```powershell
# Запустите PowerShell как администратор
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## ⚡ Оптимальная конфигурация для RTX 3080

### Для ежедневного использования:
```powershell
.\run.ps1 audio.wav -Model large -MaxSpeakers 5 -MinSegment 0.3
```

### Для коротких файлов (<10 мин):
```powershell
.\run.ps1 audio.wav -Model large -MaxSpeakers 3 -MinSegment 0.5
```

### Для длинных файлов (>1 час):
```powershell
.\run.ps1 audio.wav -Model large -TimeLimit 3600 -MaxSpeakers 4
```

### Для максимального качества:
```powershell
.\run.ps1 audio.wav -Model large -AlignmentStrategy smart -MinSegment 0.2
```

## 📊 Мониторинг GPU

### Во время обработки:
```powershell
# В отдельном окне PowerShell
nvidia-smi -l 1  # Обновление каждую секунду
```

### Проверка использования:
- **Utilization:** должно быть 90-100%
- **Memory:** 4-8GB для больших моделей
- **Temperature:** обычно 60-80°C

## 🎯 Специфика Windows

### Пути к файлам:
```powershell
# Используйте обратные слеши
.\run.ps1 input\my_audio.wav

# Или универсальные слеши (работают тоже)
.\run.ps1 input/my_audio.wav
```

### Переменные окружения:
```powershell
# В .env файле
HUGGINGFACE_TOKEN=your_token_here
LOCAL_MODELS_DIR=C:\whisper-models

# Или через PowerShell
$env:HUGGINGFACE_TOKEN="your_token_here"
```

### Антивирус:
- Добавьте папку проекта в исключения
- Docker может показаться подозрительным некоторым антивирусам

## 🏆 Результат

С RTX 3080 на Windows у вас будет:
- ⚡ **Сверхбыстрая обработка** (10-20x ускорение)
- 🎯 **Высокое качество** транскрипции с моделью large
- 🔄 **Автоматическая диаризация** спикеров
- 🚀 **Простой запуск** одной командой

**Готово к продуктивной работе!** 🎉 