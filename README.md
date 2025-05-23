# 🎵 Whisper + PyAnnote Audio Pipeline

Пайплайн для транскрипции и диаризации аудиофайлов с использованием OpenAI Whisper и PyAnnote Audio.

## 🚀 Возможности

- **Транскрипция аудио** с помощью OpenAI Whisper
- **Диаризация спикеров** с помощью PyAnnote Audio  
- **Совмещение результатов** - текст с привязкой к спикерам
- **Поддержка различных форматов** аудио (wav, mp3, m4a, flac и др.)
- **Оптимизация для Apple Silicon** (Mac M1/M2)
- **Контейнеризация** с Docker
- **Множественные форматы вывода** (JSON, CSV, TXT)
- **🔒 Офлайн режим** - работа без HuggingFace токена

## 📋 Требования

- Python 3.8+
- macOS (оптимизировано для Apple Silicon M1/M2)
- HuggingFace токен для модели диаризации (опционально, если используете локальные модели)
- Docker (опционально)

## 🛠 Установка

### Способ 1: Локальная установка (рекомендуется)

1. **Клонируйте репозиторий:**
   ```bash
   git clone <your-repo-url>
   cd whisper-diarization-pipeline
   ```

2. **Запустите скрипт установки:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Активируйте виртуальное окружение:**
   ```bash
   source venv/bin/activate
   ```

### Способ 2: Ручная установка

1. **Создайте виртуальное окружение:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Установите зависимости:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Создайте директории:**
   ```bash
   mkdir -p input output
   ```

## 🔑 Настройка (два варианта)

### Вариант A: С HuggingFace токеном (онлайн)

Для работы диаризации спикеров необходим токен HuggingFace:

1. **Получите токен:**
   - Зайдите на https://huggingface.co/settings/tokens
   - Создайте новый токен с правами на чтение
   - Примите лицензию модели: https://huggingface.co/pyannote/speaker-diarization-3.1

2. **Настройте токен:**
   ```bash
   # Способ 1: Через .env файл
   cp env.example .env
   # Отредактируйте .env и добавьте ваш токен
   
   # Способ 2: Через переменную среды
   export HUGGINGFACE_TOKEN=your_token_here
   ```

### Вариант B: Без токена (офлайн) 🔒

**Более безопасный и быстрый способ!**

1. **Скачайте модели один раз:**
   ```bash
   source venv/bin/activate
   export HUGGINGFACE_TOKEN=your_token_here  # только для скачивания
   python download_models.py
   ```

2. **Используйте без токена:**
   ```bash
   unset HUGGINGFACE_TOKEN  # удаляем токен
   export LOCAL_MODELS_DIR=models
   python main.py input/audio.wav  # работает без токена!
   ```

📖 **Подробная инструкция**: [OFFLINE_USAGE.md](OFFLINE_USAGE.md)

## 🎯 Использование

### Базовое использование

```bash
# Активируйте виртуальное окружение
source venv/bin/activate

# Поместите аудиофайл в папку input/
cp your_audio.wav input/

# С токеном (онлайн)
python main.py input/your_audio.wav

# Без токена (офлайн)
python main.py input/your_audio.wav --local-models models/
```

### Расширенные опции

```bash
# Выбор модели Whisper
python main.py input/audio.wav --model large

# Указание директории вывода
python main.py input/audio.wav --output results/

# Офлайн режим с локальными моделями
python main.py input/audio.wav --local-models models/

# Комбинирование опций
python main.py input/audio.wav --model medium --local-models models/ --output results/

# Помощь по параметрам
python main.py --help
```

### Доступные модели Whisper

- `tiny` - самая быстрая, низкое качество
- `base` - сбалансированная (по умолчанию)
- `small` - хорошее качество
- `medium` - высокое качество
- `large` - максимальное качество, медленная

## 🔒 Скачивание моделей для офлайн использования

```bash
# Скачать все модели
python download_models.py

# Скачать только нужные модели Whisper
python download_models.py --whisper-models "base,small"

# Скачать только модель диаризации
python download_models.py --skip-whisper

# Настроить размещение
python download_models.py --models-dir /path/to/models
```

## 🐳 Использование с Docker

### Сборка образа

```bash
# Настройте токен в .env файле (для скачивания моделей)
cp env.example .env
# Отредактируйте .env

# Соберите образ
docker-compose build
```

### Запуск

```bash
# Поместите аудиофайлы в папку input/
cp your_audio.wav input/

# С токеном
docker-compose run --rm whisper-diarization /app/input/your_audio.wav

# С локальными моделями (рекомендуется)
docker-compose run --rm whisper-diarization /app/input/your_audio.wav --local-models /app/models
```

## 📁 Структура проекта

```
whisper-diarization-pipeline/
├── main.py                 # Основной код пайплайна
├── download_models.py      # Скрипт скачивания моделей
├── requirements.txt        # Python зависимости
├── Dockerfile             # Конфигурация Docker
├── docker-compose.yml     # Docker Compose конфигурация
├── setup.sh              # Скрипт установки
├── env.example           # Пример переменных среды
├── README.md             # Документация
├── OFFLINE_USAGE.md      # Инструкция по офлайн использованию
├── input/                # Папка для входных аудиофайлов
├── output/               # Папка для результатов
├── models/               # Локальные модели (после скачивания)
└── venv/                 # Виртуальное окружение (создается автоматически)
```

## 📄 Форматы вывода

Результаты сохраняются в папке `output/` в трех форматах:

### 1. JSON (`*_result.json`)
Полные результаты в структурированном формате:
```json
{
  "audio_file": "example.wav",
  "transcription": "Полный текст...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Привет, как дела?",
      "speaker": "SPEAKER_00"
    }
  ],
  "language": "ru",
  "has_speaker_diarization": true
}
```

### 2. CSV (`*_segments.csv`)
Таблица с сегментами:
```csv
start,end,text,speaker
0.0,5.2,"Привет, как дела?",SPEAKER_00
5.2,10.1,"Отлично, спасибо!",SPEAKER_01
```

### 3. TXT (`*_transcript.txt`)
Читаемый формат с разбивкой по спикерам:
```
Транскрипция: example.wav
Язык: ru
Диаризация: Да

=== ТРАНСКРИПЦИЯ ПО СПИКЕРАМ ===

[SPEAKER_00]:
00:00 - Привет, как дела?

[SPEAKER_01]:
00:05 - Отлично, спасибо!
```

## ⚡ Производительность

### Рекомендации для Mac M1/M2:

- Используйте модель `base` для повседневной работы
- Модель `large` дает лучшее качество, но работает медленнее
- Диаризация добавляет ~30% времени обработки
- Для коротких файлов (<5 мин) модель `small` оптимальна
- **Офлайн режим быстрее** - модели загружаются мгновенно

### Примерное время обработки (Mac M1):

| Длительность аудио | Модель | Время обработки | Офлайн режим |
|-------------------|--------|-----------------|--------------|
| 1 минута | base | ~30 секунд | ~25 секунд |
| 1 минута | large | ~60 секунд | ~50 секунд |
| 10 минут | base | ~3 минуты | ~2.5 минуты |
| 10 минут | large | ~8 минут | ~7 минут |

## 🐛 Решение проблем

### Ошибка "HuggingFace token not found"
```bash
# Вариант 1: Проверьте токен
echo $HUGGINGFACE_TOKEN

# Вариант 2: Используйте локальные модели (рекомендуется)
python download_models.py
export LOCAL_MODELS_DIR=models
python main.py input/audio.wav --local-models models/
```

### Ошибка загрузки модели диаризации
1. Убедитесь, что токен корректный
2. Примите лицензию модели на HuggingFace
3. Проверьте интернет-соединение
4. **Или используйте офлайн режим** (см. [OFFLINE_USAGE.md](OFFLINE_USAGE.md))

### Проблемы с аудиоформатами
Пайплайн автоматически конвертирует аудио в WAV. Поддерживаются:
- WAV, MP3, M4A, FLAC, OGG, WMA

### Ошибки памяти
- Используйте меньшую модель Whisper (`tiny`, `base`)
- Разбейте длинные аудиофайлы на части
- Закройте другие приложения

## 📝 Примеры использования

### Обработка интервью (с диаризацией)
```bash
python main.py input/interview.wav --model medium --local-models models/
```

### Массовая обработка
```bash
for file in input/*.wav; do
    python main.py "$file" --model base --local-models models/
done
```

### Только транскрипция (без диаризации)
```bash
# Просто не указывайте токен или локальные модели
python main.py input/audio.wav
```

### Быстрая обработка
```bash
# Используйте tiny модель для быстрого результата
python main.py input/audio.wav --model tiny --local-models models/
```

## 🔒 Безопасность и приватность

### Офлайн режим
- ✅ Данные не покидают локальную машину
- ✅ Нет зависимости от интернета
- ✅ Токены не нужны в продакшене
- ✅ Модели кэшируются локально

### Продакшен deployment
```bash
# Настройка один раз
python download_models.py --models-dir /opt/whisper-models

# В продакшене
export LOCAL_MODELS_DIR=/opt/whisper-models
python main.py audio.wav  # токен не нужен!
```

## 🤝 Поддержка

Если у вас возникли проблемы:

1. Проверьте системные требования
2. Убедитесь, что все зависимости установлены
3. Попробуйте офлайн режим (см. [OFFLINE_USAGE.md](OFFLINE_USAGE.md))
4. Проверьте права доступа к файлам
5. Посмотрите логи ошибок

## 📄 Лицензия

Проект использует модели с различными лицензиями:
- OpenAI Whisper: MIT License
- PyAnnote Audio: MIT License
- Модели HuggingFace: см. соответствующие страницы моделей

---

*Разработано для эффективной работы на Apple Silicon (Mac M1/M2) с поддержкой офлайн режима* 