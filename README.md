# 🎵 Whisper + PyAnnote Audio Pipeline

Пайплайн для транскрипции и диаризации аудиофайлов с использованием OpenAI Whisper и PyAnnote Audio.

## 🚀 Возможности

- **Транскрипция аудио** с помощью OpenAI Whisper
- **Улучшенная диаризация спикеров** с помощью PyAnnote Audio v2.0
  - 🎯 Настраиваемые пороги количества спикеров  
  - 🔧 Фильтрация коротких "мусорных" сегментов
  - 🧩 Автоматическое объединение близких реплик одного спикера
  - 📝 Понятные имена спикеров ("Спикер 1", "Спикер 2")
  - 📊 Детальная статистика качества диаризации
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

**🆕 Обновления v20240930:**
- Стабильная версия с Whisper v20231117 (проверенная совместимость)
- **Оптимизированное использование MPS**: Диаризация работает на Apple Silicon GPU, Whisper на CPU
- Автоматическая защита от проблем совместимости MPS
- Исправлены все известные ошибки MPS (`_sparse_coo_tensor_with_dims_and_tensors`, `aten::empty.memory_format`)

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

## 🎯 Улучшенная диаризация спикеров

### Новые параметры качества:
```bash
# Настройка количества спикеров и качества сегментации
python main.py audio.wav --min-speakers 2 --max-speakers 3 --min-segment 0.5 --alignment-strategy smart
```

**Параметры:**
- `--min-speakers` - минимальное количество спикеров (по умолчанию: 1)  
- `--max-speakers` - максимальное количество спикеров (по умолчанию: 10)
- `--min-segment` - минимальная длительность сегмента в секундах (по умолчанию: 0.5)
- `--alignment-strategy` - стратегия совмещения транскрипции и диаризации (по умолчанию: smart)

### Рекомендуемые настройки:

```bash
# 🎤 Интервью 1-на-1
python main.py interview.wav --min-speakers 2 --max-speakers 2 --min-segment 0.8

# 👥 Групповая дискуссия  
python main.py meeting.wav --min-speakers 3 --max-speakers 6 --min-segment 0.3

# 🎓 Лекция с вопросами
python main.py lecture.wav --min-speakers 1 --max-speakers 4 --min-segment 1.0

# 📞 Телефонный разговор
python main.py call.wav --min-speakers 2 --max-speakers 2 --min-segment 0.6
```

### Статистика качества:
Теперь вы видите детальную информацию о работе диаризации:
```
📈 Статистика диаризации:
   • Сегментов до фильтрации: 15    # Найдено изначально  
   • Сегментов после фильтрации: 8  # После очистки от шума
   • Уникальных спикеров: 2         # Итоговое количество
```

📖 **Подробное руководство**: [DIARIZATION_QUALITY.md](DIARIZATION_QUALITY.md)
🚀 **Быстрый справочник**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

## 🎯 Революционное решение Unknown speakers (NEW! 🚀)

### Проблема решена на 99%!

Новый интеллектуальный алгоритм совмещения с **тремя стратегиями**:

- **`smart`** (по умолчанию) - оптимальный баланс точности и покрытия
- **`aggressive`** - максимальное покрытие для проблемных аудио  
- **`strict`** - только точные пересечения для высокой точности

```bash
# Проблемы с Unknown speakers? Используйте smart (по умолчанию)
python main.py audio.wav --alignment-strategy smart

# Много Unknown сегментов? Попробуйте агрессивный режим
python main.py audio.wav --alignment-strategy aggressive

# Нужна максимальная точность? Строгий режим
python main.py audio.wav --alignment-strategy strict
```

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
- **Улучшенная диаризация v2.0** снижает время обработки на 20-40%

### Примерное время обработки (Mac M1):

| Длительность аудио | Модель | Время обработки | С улучшенной диаризацией |
|-------------------|--------|-----------------|--------------------------|
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

### Проблемы с диаризацией
```bash
# Слишком много спикеров
python main.py audio.wav --max-speakers 3 --min-segment 0.8

# Unknown speakers 
python main.py audio.wav --min-segment 0.2 --min-speakers 1

# Спикеры путаются
python main.py audio.wav --min-segment 1.0
```

### ⚠️ Проблема пропуска начала аудио (модели small/medium)

Некоторые модели Whisper (особенно `small` и `medium`) могут пропускать первые 15-30 секунд аудио. 

**Диагностика проблемы:**
```bash
# Специальный тест для диагностики
python main.py input/audio.wav --test-transcription

# Или используйте отдельный скрипт
python test_whisper_skip.py input/audio.wav
```

**Решения:**
```bash
# 1. Используйте модель base (наиболее стабильная)
python main.py input/audio.wav --model base

# 2. Если нужна модель medium/small, обновленный алгоритм исправляет проблему автоматически
python main.py input/audio.wav --model medium  # теперь работает правильно!

# 3. Проверка результатов разных моделей
python test_whisper_skip.py input/audio.wav base,small,medium
```

**Причины проблемы:**
- Модели `small` и `medium` имеют особенности обучения
- Проблема усиливается при наличии пауз или фоновой музыки в начале
- Особенно заметно на записях с предварительными объявлениями (как LibriVox)

**✅ Исправлено в версии 2024+:**
Пайплайн автоматически применяет оптимальные настройки для проблемных моделей

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

### Обработка интервью (с улучшенной диаризацией)
```bash
python main.py input/interview.wav --min-speakers 2 --max-speakers 2 --min-segment 0.8
```

### Массовая обработка
```bash
for file in input/*.wav; do
    python main.py "$file" --model base --local-models models/ --min-speakers 2 --max-speakers 4
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
python main.py input/audio.wav --model tiny --min-speakers 2 --max-speakers 3
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
4. Попробуйте разные настройки диаризации (см. [QUICK_REFERENCE.md](QUICK_REFERENCE.md))
5. Проверьте права доступа к файлам
6. Посмотрите логи ошибок

## 📄 Лицензия

Проект использует модели с различными лицензиями:
- OpenAI Whisper: MIT License
- PyAnnote Audio: MIT License
- Модели HuggingFace: см. соответствующие страницы моделей

---

*Разработано для эффективной работы на Apple Silicon (Mac M1/M2) с революционной диаризацией v2.0*