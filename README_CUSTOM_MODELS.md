# Поддержка кастомных моделей Whisper

Этот пайплайн теперь поддерживает использование кастомных моделей Whisper из HuggingFace в дополнение к стандартным моделям OpenAI Whisper.

## 🚀 Быстрый старт с русской моделью

### 1. Установка зависимостей
```bash
pip install transformers accelerate safetensors
```

### 2. Настройка русской модели
У вас уже есть папка `whisper-large-v3-russian-pt` с моделью. Запустите скрипт автоматической настройки:

```bash
python setup_russian_model.py
```

### 3. Использование
```bash
python main.py input.wav --custom-model models/custom_whisper/whisper-large-v3-russian
```

## 📖 Подробное руководство

### Поддерживаемые форматы моделей

1. **Стандартные модели Whisper** (OpenAI): tiny, base, small, medium, large
2. **Кастомные модели HuggingFace**: модели в формате transformers

### Установка кастомных моделей

#### Способ 1: Из HuggingFace Hub
```bash
# Скачать модель напрямую из HF Hub
python download_models.py --custom-whisper-model antony66/whisper-large-v3-russian

# Использовать
python main.py input.wav --custom-model antony66/whisper-large-v3-russian
```

#### Способ 2: Копирование локальной модели
```bash
# Скопировать вашу локальную модель в кеш
python download_models.py --copy-local-model whisper-large-v3-russian-pt --local-model-name whisper-large-v3-russian

# Использовать
python main.py input.wav --custom-model models/custom_whisper/whisper-large-v3-russian
```

#### Способ 3: Автоматическая настройка (для вашей модели)
```bash
python setup_russian_model.py
```

### Управление кастомными моделями

#### Просмотр доступных моделей
```bash
python download_models.py --list-custom
```

#### Структура директорий
```
models/
├── custom_whisper/           # Кастомные модели Whisper
│   ├── whisper-large-v3-russian/
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00002.safetensors
│   │   ├── model-00002-of-00002.safetensors
│   │   └── ...
│   └── другие_модели/
├── pyannote/                 # Модели диаризации
└── whisper/                  # Стандартные модели Whisper (кеш)
```

### Примеры использования

#### Стандартная модель
```bash
python main.py input.wav --model large
```

#### Кастомная локальная модель
```bash
python main.py input.wav --custom-model models/custom_whisper/whisper-large-v3-russian
```

#### Кастомная модель из HF Hub
```bash
python main.py input.wav --custom-model antony66/whisper-large-v3-russian
```

#### Полный пайплайн с диаризацией
```bash
python main.py input.wav \
  --custom-model models/custom_whisper/whisper-large-v3-russian \
  --min-speakers 2 \
  --max-speakers 5 \
  --alignment-strategy smart
```

### Преимущества кастомных моделей

1. **Специализация**: Модели, обученные на специфических данных (например, русский язык)
2. **Качество**: Потенциально лучшее качество для целевого языка/домена
3. **Контроль**: Возможность использования собственных обученных моделей
4. **Гибкость**: Поддержка различных архитектур и размеров

### Технические детали

#### Различия в обработке

**Стандартные модели:**
- Используют библиотеку `whisper`
- Прямой API для транскрипции
- Встроенная поддержка временных меток

**Кастомные модели:**
- Используют библиотеку `transformers`
- Требуют процессор для обработки аудио
- Упрощенная обработка временных меток (пока что)

#### Требования к памяти

- **Стандартные модели**: Как обычно для Whisper
- **Кастомные модели**: Могут требовать больше GPU памяти
- **Рекомендация**: Используйте `--device cpu` для больших моделей на слабом железе

### Устранение проблем

#### Ошибка "transformers не установлен"
```bash
pip install transformers accelerate
```

#### Ошибка загрузки модели на MPS (Apple Silicon)
```bash
python main.py input.wav --custom-model path/to/model --device cpu
```

#### Нет временных меток в кастомной модели
- Это временное ограничение
- Сегменты разбиваются по предложениям
- В будущих версиях будет улучшено

### Планы развития

- [ ] Улучшенная поддержка временных меток для кастомных моделей
- [ ] Автоматическое определение лучших параметров для каждой модели
- [ ] Поддержка моделей других архитектур
- [ ] Кеширование и оптимизация загрузки
- [ ] Батчевая обработка для больших файлов

### Вклад в проект

Если у вас есть идеи по улучшению поддержки кастомных моделей, создавайте issues или pull requests!

## 🤝 Благодарности

- [antony66/whisper-large-v3-russian](https://huggingface.co/antony66/whisper-large-v3-russian) - за отличную русскую модель Whisper
- HuggingFace transformers team - за гибкую библиотеку
- OpenAI - за оригинальный Whisper 