# 🔒 Работа без HuggingFace токена (Офлайн режим)

Этот гайд покажет, как настроить пайплайн для работы без интернета и HuggingFace токена.

## 🎯 Зачем это нужно?

- **Приватность** - модели работают локально
- **Стабильность** - нет зависимости от интернета
- **Безопасность** - не нужно хранить токены в продакшене
- **Скорость** - модели уже загружены локально

## 📦 Способ 1: Автоматическое скачивание

### Шаг 1: Получите HuggingFace токен (однократно)

1. Зайдите на https://huggingface.co/settings/tokens
2. Создайте токен с правами чтения
3. Примите лицензию: https://huggingface.co/pyannote/speaker-diarization-3.1

### Шаг 2: Скачайте модели

```bash
# Активируйте виртуальное окружение
source venv/bin/activate

# Установите токен
export HUGGINGFACE_TOKEN=your_token_here

# Скачайте все модели (займет ~10-15 минут)
python download_models.py

# Или скачайте только нужные модели Whisper
python download_models.py --whisper-models "base,small"

# Или пропустите Whisper (они кэшируются автоматически)
python download_models.py --skip-whisper
```

### Шаг 3: Используйте без токена

```bash
# Теперь можно работать без токена!
unset HUGGINGFACE_TOKEN

# Укажите папку с моделями
python main.py input/audio.wav --local-models models/

# Или установите переменную среды один раз
export LOCAL_MODELS_DIR=models
python main.py input/audio.wav
```

## 🛠 Способ 2: Ручная настройка

### Структура папок

Создайте структуру:
```
models/
├── local_config.json
├── whisper/               # Whisper модели (кэшируются автоматически)
└── pyannote/
    └── speaker-diarization-3.1/
        ├── config.yaml
        ├── pytorch_model.bin
        └── ...
```

### Скачивание через Python

```python
import os
from pathlib import Path
from pyannote.audio import Pipeline

# Создайте папки
models_dir = Path("models")
pyannote_dir = models_dir / "pyannote"
pyannote_dir.mkdir(parents=True, exist_ok=True)

# Установите токен
os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"

# Загрузите и сохраните модель
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
)

# Сохраните локально
local_path = pyannote_dir / "speaker-diarization-3.1"
pipeline.to(local_path)

print(f"Модель сохранена в: {local_path}")
```

## 🐳 Docker с локальными моделями

### Обновите Dockerfile:

```dockerfile
# Добавьте в Dockerfile перед ENTRYPOINT:
COPY models/ /app/models/
ENV LOCAL_MODELS_DIR=/app/models
```

### Docker Compose:

```yaml
# Добавьте в docker-compose.yml:
services:
  whisper-diarization:
    build: .
    volumes:
      - ./models:/app/models
    environment:
      - LOCAL_MODELS_DIR=/app/models
```

## 🔄 Портативность

### Способ 1: Архив моделей

```bash
# Создайте архив моделей
tar -czf whisper-models.tar.gz models/

# На другой машине:
tar -xzf whisper-models.tar.gz
export LOCAL_MODELS_DIR=models
python main.py input/audio.wav
```

### Способ 2: Shared storage

```bash
# Сохраните модели в общую папку
python download_models.py --models-dir /shared/whisper-models

# На любой машине:
export LOCAL_MODELS_DIR=/shared/whisper-models
python main.py input/audio.wav
```

## ⚡ Оптимизация размера

### Только нужные модели Whisper:

```bash
# Скачайте только base модель
python download_models.py --whisper-models "base" --models-dir models-minimal/
```

### Размеры моделей:

| Модель | Размер | Качество |
|--------|--------|----------|
| tiny | ~39 MB | Базовое |
| base | ~74 MB | Хорошее |
| small | ~244 MB | Отличное |
| medium | ~769 MB | Высокое |
| large | ~1550 MB | Максимальное |

**PyAnnote модель**: ~1.2 GB

## 🛡️ Безопасность

### Настройка CI/CD:

```yaml
# В .github/workflows/
- name: Download models
  run: |
    export HUGGINGFACE_TOKEN=${{ secrets.HF_TOKEN }}
    python download_models.py --models-dir ./models
    
- name: Run pipeline
  run: |
    export LOCAL_MODELS_DIR=./models
    python main.py test_audio.wav
```

### Продакшен настройка:

```bash
# Один раз настройте модели
export HUGGINGFACE_TOKEN=token_for_download
python download_models.py --models-dir /opt/whisper-models

# В продакшене только:
export LOCAL_MODELS_DIR=/opt/whisper-models
# Токен больше не нужен!
```

## 🔍 Проверка установки

```bash
# Проверьте структуру папок
find models/ -type f -name "*.json" -o -name "*.bin" -o -name "*.yaml"

# Тест без токена
unset HUGGINGFACE_TOKEN
python main.py input/test_audio.wav --local-models models/

# Должно работать без ошибок!
```

## ❓ Решение проблем

### Ошибка "Model not found":
```bash
# Проверьте структуру папок
ls -la models/pyannote/speaker-diarization-3.1/

# Убедитесь, что есть config.yaml
cat models/pyannote/speaker-diarization-3.1/config.yaml
```

### Ошибка загрузки локальной модели:
```bash
# Пересоздайте модель
rm -rf models/pyannote/
python download_models.py --skip-whisper
```

### Большой размер папки models/:
```bash
# Очистите кэш Whisper
rm -rf ~/.cache/whisper/

# Оставьте только нужные модели
python download_models.py --whisper-models "base" --models-dir models-small/
```

## 🎉 Преимущества офлайн режима

✅ **Приватность** - данные не покидают локальную машину  
✅ **Надежность** - нет зависимости от интернета  
✅ **Скорость** - модели загружаются мгновенно  
✅ **Безопасность** - нет токенов в коде  
✅ **Масштабируемость** - легко разворачивать на множестве серверов  

---

💡 **Совет**: После настройки локальных моделей можно удалить HuggingFace токен из переменных среды для дополнительной безопасности. 