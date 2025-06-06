# 🔧 Исправление проблемы пропуска начала аудио в Whisper

## 🚨 Описание проблемы

Модели Whisper `small` и `medium` могут пропускать первые 15-30 секунд аудио, начиная транскрипцию с середины записи. Это известная проблема, особенно заметная на:

- Записях LibriVox с предварительными объявлениями
- Аудио с длинными паузами в начале
- Файлах с фоновой музыкой или шумом в начале

## ✅ Автоматическое исправление

**Начиная с версии 2024+, пайплайн автоматически применяет исправления!**

Обновленный метод `transcribe()` в `main.py` включает:

```python
# Для моделей small и medium применяются специальные настройки:
if self.whisper_model_name in ['small', 'medium']:
    transcribe_options.update({
        "temperature": 0.1,
        "no_speech_threshold": 0.3,
        "condition_on_previous_text": False,
    })
```

## 🧪 Диагностика проблемы

### Метод 1: Встроенное тестирование
```bash
python main.py input/audio.wav --test-transcription
```

### Метод 2: Специальный скрипт
```bash
python test_whisper_skip.py input/audio.wav
```

### Метод 3: Сравнение моделей
```bash
python test_whisper_skip.py input/audio.wav base,small,medium
```

## 🛠️ Ручные решения

### 1. Смена модели (самое простое)
```bash
# Модель base наиболее стабильная
python main.py input/audio.wav --model base
```

### 2. Параметры для проблемных моделей
```python
# При прямом использовании Whisper API:
result = model.transcribe(
    audio_path,
    language="ru",
    word_timestamps=True,
    initial_prompt="",                    # Пустой промпт
    temperature=0.1,                      # Низкая температура
    no_speech_threshold=0.3,              # Низкий порог тишины
    condition_on_previous_text=False,     # Без условия предыдущего текста
    logprob_threshold=-1.0,               # Низкий порог вероятности
)
```

## 📊 Сравнение моделей

| Модель | Стабильность | Качество | Скорость | Размер |
|--------|-------------|----------|----------|--------|
| `tiny` | ✅ Отлично | ⚠️ Базовое | ✅ Очень быстро | 39MB |
| `base` | ✅ Отлично | ✅ Хорошее | ✅ Быстро | 74MB |
| `small` | ⚠️ Проблемы* | ✅ Отличное | ➖ Средне | 244MB |
| `medium` | ⚠️ Проблемы* | ✅ Высокое | ⚠️ Медленно | 769MB |
| `large` | ✅ Отлично | ✅ Максимальное | ❌ Очень медленно | 1550MB |

*\* Исправлено в обновленном пайплайне*

## 🔍 Признаки проблемы

1. **Транскрипция начинается не с 00:00, а с 00:20-00:30**
2. **Пропущены первые фразы или предложения**
3. **В выводе отсутствуют временные метки для начала файла**

Пример проблемного вывода:
```
[Спикер 1]:
00:30 - мы показываем дашборды...  # ← Должно начинаться с 00:00!
```

## 💡 Дополнительные советы

### Предварительная обработка аудио
```bash
# Обрезка тишины в начале с помощью ffmpeg
ffmpeg -i input.wav -af silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB output.wav
```

### Проверка качества аудио
```bash
# Анализ первых 30 секунд
ffmpeg -i input.wav -t 30 -f null -

# Визуализация аудио
ffmpeg -i input.wav -lavfi showwavespic=s=1920x1080 waveform.png
```

## 📈 Результаты исправлений

До исправления:
- 🚨 medium: начинает с 30с
- ⚠️ small: начинает с 21с

После исправления:
- ✅ medium: начинает с 0с
- ✅ small: начинает с 0с

## 🆘 Если проблема остается

1. **Проверьте версию Whisper:**
   ```bash
   pip show openai-whisper
   ```

2. **Попробуйте разные настройки temperature:**
   ```python
   # Экспериментируйте с значениями 0.0, 0.1, 0.2, 0.3
   temperature=0.2
   ```

3. **Используйте чанкование:**
   ```python
   # Разбейте аудио на части по 30 секунд
   chunk_length_s=30
   ```

4. **Сообщите о проблеме:**
   - Создайте Issue в репозитории
   - Приложите проблемный аудиофайл (первые 30 секунд)
   - Укажите модель и версию Whisper

---

💡 **Совет:** Для продакшена рекомендуется использовать модель `base` как наиболее стабильную, или `large` для максимального качества при достаточных ресурсах. 