# 🚀 Быстрый справочник команд

## 🎯 Основные команды

### Стандартная обработка
```bash
python main.py audio.wav                    # Базовая обработка
python main.py audio.wav --model large      # Лучшее качество транскрипции
```

### Улучшенная диаризация v2.0
```bash
# Интервью двух людей
python main.py interview.wav --min-speakers 2 --max-speakers 2 --min-segment 0.8

# Подкаст 2-3 человека
python main.py podcast.wav --min-speakers 2 --max-speakers 3 --min-segment 0.5

# Групповая встреча
python main.py meeting.wav --min-speakers 3 --max-speakers 6 --min-segment 0.3

# Лекция с вопросами
python main.py lecture.wav --min-speakers 1 --max-speakers 4 --min-segment 1.0

# Телефонный разговор
python main.py call.wav --min-speakers 2 --max-speakers 2 --min-segment 0.6
```

## 🔧 Настройка параметров

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `--min-speakers` | 1-10 | Минимальное количество спикеров |
| `--max-speakers` | 1-10 | Максимальное количество спикеров |
| `--min-segment` | 0.1-2.0 | Минимальная длительность сегмента (сек) |
| `--alignment-strategy` | strict/smart/aggressive | Стратегия совмещения транскрипции и диаризации |
| `--model` | tiny/base/small/medium/large | Качество транскрипции |
| `--output` | путь | Папка для результатов |

## 🎯 Стратегии совмещения (NEW!)

### Решение проблемы Unknown speakers:

```bash
# 🔧 Smart (по умолчанию) - оптимальный баланс
python main.py audio.wav --alignment-strategy smart

# 🚀 Aggressive - максимум совмещений, минимум Unknown
python main.py audio.wav --alignment-strategy aggressive

# 🎯 Strict - только точные пересечения
python main.py audio.wav --alignment-strategy strict
```

## 🚨 Решение проблем

### Слишком много спикеров
```bash
# Было: 7 спикеров, нужно: 2
python main.py audio.wav --max-speakers 2 --min-segment 0.8
```

### Unknown speakers  
```bash
# НОВОЕ! Попробуйте smart стратегию (по умолчанию)
python main.py audio.wav --alignment-strategy smart

# Много неопознанных сегментов? Агрессивный режим
python main.py audio.wav --alignment-strategy aggressive

# Комбинируйте с другими параметрами
python main.py audio.wav --min-segment 0.2 --alignment-strategy aggressive
```

### Спикеры путаются
```bash
# Один человек = разные спикеры
python main.py audio.wav --min-segment 1.0 --max-speakers 3
```

## 📊 Интерпретация результатов

### Хорошая статистика:
```
📈 Статистика диаризации:
   • Сегментов до фильтрации: 20      # ✅ Разумное количество
   • Сегментов после фильтрации: 12   # ✅ ~40% отфильтровано  
   • Уникальных спикеров: 2           # ✅ Соответствует ожиданиям
```

### Проблемная статистика:
```
📈 Статистика диаризации:  
   • Сегментов до фильтрации: 50      # ❌ Слишком много
   • Сегментов после фильтрации: 8    # ❌ >80% отфильтровано
   • Уникальных спикеров: 7           # ❌ Больше реальных
```

## 🎨 Шаблоны команд

### Скопируйте и адаптируйте:

```bash
# 🎤 ИНТЕРВЬЮ (строгое разделение на 2)
python main.py YOUR_FILE.wav --min-speakers 2 --max-speakers 2 --min-segment 0.8

# 👥 ВСТРЕЧА (гибкое количество участников)  
python main.py YOUR_FILE.wav --min-speakers 3 --max-speakers 6 --min-segment 0.3

# 🎓 ЛЕКЦИЯ (один основной + вопросы)
python main.py YOUR_FILE.wav --min-speakers 1 --max-speakers 3 --min-segment 1.0

# 📞 ЗВОНОК (только 2 человека)
python main.py YOUR_FILE.wav --min-speakers 2 --max-speakers 2 --min-segment 0.6

# 🎙️ ПОДКАСТ (обычно 2-3 ведущих)
python main.py YOUR_FILE.wav --min-speakers 2 --max-speakers 3 --min-segment 0.5
```

## 💡 Полезные советы

1. **Начните с консервативных настроек**: `--min-segment 0.8`
2. **Ограничьте max-speakers** реальным количеством участников  
3. **Для шумного аудио**: увеличьте `--min-segment` до 1.0
4. **Для коротких реплик**: уменьшите `--min-segment` до 0.3
5. **Смотрите на статистику** - она подскажет, нужно ли менять настройки

---
📖 **Подробное руководство**: [DIARIZATION_QUALITY.md](DIARIZATION_QUALITY.md) 