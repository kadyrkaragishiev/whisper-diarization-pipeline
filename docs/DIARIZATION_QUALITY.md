# 🎯 Улучшение качества диаризации PyAnnote

## 🚀 Новые возможности (v2.0)

### ✨ Что улучшено:
- **Фильтрация коротких сегментов** - убираем "мусорные" сегменты < 0.5 сек
- **Объединение соседних сегментов** - склеиваем близкие реплики одного спикера
- **Настраиваемые пороги** - контроль min/max количества спикеров
- **Понятные имена спикеров** - "Спикер 1", "Спикер 2" вместо SPEAKER_00
- **Детальная статистика** - видно, сколько сегментов отфильтровано

## 🔧 Новые параметры

### Базовые настройки:
```bash
# Для интервью 1-на-1
python main.py audio.wav --min-speakers 2 --max-speakers 2 --min-segment 0.8

# Для группового разговора
python main.py audio.wav --min-speakers 3 --max-speakers 6 --min-segment 0.3

# Для лекции (1 основной спикер + вопросы)
python main.py audio.wav --min-speakers 1 --max-speakers 3 --min-segment 1.0
```

### Параметры качества:
- `--min-speakers` - минимальное количество спикеров (по умолчанию: 1)
- `--max-speakers` - максимальное количество спикеров (по умолчанию: 10)
- `--min-segment` - минимальная длительность сегмента в секундах (по умолчанию: 0.5)
- `--alignment-strategy` - стратегия совмещения транскрипции и диаризации (по умолчанию: smart)

## 🎯 Стратегии совмещения (NEW! 🚀)

### Проблема Unknown speakers решена!

Новый интеллектуальный алгоритм совмещения с **тремя стратегиями**:

#### `--alignment-strategy strict` 
- **Строгое пересечение** по времени
- Консервативный подход
- Может оставлять Unknown сегменты при плохом совпадении времени

#### `--alignment-strategy smart` (по умолчанию)
- **Комбинированный подход**:
  1. Пересечение по времени (приоритет)
  2. Близость по времени (до 2 сек разрыва)
  3. Близость к средней точке сегмента (до 3 сек)
- **Постобработка Unknown сегментов**:
  1. Ближайший спикер по времени
  2. Контекст соседних сегментов
  3. Самый частый спикер в аудио
- ✅ **Оптимальный выбор для большинства случаев**

#### `--alignment-strategy aggressive`
- **Максимально агрессивное совмещение**
- Связывает сегменты с разрывом до 5 секунд
- Практически исключает Unknown сегменты
- Может быть менее точным при перекрывающихся голосах

### Примеры использования:

```bash
# Проблемы с Unknown speakers? Используйте smart (по умолчанию)
python main.py audio.wav --alignment-strategy smart

# Очень много Unknown? Попробуйте aggressive
python main.py audio.wav --alignment-strategy aggressive

# Нужна максимальная точность? Используйте strict
python main.py audio.wav --alignment-strategy strict
```

## 🎯 Рекомендации по использованию

### 📊 Для разных типов аудио:

| Тип аудио | min-speakers | max-speakers | min-segment | Примечание |
|-----------|--------------|--------------|-------------|------------|
| Интервью 1-на-1 | 2 | 2 | 0.8 | Четкое разделение на 2 голоса |
| Подкаст (2-3 человека) | 2 | 3 | 0.5 | Обычная настройка |
| Групповая дискуссия | 3 | 8 | 0.3 | Много коротких реплик |
| Лекция/презентация | 1 | 3 | 1.0 | Основной + редкие вопросы |
| Телефонный разговор | 2 | 2 | 0.6 | Может быть шум линии |

### 🔍 Диагностика проблем:

#### Проблема: "Слишком много спикеров"
**Симптомы:** 5+ спикеров вместо 2-3 реальных
**Решение:**
```bash
# Уменьшите max-speakers и увеличьте min-segment
python main.py audio.wav --max-speakers 3 --min-segment 0.8
```

#### Проблема: "Unknown speaker" или пропущенные спикеры  
**Симптомы:** Некоторые части не имеют спикера
**Решения:**
```bash
# Стратегия 1: Используйте smart алгоритм (по умолчанию)
python main.py audio.wav --alignment-strategy smart

# Стратегия 2: Агрессивное совмещение
python main.py audio.wav --alignment-strategy aggressive

# Стратегия 3: Уменьшите min-segment для захвата коротких сегментов
python main.py audio.wav --min-segment 0.2 --alignment-strategy smart

# Стратегия 4: Комбинированный подход
python main.py audio.wav --min-segment 0.3 --alignment-strategy aggressive --min-speakers 1
```

#### Проблема: "Спикеры путаются"
**Симптомы:** Один человек определяется как разные спикеры
**Решение:**
```bash
# Увеличьте min-segment для объединения близких сегментов
python main.py audio.wav --min-segment 1.0
```

## 🛠️ Продвинутые техники

### 1. Предобработка аудио
```bash
# Сначала очистите аудио (удалите шум, нормализуйте)
ffmpeg -i input.wav -af "highpass=f=80,lowpass=f=8000,volume=2" clean_audio.wav
python main.py clean_audio.wav --min-speakers 2 --max-speakers 3
```

### 2. Итеративная настройка
```bash
# Шаг 1: Быстрая оценка
python main.py audio.wav --model tiny --min-speakers 2 --max-speakers 5

# Шаг 2: Точная настройка (на основе результатов шага 1)
python main.py audio.wav --model base --min-speakers 2 --max-speakers 2 --min-segment 0.6
```

### 3. Пакетная обработка с разными настройками
```bash
#!/bin/bash
# Тестируем разные настройки
for min_seg in 0.3 0.5 0.8; do
    for max_spk in 2 3 4; do
        echo "Тестируем: min-segment=$min_seg, max-speakers=$max_spk"
        python main.py audio.wav --min-segment $min_seg --max-speakers $max_spk \
               --output "results_${min_seg}_${max_spk}"
    done
done
```

## 📈 Мониторинг качества

### Обращайте внимание на статистику:
```
📈 Статистика диаризации:
   • Сегментов до фильтрации: 15    # Было найдено изначально
   • Сегментов после фильтрации: 8  # Осталось после очистки  
   • Уникальных спикеров: 2         # Итоговое количество
```

**Хорошие показатели:**
- Отфильтровано 20-40% сегментов (убрали шум)
- Количество спикеров соответствует ожиданиям
- Мало сегментов < 1 секунды

**Плохие показатели:**
- Отфильтровано > 60% сегментов (слишком агрессивная фильтрация)
- Найдено спикеров намного больше реальных
- Много коротких сегментов (< 0.5 сек)

## 🎨 Примеры реальных кейсов

### Пример 1: YouTube интервью
```bash
# Проблема: шумное интервью с 2 людьми, но находит 5 спикеров
python main.py interview.wav --min-speakers 2 --max-speakers 2 --min-segment 1.0

# Результат: чистое разделение на 2 спикера
```

### Пример 2: Групповая встреча
```bash
# Проблема: 4 человека, но некоторые говорят редко
python main.py meeting.wav --min-speakers 3 --max-speakers 5 --min-segment 0.4

# Результат: находит всех участников, включая редкие реплики
```

### Пример 3: Лекция с вопросами  
```bash
# Проблема: 1 лектор + студенты задают вопросы
python main.py lecture.wav --min-speakers 1 --max-speakers 4 --min-segment 0.8

# Результат: лектор + отдельные голоса для вопросов
```

## 🔧 Решение частых проблем

### "Галлюцинация" спикеров
**Причины:** Фоновый шум, эхо, музыка, разные интонации
**Решения:**
1. Увеличить `--min-segment` до 0.8-1.0
2. Ограничить `--max-speakers` реальным количеством
3. Очистить аудио от шумов

### Unknown speakers
**Причины:** Тихие участки, некачественное аудио, перекрывающиеся голоса
**Решения:**
1. Уменьшить `--min-segment` до 0.2-0.3
2. Установить `--min-speakers 1`
3. Проверить громкость аудио

### Смешивание спикеров
**Причины:** Похожие голоса, плохое качество записи
**Решения:**
1. Увеличить качество модели: `--model medium` или `--model large`
2. Настроить `--min-segment` для лучшего объединения
3. Предобработать аудио

---

💡 **Совет:** Начните с консервативных настроек (`--min-segment 0.8`, `--max-speakers` равно ожидаемому количеству), затем подстраивайте параметры на основе результатов. 