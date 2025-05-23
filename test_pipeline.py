#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы пайплайна
"""

import sys
import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_audio():
    """Создаем улучшенный тестовый аудиофайл с двумя спикерами"""
    # Создаем более реалистичный аудио с двумя разными "спикерами"
    duration = 15  # секунд
    sample_rate = 16000
    
    # Генерируем различные паттерны для имитации разных спикеров
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Спикер 1: более низкий тон с модуляцией (0-4 секунды)
    segment1_duration = 4
    t1 = t[:int(segment1_duration * sample_rate)]
    speaker1_part1 = (np.sin(2 * np.pi * 300 * t1) + 
                      0.3 * np.sin(2 * np.pi * 450 * t1) +
                      0.1 * np.sin(2 * np.pi * 150 * t1)) * 0.3
    
    # Пауза (4-5 секунд)
    pause1_duration = 1
    pause1 = np.zeros(int(pause1_duration * sample_rate))
    
    # Спикер 2: более высокий тон (5-9 секунд) 
    segment2_duration = 4
    t2 = t[int(5 * sample_rate):int(9 * sample_rate)]
    speaker2_part1 = (np.sin(2 * np.pi * 600 * t2) + 
                      0.2 * np.sin(2 * np.pi * 800 * t2) +
                      0.15 * np.sin(2 * np.pi * 400 * t2)) * 0.3
    
    # Пауза (9-10 секунд)
    pause2_duration = 1
    pause2 = np.zeros(int(pause2_duration * sample_rate))
    
    # Спикер 1 снова (10-14 секунд)
    segment3_duration = 4
    t3 = t[int(10 * sample_rate):int(14 * sample_rate)]
    speaker1_part2 = (np.sin(2 * np.pi * 280 * t3) + 
                      0.35 * np.sin(2 * np.pi * 420 * t3) +
                      0.12 * np.sin(2 * np.pi * 140 * t3)) * 0.3
    
    # Финальная пауза
    final_pause = np.zeros(int(1 * sample_rate))
    
    # Объединяем все части
    audio = np.concatenate([
        speaker1_part1,  # 0-4с: Спикер 1
        pause1,          # 4-5с: Пауза
        speaker2_part1,  # 5-9с: Спикер 2  
        pause2,          # 9-10с: Пауза
        speaker1_part2,  # 10-14с: Спикер 1
        final_pause      # 14-15с: Пауза
    ])
    
    # Добавляем легкий шум для реалистичности
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # Сохраняем в input/
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    test_file = input_dir / "test_diarization.wav"
    sf.write(test_file, audio, sample_rate)
    
    return test_file

def main():
    print("🧪 Тест пайплайна Whisper + PyAnnote Audio")
    
    # Проверяем, есть ли основной модуль
    if not Path("main.py").exists():
        print("❌ Файл main.py не найден!")
        sys.exit(1)
    
    # Создаем тестовый аудиофайл
    print("🔊 Создаем тестовый аудиофайл...")
    test_file = create_test_audio()
    print(f"✅ Тестовый файл создан: {test_file}")
    
    # Проверяем зависимости
    try:
        import whisper
        import torch
        print("✅ Whisper установлен")
        
        try:
            from pyannote.audio import Pipeline
            print("✅ PyAnnote Audio установлен")
        except ImportError:
            print("⚠️  PyAnnote Audio не установлен или есть проблемы с импортом")
        
        # Проверяем устройство
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        print(f"🔧 Доступное устройство: {device}")
        
        # Пытаемся загрузить модель Whisper
        print("📥 Проверяем загрузку модели Whisper...")
        # Используем CPU для Whisper чтобы избежать проблем с MPS в версии 20240930
        model = whisper.load_model("tiny", device="cpu")  # Принудительно используем CPU
        print("✅ Модель Whisper загружена успешно (на CPU)")
        
        print("\n🎯 Для полного тестирования запустите:")
        print(f"python main.py {test_file} --model tiny")
        print("\n💡 Для использования диаризации не забудьте настроить HuggingFace токен!")
        print("\n⚠️  Примечание: Whisper v20240930 может иметь проблемы с MPS на некоторых Mac,")
        print("    поэтому тест загружает модель на CPU. В основном пайплайне можно попробовать MPS.")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Установите зависимости: pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 