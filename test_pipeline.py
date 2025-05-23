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
    """Создаем тестовый аудиофайл"""
    # Создаем простой синтезированный аудио с двумя "спикерами"
    duration = 10  # секунд
    sample_rate = 16000
    
    # Генерируем два разных тона (имитируем разных спикеров)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Спикер 1: частота 440 Hz (первые 5 секунд)
    speaker1 = np.sin(2 * np.pi * 440 * t[:len(t)//2]) * 0.3
    
    # Спикер 2: частота 880 Hz (вторые 5 секунд) 
    speaker2 = np.sin(2 * np.pi * 880 * t[len(t)//2:]) * 0.3
    
    # Объединяем
    audio = np.concatenate([speaker1, speaker2])
    
    # Сохраняем в input/
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    test_file = input_dir / "test_audio.wav"
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
        model = whisper.load_model("tiny", device=device)  # Используем tiny для быстрого теста
        print("✅ Модель Whisper загружена успешно")
        
        print("\n🎯 Для полного тестирования запустите:")
        print(f"python main.py {test_file} --model tiny")
        print("\n💡 Для использования диаризации не забудьте настроить HuggingFace токен!")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Установите зависимости: pip install -r requirements.txt")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 