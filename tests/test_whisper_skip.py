#!/usr/bin/env python3
"""
Скрипт для тестирования проблемы пропуска начала аудио в Whisper
"""

import sys
from main import AudioProcessor
from pathlib import Path

def test_audio_file(audio_path: str, models_to_test: list = None):
    """
    Тестирует проблему пропуска начала аудио с разными моделями
    """
    if models_to_test is None:
        models_to_test = ['tiny', 'base', 'small', 'medium']
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"❌ Файл не найден: {audio_path}")
        return
    
    print(f"🎵 Тестируем файл: {audio_path}")
    print(f"📊 Модели для тестирования: {', '.join(models_to_test)}")
    print("=" * 60)
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🧠 Тестируем модель: {model_name}")
        print("-" * 40)
        
        try:
            # Создаем процессор с текущей моделью
            processor = AudioProcessor(whisper_model=model_name)
            
            # Простая транскрипция (старый способ)
            print("📝 Простая транскрипция:")
            simple_result = processor.whisper_model.transcribe(
                str(audio_path),
                language="ru",
                word_timestamps=True
            )
            
            simple_segments = simple_result.get('segments', [])
            simple_start = simple_segments[0]['start'] if simple_segments else 0
            
            print(f"   🕒 Начинается с: {simple_start:.1f}с")
            print(f"   📊 Сегментов: {len(simple_segments)}")
            
            # Улучшенная транскрипция (новый способ)
            print("🔧 Улучшенная транскрипция:")
            improved_result = processor.transcribe(str(audio_path))
            
            improved_segments = improved_result.get('segments', [])
            improved_start = improved_segments[0]['start'] if improved_segments else 0
            
            print(f"   🕒 Начинается с: {improved_start:.1f}с")
            print(f"   📊 Сегментов: {len(improved_segments)}")
            
            # Сохраняем результаты
            results[model_name] = {
                'simple_start': simple_start,
                'improved_start': improved_start,
                'simple_segments': len(simple_segments),
                'improved_segments': len(improved_segments),
                'has_skip_issue': simple_start > 5.0 or improved_start > 5.0
            }
            
            # Показываем улучшение
            if simple_start > improved_start:
                improvement = simple_start - improved_start
                print(f"   ✅ Улучшение: на {improvement:.1f}с раньше!")
            elif improved_start > simple_start:
                degradation = improved_start - simple_start
                print(f"   ⚠️  Ухудшение: на {degradation:.1f}с позже")
            else:
                print("   ➖ Без изменений")
                
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results[model_name] = {'error': str(e)}
    
    # Итоговый анализ
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ АНАЛИЗ")
    print("=" * 60)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"❌ {model_name}: {result['error']}")
        else:
            simple_start = result['simple_start']
            improved_start = result['improved_start']
            
            status = ""
            if simple_start > 10.0:
                status += "🚨 СЕРЬЕЗНАЯ ПРОБЛЕМА "
            elif simple_start > 5.0:
                status += "⚠️ ПРОБЛЕМА "
            else:
                status += "✅ ОК "
            
            print(f"{status} {model_name}:")
            print(f"   Простая: {simple_start:.1f}с | Улучшенная: {improved_start:.1f}с")
    
    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    
    problematic_models = [name for name, result in results.items() 
                         if 'error' not in result and result['has_skip_issue']]
    
    if problematic_models:
        print(f"⚠️  Проблемные модели: {', '.join(problematic_models)}")
        print("🔧 Попробуйте:")
        print("   • Использовать модель 'base' вместо 'small' или 'medium'")
        print("   • Добавить пустой initial_prompt=''")
        print("   • Уменьшить no_speech_threshold до 0.2-0.3")
        print("   • Установить condition_on_previous_text=False")
    else:
        print("✅ Проблем не обнаружено с текущими настройками!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python test_whisper_skip.py <путь_к_аудио>")
        print("Пример: python test_whisper_skip.py input/1.m4a")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Можно указать конкретные модели для тестирования
    models = ['base', 'small', 'medium'] if len(sys.argv) < 3 else sys.argv[2].split(',')
    
    test_audio_file(audio_file, models) 