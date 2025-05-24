#!/usr/bin/env python3
"""
Быстрая настройка русской модели Whisper
Копирует модель из whisper-large-v3-russian-pt в кеш моделей
"""

import os
import sys
import shutil
from pathlib import Path

def setup_russian_model():
    """Настройка русской модели Whisper"""
    
    # Пути
    current_dir = Path.cwd()
    source_path = current_dir / "whisper-large-v3-russian-pt"
    models_dir = current_dir / "models"
    custom_whisper_dir = models_dir / "custom_whisper"
    dest_path = custom_whisper_dir / "whisper-large-v3-russian"
    
    print("🇷🇺 Настройка русской модели Whisper...")
    print(f"📁 Источник: {source_path}")
    print(f"📁 Назначение: {dest_path}")
    
    # Проверяем наличие исходной модели
    if not source_path.exists():
        print(f"❌ Папка с моделью не найдена: {source_path}")
        print("💡 Убедитесь, что папка whisper-large-v3-russian-pt находится в текущей директории")
        return False
    
    # Проверяем файлы модели
    required_files = ["config.json", "generation_config.json"]
    for file in required_files:
        if not (source_path / file).exists():
            print(f"❌ Отсутствует файл: {file}")
            return False
    
    # Создаем директории
    custom_whisper_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Удаляем существующую модель если есть
        if dest_path.exists():
            print("🗑️  Удаляем существующую модель...")
            shutil.rmtree(dest_path)
        
        # Копируем модель
        print("📁 Копируем модель...")
        shutil.copytree(source_path, dest_path)
        
        print("✅ Модель успешно скопирована!")
        print(f"📍 Расположение: {dest_path}")
        
        # Информация по использованию
        print("\n🚀 Готово! Теперь вы можете использовать русскую модель:")
        print(f"python main.py input.wav --custom-model {dest_path}")
        print("\n💡 Альтернативно:")
        print(f"python main.py input.wav --custom-model models/custom_whisper/whisper-large-v3-russian")
        
        # Проверяем размер модели
        total_size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
        print(f"\n📊 Размер модели: {total_size / (1024**3):.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка копирования: {e}")
        return False

def main():
    """Главная функция"""
    print("🇷🇺 Настройка русской модели Whisper antony66/whisper-large-v3-russian")
    print("=" * 60)
    
    if setup_russian_model():
        print("\n🎉 Настройка завершена успешно!")
        
        # Предлагаем установить зависимости
        print("\n📦 Не забудьте установить необходимые зависимости:")
        print("pip install transformers accelerate")
        
        # Показываем пример использования
        print("\n📝 Пример использования:")
        print("python main.py sample.wav --custom-model models/custom_whisper/whisper-large-v3-russian")
        
    else:
        print("\n❌ Настройка не завершена")
        sys.exit(1)

if __name__ == "__main__":
    main() 