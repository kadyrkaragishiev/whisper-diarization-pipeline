#!/usr/bin/env python3
"""
Скрипт для предварительного скачивания моделей
Позволяет скачать модели один раз с токеном, а потом использовать без токена
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

import click
import torch
from pyannote.audio import Pipeline
import whisper


class ModelDownloader:
    """Класс для скачивания и сохранения моделей"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Папки для разных типов моделей
        self.whisper_dir = self.models_dir / "whisper"
        self.pyannote_dir = self.models_dir / "pyannote"
        
        self.whisper_dir.mkdir(exist_ok=True)
        self.pyannote_dir.mkdir(exist_ok=True)
        
        # Определяем устройство
        self.device = torch.device("cpu")
    
    def download_whisper_models(self, models: list = None):
        """Скачивание моделей Whisper"""
        if models is None:
            models = ["tiny", "base", "small", "medium", "large"]
        
        print("📥 Скачиваем модели Whisper...")
        
        for model_name in models:
            try:
                print(f"⬇️  Скачиваем {model_name}...")
                model = whisper.load_model(model_name, device=self.device)
                print(f"✅ {model_name} скачана и кэширована")
                
            except Exception as e:
                print(f"❌ Ошибка скачивания {model_name}: {e}")
    
    def download_pyannote_model(self, hf_token: str, model_name: str = "pyannote/speaker-diarization-3.1"):
        """Скачивание модели PyAnnote"""
        print(f"📥 Скачиваем модель диаризации: {model_name}")
        
        try:
            # Скачиваем модель с токеном
            pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
            pipeline.to(self.device)  # Перемещаем на нужное устройство
            # pipeline.save_pretrained(local_path)  # Удалено, не поддерживается
            print(f"✅ Модель загружена через pyannote.audio")
            # Создаем конфиг файл для использования без токена
            config_path = self.pyannote_dir / "config.yaml"
            with open(config_path, 'w') as f:
                f.write(f"model_name: {model_name}\n")
                f.write(f"device: {self.device}\n")
            print(f"📝 Конфигурация сохранена в: {config_path}")
        except Exception as e:
            print(f"❌ Ошибка скачивания модели диаризации: {e}")
            print("💡 Проверьте токен и доступ к модели")
            return False
        return True
    
    def create_local_config(self):
        """Создание конфигурации для локального использования"""
        config = {
            "whisper_cache": str(Path.home() / ".cache" / "whisper"),
            "pyannote_model": str(self.pyannote_dir / "speaker-diarization-3.1"),
            "use_local_models": True,
            "device": str(self.device)
        }
        
        config_file = self.models_dir / "local_config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📝 Локальная конфигурация создана: {config_file}")
        return config_file


@click.command()
@click.option('--hf-token', envvar='HUGGINGFACE_TOKEN', 
              help='HuggingFace токен (обязательно для первого скачивания)')
@click.option('--models-dir', default='models', 
              help='Директория для сохранения моделей')
@click.option('--whisper-models', default='tiny,base,small,medium,large',
              help='Модели Whisper для скачивания (через запятую)')
@click.option('--skip-whisper', is_flag=True, 
              help='Пропустить скачивание моделей Whisper')
@click.option('--skip-pyannote', is_flag=True,
              help='Пропустить скачивание модели PyAnnote')
def main(hf_token: Optional[str], models_dir: str, whisper_models: str, 
         skip_whisper: bool, skip_pyannote: bool):
    """
    Скачивание моделей для офлайн использования
    """
    
    print("📦 Скачивание моделей для локального использования")
    print(f"📁 Директория: {models_dir}")
    
    downloader = ModelDownloader(models_dir)
    
    # Скачиваем модели Whisper
    if not skip_whisper:
        models_list = [m.strip() for m in whisper_models.split(',')]
        downloader.download_whisper_models(models_list)
    else:
        print("⏭️  Пропускаем скачивание Whisper моделей")
    
    # Скачиваем модель PyAnnote
    if not skip_pyannote:
        if not hf_token:
            print("❌ HuggingFace токен необходим для скачивания модели диаризации")
            print("💡 Установите токен: export HUGGINGFACE_TOKEN=your_token")
            print("💡 Или используйте флаг --skip-pyannote")
            sys.exit(1)
        
        success = downloader.download_pyannote_model(hf_token)
        if not success:
            print("❌ Не удалось скачать модель диаризации")
            sys.exit(1)
    else:
        print("⏭️  Пропускаем скачивание PyAnnote модели")
    
    # Создаем локальную конфигурацию
    config_file = downloader.create_local_config()
    
    print("\n✅ Скачивание завершено!")
    print("\n📋 Теперь вы можете использовать пайплайн без токена:")
    print(f"python main.py input/audio.wav --local-models {models_dir}")
    print("\n💡 Или установите переменную среды:")
    print(f"export LOCAL_MODELS_DIR={models_dir}")


if __name__ == "__main__":
    main() 