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

# Добавляем поддержку transformers для кастомных моделей
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  transformers не установлен. Кастомные модели HF недоступны.")
    HF_TRANSFORMERS_AVAILABLE = False


class ModelDownloader:
    """Класс для скачивания и сохранения моделей"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Папки для разных типов моделей
        self.whisper_dir = self.models_dir / "whisper"
        self.pyannote_dir = self.models_dir / "pyannote"
        self.custom_whisper_dir = self.models_dir / "custom_whisper"
        
        self.whisper_dir.mkdir(exist_ok=True)
        self.pyannote_dir.mkdir(exist_ok=True)
        self.custom_whisper_dir.mkdir(exist_ok=True)
        
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
    
    def download_custom_whisper_model(self, model_id: str, local_name: Optional[str] = None):
        """Скачивание кастомной модели Whisper из HuggingFace"""
        if not HF_TRANSFORMERS_AVAILABLE:
            print("❌ transformers не установлен. Установите: pip install transformers")
            return False
        
        print(f"📥 Скачиваем кастомную модель Whisper: {model_id}")
        
        try:
            # Определяем имя для сохранения
            save_name = local_name or model_id.replace("/", "_")
            save_path = self.custom_whisper_dir / save_name
            
            print(f"⬇️  Загружаем модель...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            print(f"⬇️  Загружаем процессор...")
            processor = AutoProcessor.from_pretrained(model_id)
            
            print(f"💾 Сохраняем в: {save_path}")
            
            # Сохраняем модель и процессор
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            
            print(f"✅ Кастомная модель сохранена: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка скачивания кастомной модели: {e}")
            return False
    
    def copy_local_custom_model(self, source_path: str, model_name: str):
        """Копирование локальной кастомной модели в кеш"""
        source = Path(source_path)
        if not source.exists():
            print(f"❌ Путь не найден: {source_path}")
            return False
        
        dest_path = self.custom_whisper_dir / model_name
        
        try:
            print(f"📁 Копируем модель из {source} в {dest_path}")
            
            if dest_path.exists():
                shutil.rmtree(dest_path)
            
            shutil.copytree(source, dest_path)
            
            print(f"✅ Модель скопирована: {dest_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка копирования модели: {e}")
            return False
    
    def list_custom_models(self):
        """Список кастомных моделей в кеше"""
        if not self.custom_whisper_dir.exists():
            print("📁 Кастомные модели не найдены")
            return []
        
        models = []
        for model_dir in self.custom_whisper_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models.append(model_dir.name)
        
        if models:
            print("📋 Кастомные модели в кеше:")
            for model in models:
                print(f"  • {model}")
        else:
            print("📁 Кастомные модели не найдены")
        
        return models
    
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
@click.option('--custom-whisper-model', 
              help='ID кастомной модели Whisper для скачивания (например, antony66/whisper-large-v3-russian)')
@click.option('--custom-model-name',
              help='Локальное имя для кастомной модели (по умолчанию из ID)')
@click.option('--copy-local-model',
              help='Путь к локальной кастомной модели для копирования в кеш')
@click.option('--local-model-name', default='whisper-large-v3-russian',
              help='Имя для локальной модели при копировании')
@click.option('--list-custom', is_flag=True,
              help='Показать список кастомных моделей в кеше')
def main(hf_token: Optional[str], models_dir: str, whisper_models: str, 
         skip_whisper: bool, skip_pyannote: bool, custom_whisper_model: Optional[str],
         custom_model_name: Optional[str], copy_local_model: Optional[str],
         local_model_name: str, list_custom: bool):
    """
    Скачивание моделей для офлайн использования
    """
    
    print("📦 Скачивание моделей для локального использования")
    print(f"📁 Директория: {models_dir}")
    
    downloader = ModelDownloader(models_dir)
    
    # Показать список кастомных моделей
    if list_custom:
        downloader.list_custom_models()
        return
    
    # Копирование локальной кастомной модели
    if copy_local_model:
        success = downloader.copy_local_custom_model(copy_local_model, local_model_name)
        if success:
            print(f"\n✅ Локальная модель скопирована!")
            print(f"💡 Используйте: python main.py input.wav --custom-model {models_dir}/custom_whisper/{local_model_name}")
        return
    
    # Скачивание кастомной модели Whisper
    if custom_whisper_model:
        if not HF_TRANSFORMERS_AVAILABLE:
            print("❌ Для кастомных моделей нужна библиотека transformers")
            print("💡 Установите: pip install transformers")
            sys.exit(1)
        
        success = downloader.download_custom_whisper_model(custom_whisper_model, custom_model_name)
        if success:
            model_name = custom_model_name or custom_whisper_model.replace("/", "_")
            print(f"\n✅ Кастомная модель скачана!")
            print(f"💡 Используйте: python main.py input.wav --custom-model {models_dir}/custom_whisper/{model_name}")
        return
    
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
    
    # Показываем доступные кастомные модели
    print("\n📋 Доступные кастомные модели:")
    downloader.list_custom_models()


if __name__ == "__main__":
    main() 