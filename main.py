#!/usr/bin/env python3
"""
Whisper + PyAnnote Audio Pipeline
Транскрипция и диаризация аудиофайлов
"""

import os
import sys
import warnings
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import click
import torch
import whisper
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Segment
import pandas as pd
from tqdm import tqdm
import time

# Подавляем предупреждения
warnings.filterwarnings("ignore")


class AudioProcessor:
    """Класс для обработки аудио: транскрипция + диаризация"""
    
    def __init__(self, whisper_model: str = "base", hf_token: Optional[str] = None, 
                 local_models_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Инициализация процессора
        
        Args:
            whisper_model: Модель Whisper (tiny, base, small, medium, large)
            hf_token: HuggingFace токен для PyAnnotate
            local_models_dir: Директория с локально сохраненными моделями
            device: Устройство для инференса (cpu, cuda, mps)
        """
        self.whisper_model_name = whisper_model
        self.hf_token = hf_token
        self.local_models_dir = Path(local_models_dir) if local_models_dir else None
        
        # Выбор устройства
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"
            
        print(f"🔧 Используется устройство: {self.device}")
        
        # Загружаем модели
        self._load_models()
    
    def _load_local_config(self) -> Optional[Dict]:
        """Загрузка локальной конфигурации моделей"""
        if not self.local_models_dir:
            return None
            
        config_file = self.local_models_dir / "local_config.json"
        if not config_file.exists():
            return None
            
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Ошибка чтения локальной конфигурации: {e}")
            return None
    
    def _load_models(self):
        """Загрузка моделей Whisper и PyAnnotate"""
        print("📥 Загружаем модель Whisper...")
        self.whisper_model = whisper.load_model(
            self.whisper_model_name, 
            device=self.device
        )
        
        print("📥 Загружаем модель диаризации...")
        self.diarization_pipeline = None
        
        # Проверяем локальные модели
        local_config = self._load_local_config()
        
        if local_config and self.local_models_dir:
            # Пытаемся загрузить локальную модель
            try:
                pyannote_model_path = self.local_models_dir / "pyannote" / "speaker-diarization-3.1"
                
                if pyannote_model_path.exists():
                    print(f"🏠 Загружаем локальную модель диаризации из: {pyannote_model_path}")
                    self.diarization_pipeline = Pipeline.from_pretrained(str(pyannote_model_path))
                    
                    if self.device != "cpu":
                        self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))
                    
                    print("✅ Локальная модель диаризации загружена")
                    return
                    
            except Exception as e:
                print(f"⚠️  Ошибка загрузки локальной модели: {e}")
                print("🔄 Пытаемся загрузить модель из HuggingFace...")
        
        # Загружаем модель из HuggingFace
        try:
            if not self.hf_token:
                print("⚠️  HuggingFace токен не найден")
                print("💡 Для диаризации нужен токен или локальные модели")
                print("💡 Скачайте модели: python download_models.py")
                return
                
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            
            if self.device != "cpu":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))
                
            print("✅ Модель диаризации загружена из HuggingFace")
            
        except Exception as e:
            print(f"⚠️  Ошибка загрузки модели диаризации: {e}")
            print("💡 Убедитесь, что у вас есть HuggingFace токен и доступ к pyannote/speaker-diarization-3.1")
            print("💡 Или скачайте модели локально: python download_models.py")
            self.diarization_pipeline = None
    
    def _prepare_audio(self, audio_path: str) -> str:
        """
        Подготовка аудио для обработки (конвертация в нужный формат)
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Путь к подготовленному аудиофайлу
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
        
        # Если уже wav файл, возвращаем как есть
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path)
        
        # Конвертируем в wav
        print("🔄 Конвертация аудио в WAV формат...")
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_wav.name, audio, sr)
        
        return temp_wav.name
    
    def transcribe(self, audio_path: str) -> Dict:
        """
        Транскрипция аудио с помощью Whisper
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Результат транскрипции
        """
        print("🎤 Начинаем транскрипцию...")
        
        result = self.whisper_model.transcribe(
            audio_path,
            language="ru",  # Можно сделать auto-detection
            word_timestamps=True
        )
        
        return result
    
    def diarize(self, audio_path: str) -> Optional[Dict]:
        """
        Диаризация аудио с помощью PyAnnotate
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Результат диаризации или None если модель недоступна
        """
        if self.diarization_pipeline is None:
            print("⚠️  Диаризация недоступна - модель не загружена")
            return None
            
        print("👥 Начинаем диаризацию спикеров...")
        
        try:
            diarization = self.diarization_pipeline(audio_path)
            
            # Конвертируем результат в удобный формат
            speakers = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return {"speakers": speakers}
            
        except Exception as e:
            print(f"⚠️  Ошибка диаризации: {e}")
            return None
    
    def _align_transcription_with_speakers(self, transcription: Dict, diarization: Dict) -> List[Dict]:
        """
        Совмещение транскрипции с информацией о спикерах
        
        Args:
            transcription: Результат транскрипции
            diarization: Результат диаризации
            
        Returns:
            Список сегментов с текстом и спикерами
        """
        if not diarization or "speakers" not in diarization:
            # Если диаризации нет, возвращаем только транскрипцию
            segments = []
            for segment in transcription.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": "Unknown"
                })
            return segments
        
        # Совмещаем транскрипцию и диаризацию
        aligned_segments = []
        speaker_segments = diarization["speakers"]
        transcription_segments = transcription.get("segments", [])
        
        for t_segment in transcription_segments:
            t_start, t_end = t_segment["start"], t_segment["end"]
            t_mid = (t_start + t_end) / 2
            
            # Находим спикера для этого сегмента
            best_speaker = "Unknown"
            best_overlap = 0
            
            for s_segment in speaker_segments:
                s_start, s_end = s_segment["start"], s_segment["end"]
                
                # Вычисляем пересечение
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = s_segment["speaker"]
            
            aligned_segments.append({
                "start": t_start,
                "end": t_end,
                "text": t_segment["text"].strip(),
                "speaker": best_speaker
            })
        
        return aligned_segments
    
    def process(self, audio_path: str, output_dir: str = "output") -> Dict:
        """
        Полная обработка аудио: транскрипция + диаризация
        
        Args:
            audio_path: Путь к аудиофайлу
            output_dir: Директория для сохранения результатов
            
        Returns:
            Результаты обработки
        """
        # Создаем директорию вывода
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Подготавливаем аудио
        prepared_audio = self._prepare_audio(audio_path)
        
        try:
            # Транскрипция
            start_time = time.time()
            transcription_result = self.transcribe(prepared_audio)
            transcription_time = time.time() - start_time
            
            # Диаризация
            start_time = time.time()
            diarization_result = self.diarize(prepared_audio)
            diarization_time = time.time() - start_time
            
            # Совмещаем результаты
            aligned_segments = self._align_transcription_with_speakers(
                transcription_result, 
                diarization_result
            )
            
            # Подготавливаем итоговый результат
            result = {
                "audio_file": str(Path(audio_path).name),
                "transcription": transcription_result.get("text", ""),
                "segments": aligned_segments,
                "language": transcription_result.get("language", "unknown"),
                "has_speaker_diarization": diarization_result is not None,
                "transcription_time": transcription_time,
                "diarization_time": diarization_time
            }
            
            # Сохраняем результаты
            self._save_results(result, output_path, Path(audio_path).stem)
            
            return result
            
        finally:
            # Удаляем временный файл если он был создан
            if prepared_audio != audio_path and os.path.exists(prepared_audio):
                os.unlink(prepared_audio)
    
    def _save_results(self, result: Dict, output_path: Path, base_name: str):
        """Сохранение результатов в различных форматах"""
        
        # JSON
        json_path = output_path / f"{base_name}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 Результат сохранен: {json_path}")
        
        # CSV
        if result["segments"]:
            df = pd.DataFrame(result["segments"])
            csv_path = output_path / f"{base_name}_segments.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"💾 Сегменты сохранены: {csv_path}")
        
        # TXT (читаемый формат)
        txt_path = output_path / f"{base_name}_transcript.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Транскрипция: {result['audio_file']}\n")
            f.write(f"Язык: {result['language']}\n")
            f.write(f"Диаризация: {'Да' if result['has_speaker_diarization'] else 'Нет'}\n\n")
            
            if result["has_speaker_diarization"]:
                f.write("=== ТРАНСКРИПЦИЯ ПО СПИКЕРАМ ===\n\n")
                current_speaker = None
                for segment in result["segments"]:
                    if segment["speaker"] != current_speaker:
                        current_speaker = segment["speaker"]
                        f.write(f"\n[{current_speaker}]:\n")
                    
                    start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
                    f.write(f"{start_time} - {segment['text']}\n")
            else:
                f.write("=== ТРАНСКРИПЦИЯ ===\n\n")
                f.write(result["transcription"])
        
        print(f"💾 Транскрипт сохранен: {txt_path}")


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--model', '-m', default='large', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Модель Whisper для использования')
@click.option('--output', '-o', default='output', 
              help='Директория для сохранения результатов')
@click.option('--hf-token', envvar='HUGGINGFACE_TOKEN', 
              help='HuggingFace токен (можно задать в переменной HUGGINGFACE_TOKEN)')
@click.option('--local-models', envvar='LOCAL_MODELS_DIR',
              help='Директория с локальными моделями (можно задать в переменной LOCAL_MODELS_DIR)')
@click.option('--device', default=None, type=click.Choice(['cpu', 'cuda', 'mps']), help='Устройство для инференса (cpu, cuda, mps)')
def main(audio_file: str, model: str, output: str, hf_token: Optional[str], local_models: Optional[str], device: Optional[str]):
    """
    Пайплайн транскрипции и диаризации аудио
    
    AUDIO_FILE: Путь к аудиофайлу для обработки
    """
    
    print("🎵 Whisper + PyAnnote Audio Pipeline")
    print(f"📁 Обрабатываем: {audio_file}")
    print(f"🧠 Модель Whisper: {model}")
    print(f"📂 Результаты будут сохранены в: {output}")
    
    if local_models:
        print(f"🏠 Используем локальные модели из: {local_models}")
    elif not hf_token:
        print("⚠️  HuggingFace токен не найден. Диаризация будет недоступна.")
        print("💡 Установите токен: export HUGGINGFACE_TOKEN=your_token")
        print("💡 Или скачайте модели локально: python download_models.py")
    
    try:
        # Создаем процессор
        processor = AudioProcessor(
            whisper_model=model, 
            hf_token=hf_token,
            local_models_dir=local_models,
            device=device
        )
        
        # Обрабатываем аудио
        result = processor.process(audio_file, output)
        
        print("\n✅ Обработка завершена!")
        print(f"📊 Найдено сегментов: {len(result['segments'])}")
        
        if result['has_speaker_diarization']:
            speakers = set(seg['speaker'] for seg in result['segments'])
            print(f"👥 Обнаружено спикеров: {len(speakers)} ({', '.join(speakers)})")
        
        print(f"🔤 Язык: {result['language']}")
        print(f"📝 Полный текст: {len(result['transcription'])} символов")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 