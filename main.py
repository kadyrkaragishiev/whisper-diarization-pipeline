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

# Импорты для работы с кастомными моделями HuggingFace
try:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  transformers не установлен. Кастомные модели HF недоступны.")
    print("💡 Для поддержки кастомных моделей установите: pip install transformers")
    HF_TRANSFORMERS_AVAILABLE = False

# Подавляем предупреждения
warnings.filterwarnings("ignore")


class AudioProcessor:
    """Класс для обработки аудио: транскрипция + диаризация"""
    
    def __init__(self, whisper_model: str = "base", hf_token: Optional[str] = None, 
                 local_models_dir: Optional[str] = None, device: Optional[str] = None,
                 custom_whisper_model: Optional[str] = None):
        """
        Инициализация процессора
        
        Args:
            whisper_model: Модель Whisper (tiny, base, small, medium, large)
            hf_token: HuggingFace токен для PyAnnotate
            local_models_dir: Директория с локально сохраненными моделями
            device: Устройство для инференса (cpu, cuda, mps)
            custom_whisper_model: Путь к кастомной модели Whisper (HuggingFace format) или HF model ID
        """
        self.whisper_model_name = whisper_model
        self.custom_whisper_model = custom_whisper_model
        self.hf_token = hf_token
        self.local_models_dir = Path(local_models_dir) if local_models_dir else None
        
        # Тип модели Whisper (standard или custom)
        self.whisper_model_type = "custom" if custom_whisper_model else "standard"
        
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
        
        # Инициализируем переменные для моделей
        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_pipeline = None  # Для pipeline API
        
        if self.custom_whisper_model:
            print(f"🧠 Кастомная модель Whisper: {self.custom_whisper_model}")
        else:
            print(f"🧠 Стандартная модель Whisper: {whisper_model}")
        
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
        
        # Загрузка Whisper модели
        whisper_device = self.device
        
        if self.whisper_model_type == "custom" and self.custom_whisper_model:
            # Загружаем кастомную модель через transformers
            self._load_custom_whisper_model(whisper_device)
        else:
            # Загружаем стандартную модель через whisper
            self._load_standard_whisper_model(whisper_device)
        
        # Загрузка модели диаризации (остается без изменений)
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
                else:
                    # Пытаемся загрузить из кеша без токена
                    print("🔄 Пытаемся загрузить модель из кеша...")
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                    
                    if self.device != "cpu":
                        self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))
                    
                    print("✅ Модель диаризации загружена из кеша")
                    return
                    
            except Exception as e:
                print(f"⚠️  Ошибка загрузки локальной модели: {e}")
                print("🔄 Пытаемся загрузить модель из HuggingFace...")
        
        # Пытаемся загрузить из кеша без токена (если нет локальной конфигурации)
        try:
            print("🔄 Пытаемся загрузить модель из кеша...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            
            if self.device != "cpu":
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device(self.device))
            
            print("✅ Модель диаризации загружена из кеша")
            return
            
        except Exception as e:
            print(f"⚠️  Ошибка загрузки модели из кеша: {e}")
        
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
    
    def _load_standard_whisper_model(self, whisper_device: str):
        """Загрузка стандартной модели Whisper"""
        try:
            self.whisper_model = whisper.load_model(
                self.whisper_model_name, 
                device=whisper_device
            )
            self.whisper_processor = None  # Стандартная модель не использует processor
            print(f"✅ Стандартная модель Whisper загружена на {whisper_device}")
        except Exception as e:
            if whisper_device == "mps" and ("SparseMPS" in str(e) or "aten::empty.memory_format" in str(e) or "_sparse_coo_tensor_with_dims_and_tensors" in str(e)):
                print(f"⚠️  Ошибка загрузки Whisper на MPS: {e}")
                print("🔄 Переключаемся на CPU для Whisper...")
                whisper_device = "cpu"
                self.whisper_model = whisper.load_model(
                    self.whisper_model_name, 
                    device=whisper_device
                )
                print("✅ Стандартная модель Whisper загружена на CPU")
            else:
                raise e
    
    def _load_custom_whisper_model(self, whisper_device: str):
        """Загрузка кастомной модели Whisper через transformers с pipeline API"""
        if not HF_TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers не доступен для загрузки кастомных моделей")
        
        try:
            print(f"🔄 Загружаем кастомную модель Whisper: {self.custom_whisper_model}")
            
            # Настройки типа данных как в официальной документации
            torch_dtype = torch.bfloat16 if whisper_device != "cpu" else torch.float32
            
            # Monkey patching для MPS как в документации
            if whisper_device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    setattr(torch.distributed, "is_initialized", lambda: False)
                    print("🔧 Применен monkey patch для MPS")
                except Exception as e:
                    print(f"⚠️  Не удалось применить monkey patch для MPS: {e}")
            
            # Загружаем модель с рекомендованными параметрами
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "use_safetensors": True
            }
            
            # Добавляем flash attention если доступно (для CUDA)
            if whisper_device == "cuda":
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("🚀 Включен flash_attention_2")
                except Exception:
                    print("⚠️  flash_attention_2 недоступен, используем стандартное внимание")
            
            if Path(self.custom_whisper_model).exists():
                # Локальная модель
                print("🏠 Загружаем локальную кастомную модель...")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    self.custom_whisper_model,
                    **model_kwargs
                )
                self.whisper_processor = WhisperProcessor.from_pretrained(self.custom_whisper_model)
            else:
                # Модель из HuggingFace Hub
                print("🌐 Загружаем кастомную модель из HuggingFace Hub...")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    self.custom_whisper_model,
                    **model_kwargs
                )
                self.whisper_processor = WhisperProcessor.from_pretrained(self.custom_whisper_model)
            
            # Создаем pipeline как в официальной документации
            print("🔄 Создаем ASR pipeline...")
            
            # Настраиваем device для pipeline
            pipeline_device = whisper_device
            if whisper_device == "mps":
                # Для MPS используем device index
                pipeline_device = torch.device(whisper_device)
            
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                max_new_tokens=256,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=pipeline_device
            )
            
            print(f"✅ Кастомная модель Whisper загружена с pipeline API")
            
        except Exception as e:
            if whisper_device == "mps" and ("MPS" in str(e) or "SparseMPS" in str(e)):
                print(f"⚠️  Ошибка загрузки кастомной модели на MPS: {e}")
                print("🔄 Переключаемся на CPU для кастомной модели...")
                whisper_device = "cpu"
                self._load_custom_whisper_model(whisper_device)
            else:
                print(f"❌ Ошибка загрузки кастомной модели: {e}")
                print("🔄 Переключаемся на стандартную модель...")
                self.whisper_model_type = "standard"
                self.custom_whisper_model = None
                self.whisper_pipeline = None
                self._load_standard_whisper_model(whisper_device)
    
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
    
    def transcribe(self, audio_path: str, time_limit: Optional[float] = None) -> Dict:
        """
        Транскрипция аудио с помощью Whisper
        
        Args:
            audio_path: Путь к аудиофайлу
            time_limit: Ограничение времени транскрипции в секундах
            
        Returns:
            Результат транскрипции
        """
        print("🎤 Начинаем транскрипцию...")
        
        # Если указано ограничение по времени, обрезаем аудио
        if time_limit is not None:
            print(f"⏱️  Ограничение времени: {time_limit} секунд")
            audio, sr = librosa.load(audio_path, sr=16000)
            max_samples = int(time_limit * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_wav.name, audio, sr)
                audio_path = temp_wav.name
                print(f"✂️  Аудио обрезано до {time_limit} секунд")
        
        # Выбираем метод транскрипции в зависимости от типа модели
        if self.whisper_model_type == "custom" and self.whisper_pipeline is not None:
            result = self._transcribe_with_pipeline(audio_path)
        elif self.whisper_model_type == "custom" and self.whisper_processor is not None:
            result = self._transcribe_with_custom_model(audio_path)
        else:
            result = self._transcribe_with_standard_model(audio_path)
        
        # Если мы создали временный файл, удаляем его
        if time_limit is not None and audio_path != audio_path:
            os.unlink(audio_path)
        
        return result
    
    def _transcribe_with_standard_model(self, audio_path: str) -> Dict:
        """Транскрипция со стандартной моделью Whisper"""
        # Дополнительные параметры для предотвращения пропуска начала аудио
        transcribe_options = {
            "language": "ru",
            "word_timestamps": True,
            "initial_prompt": "",  # Пустой промпт для предотвращения пропуска
            "temperature": 0.0,    # Детерминированные результаты
            "no_speech_threshold": 0.4,  # Понижаем порог для улучшения детекции речи
            "logprob_threshold": -1.0,   # Понижаем порог для логарифмической вероятности
            "compression_ratio_threshold": 2.4,  # Настройка для предотвращения пропуска
        }
        
        # Для моделей small и medium добавляем дополнительные настройки
        if self.whisper_model_name in ['small', 'medium', 'large']:
            print("🔧 Применяем дополнительные настройки для small/medium/large модели...")
            transcribe_options.update({
                "temperature": 0.1,    # Немного увеличиваем температуру
                "no_speech_threshold": 0.3,  # Еще ниже порог
                "condition_on_previous_text": False,  # Отключаем условие предыдущего текста
            })
        
        result = self.whisper_model.transcribe(
            audio_path,
            **transcribe_options
        )
        
        return result
    
    def _transcribe_with_custom_model(self, audio_path: str) -> Dict:
        """Транскрипция с кастомной моделью через transformers"""
        import librosa
        
        print("🔧 Используем кастомную модель для транскрипции...")
        
        # Загружаем аудио
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Подготавливаем входные данные
        inputs = self.whisper_processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Перемещаем на то же устройство, что и модель (с проверкой совместимости)
        if hasattr(self.whisper_model, 'device') and self.whisper_model.device != torch.device('cpu'):
            try:
                inputs = {k: v.to(self.whisper_model.device) for k, v in inputs.items()}
            except Exception as e:
                print(f"⚠️  Не удалось переместить входные данные на {self.whisper_model.device}: {e}")
                print("🔄 Используем CPU для входных данных...")
        
        # Настройки генерации для русского языка
        generate_kwargs = {
            "language": "russian",
            "task": "transcribe",
            "return_timestamps": True,
            "max_new_tokens": 256,  # Уменьшено с 448 до 256 для избежания превышения лимитов
            "do_sample": False,  # Детерминированная генерация
            "num_beams": 1,      # Greedy search
        }
        
        # Убираем return_timestamps если не поддерживается
        try:
            # Генерируем транскрипцию
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(
                    inputs["input_features"],
                    **generate_kwargs
                )
        except Exception as e:
            if "return_timestamps" in str(e):
                print("⚠️  return_timestamps не поддерживается, используем без временных меток")
                generate_kwargs.pop("return_timestamps", None)
                with torch.no_grad():
                    predicted_ids = self.whisper_model.generate(
                        inputs["input_features"],
                        **generate_kwargs
                    )
            else:
                raise e
        
        # Декодируем результат
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        # Получаем детальную информацию с временными метками
        detailed_result = self._get_detailed_transcription_custom(
            audio_path, inputs, predicted_ids, transcription
        )
        
        return detailed_result
    
    def _get_detailed_transcription_custom(self, audio_path: str, inputs: Dict, 
                                         predicted_ids: torch.Tensor, full_text: str) -> Dict:
        """Получение детальной транскрипции с временными метками для кастомной модели"""
        
        # Пока что возвращаем базовую структуру, совместимую со стандартной моделью
        # В будущем можно добавить более детальную обработку временных меток
        
        # Приблизительно разбиваем на сегменты по длине аудио
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sr
        
        # Простое разбиение на сегменты по словам/предложениям
        sentences = full_text.split('. ')
        segments = []
        
        segment_duration = audio_duration / len(sentences) if sentences else audio_duration
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, audio_duration)
                
                segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": sentence.strip() + ('.' if i < len(sentences) - 1 else '')
                })
        
        # Если нет сегментов, создаем один
        if not segments:
            segments = [{
                "start": 0.0,
                "end": audio_duration,
                "text": full_text
            }]
        
        return {
            "text": full_text,
            "segments": segments,
            "language": "ru"
        }
    
    def _transcribe_with_pipeline(self, audio_path: str) -> Dict:
        """Транскрипция с использованием pipeline API (рекомендовано для кастомных моделей)"""
        print("🔧 Используем pipeline API для транскрипции...")
        
        try:
            # Загружаем аудио как numpy array (правильный формат для pipeline)
            import librosa
            
            print("🔄 Загружаем аудио файл...")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Параметры генерации как в официальной документации
            generate_kwargs = {
                "language": "russian",
                "max_new_tokens": 256
            }
            
            # Запускаем транскрипцию через pipeline
            print("🎤 Выполняем транскрипцию через pipeline...")
            pipeline_result = self.whisper_pipeline(
                audio,  # Передаем numpy array вместо BytesIO
                generate_kwargs=generate_kwargs,
                return_timestamps=True
            )
            
            # Преобразуем результат pipeline в формат, совместимый со стандартной моделью
            result = self._convert_pipeline_result_to_standard_format(pipeline_result, audio_path)
            
            return result
            
        except Exception as e:
            print(f"⚠️  Ошибка транскрипции через pipeline: {e}")
            print("🔄 Переключаемся на альтернативный метод...")
            # Fallback на стандартный метод кастомной модели
            return self._transcribe_with_custom_model(audio_path)
    
    def _convert_pipeline_result_to_standard_format(self, pipeline_result: Dict, audio_path: str) -> Dict:
        """Конвертирует результат pipeline в стандартный формат"""
        # Pipeline возвращает результат в формате:
        # {"text": "...", "chunks": [{"timestamp": (start, end), "text": "..."}]}
        
        text = pipeline_result.get("text", "")
        chunks = pipeline_result.get("chunks", [])
        
        # Создаем сегменты в формате Whisper
        segments = []
        
        if chunks:
            for chunk in chunks:
                timestamp = chunk.get("timestamp")
                chunk_text = chunk.get("text", "")
                
                if timestamp and len(timestamp) >= 2:
                    start_time = timestamp[0] if timestamp[0] is not None else 0.0
                    end_time = timestamp[1] if timestamp[1] is not None else start_time + 1.0
                    
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": chunk_text.strip()
                    })
        
        # Если нет chunks, создаем один сегмент из всего текста
        if not segments and text:
            # Получаем длительность аудио
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=16000)
                audio_duration = len(audio) / sr
            except Exception:
                audio_duration = 30.0  # Fallback
            
            segments = [{
                "start": 0.0,
                "end": audio_duration,
                "text": text.strip()
            }]
        
        return {
            "text": text,
            "segments": segments,
            "language": "ru"
        }
    
    def diarize(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 10, 
                min_segment_duration: float = 0.5) -> Optional[Dict]:
        """
        Улучшенная диаризация аудио с настройками качества
        
        Args:
            audio_path: Путь к аудиофайлу
            min_speakers: Минимальное количество спикеров
            max_speakers: Максимальное количество спикеров
            min_segment_duration: Минимальная длительность сегмента (сек)
            
        Returns:
            Результат диаризации или None если модель недоступна
        """
        if self.diarization_pipeline is None:
            print("⚠️  Диаризация недоступна - модель не загружена")
            return None
            
        print("👥 Начинаем улучшенную диаризацию спикеров...")
        print(f"🔧 Параметры: спикеры {min_speakers}-{max_speakers}, мин. сегмент {min_segment_duration}с")
        
        try:
            # Настройки для улучшения качества диаризации
            diarization_params = {
                "min_speakers": min_speakers,
                "max_speakers": max_speakers
            }
            
            # Запускаем диаризацию с параметрами
            print("🔄 Анализируем аудио...")
            diarization = self.diarization_pipeline(audio_path, **diarization_params)
            
            # Конвертируем результат и применяем фильтры
            speakers = []
            segment_count_before = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment_count_before += 1
                duration = turn.end - turn.start
                
                # Фильтруем слишком короткие сегменты
                if duration >= min_segment_duration:
                    speakers.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                        "duration": duration
                    })
            
            # Постобработка: объединяем соседние сегменты одного спикера
            speakers = self._merge_consecutive_same_speaker(speakers)
            
            # Переименовываем спикеров в понятные имена
            speakers = self._rename_speakers(speakers)
            
            print(f"📊 Диаризация завершена: {segment_count_before} → {len(speakers)} сегментов")
            print(f"👥 Найдено спикеров: {len(set(s['speaker'] for s in speakers))}")
            
            return {
                "speakers": speakers,
                "stats": {
                    "segments_before_filter": segment_count_before,
                    "segments_after_filter": len(speakers),
                    "unique_speakers": len(set(s['speaker'] for s in speakers))
                }
            }
            
        except Exception as e:
            print(f"⚠️  Ошибка диаризации: {e}")
            return None
    
    def _merge_consecutive_same_speaker(self, speakers: List[Dict], gap_threshold: float = 0.3) -> List[Dict]:
        """
        Объединяет соседние сегменты одного спикера, разделенные короткими паузами
        
        Args:
            speakers: Список сегментов спикеров
            gap_threshold: Максимальный промежуток для объединения (сек)
        """
        if not speakers:
            return speakers
        
        # Сортируем по времени начала
        speakers = sorted(speakers, key=lambda x: x['start'])
        merged = [speakers[0]]
        
        for current in speakers[1:]:
            last = merged[-1]
            
            # Если тот же спикер и промежуток небольшой - объединяем
            if (current['speaker'] == last['speaker'] and 
                current['start'] - last['end'] <= gap_threshold):
                
                # Расширяем последний сегмент
                merged[-1] = {
                    "start": last['start'],
                    "end": current['end'],
                    "speaker": last['speaker'],
                    "duration": current['end'] - last['start']
                }
            else:
                merged.append(current)
        
        return merged
    
    def _rename_speakers(self, speakers: List[Dict]) -> List[Dict]:
        """
        Переименовывает спикеров в понятные имена (Спикер 1, Спикер 2, etc.)
        """
        if not speakers:
            return speakers
        
        # Собираем уникальных спикеров в порядке появления
        unique_speakers = []
        for speaker_data in speakers:
            speaker = speaker_data['speaker']
            if speaker not in unique_speakers:
                unique_speakers.append(speaker)
        
        # Создаем маппинг старых имен на новые
        speaker_mapping = {}
        for i, old_name in enumerate(unique_speakers):
            speaker_mapping[old_name] = f"Спикер {i + 1}"
        
        # Применяем новые имена
        for speaker_data in speakers:
            speaker_data['speaker'] = speaker_mapping[speaker_data['speaker']]
        
        return speakers
    
    def _align_transcription_with_speakers(self, transcription: Dict, diarization: Dict, 
                                          alignment_strategy: str = "smart") -> List[Dict]:
        """
        Интеллектуальное совмещение транскрипции с информацией о спикерах
        
        Args:
            transcription: Результат транскрипции
            diarization: Результат диаризации
            alignment_strategy: Стратегия совмещения ('strict', 'smart', 'aggressive')
            
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
        
        print(f"🔄 Совмещение с стратегией '{alignment_strategy}'...")
        
        for t_segment in transcription_segments:
            t_start, t_end = t_segment["start"], t_segment["end"]
            t_mid = (t_start + t_end) / 2
            
            best_speaker = self._find_best_speaker(
                t_start, t_end, t_mid, speaker_segments, alignment_strategy
            )
            
            aligned_segments.append({
                "start": t_start,
                "end": t_end,
                "text": t_segment["text"].strip(),
                "speaker": best_speaker
            })
        
        # Постобработка: устраняем оставшиеся Unknown сегменты
        aligned_segments = self._resolve_unknown_speakers(aligned_segments, speaker_segments)
        
        return aligned_segments
    
    def _find_best_speaker(self, t_start: float, t_end: float, t_mid: float, 
                          speaker_segments: List[Dict], strategy: str) -> str:
        """
        Находит лучшего спикера для транскрипционного сегмента
        """
        best_speaker = "Unknown"
        best_score = 0
        
        for s_segment in speaker_segments:
            s_start, s_end = s_segment["start"], s_segment["end"]
            score = 0
            
            if strategy == "strict":
                # Только строгое пересечение
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap = max(0, overlap_end - overlap_start)
                score = overlap
                
            elif strategy == "smart":
                # Комбинированная стратегия
                # 1. Пересечение (приоритет)
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    score = overlap * 10  # Высокий приоритет пересечениям
                else:
                    # 2. Близость по времени
                    gap_to_start = abs(t_mid - s_start)
                    gap_to_end = abs(t_mid - s_end)
                    min_gap = min(gap_to_start, gap_to_end)
                    
                    # 3. Средняя точка сегмента диаризации
                    s_mid = (s_start + s_end) / 2
                    gap_to_mid = abs(t_mid - s_mid)
                    
                    # Чем ближе, тем выше счет (но меньше чем пересечение)
                    if min_gap < 2.0:  # Максимум 2 секунды разрыва
                        score = max(0, 2.0 - min_gap)
                    elif gap_to_mid < 3.0:  # Альтернатива по средней точке
                        score = max(0, 1.0 - gap_to_mid / 3.0)
                        
            elif strategy == "aggressive":
                # Очень агрессивное совмещение
                # Любое пересечение или близость
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    score = overlap * 10
                else:
                    # Расстояние до ближайшей точки
                    distances = [
                        abs(t_start - s_start), abs(t_start - s_end),
                        abs(t_end - s_start), abs(t_end - s_end),
                        abs(t_mid - s_start), abs(t_mid - s_end)
                    ]
                    min_distance = min(distances)
                    
                    if min_distance < 5.0:  # До 5 секунд разрыва
                        score = max(0, 5.0 - min_distance)
            
            if score > best_score:
                best_score = score
                best_speaker = s_segment["speaker"]
        
        return best_speaker
    
    def _resolve_unknown_speakers(self, segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
        """
        Постобработка для устранения Unknown спикеров
        """
        if not speaker_segments:
            return segments
            
        # Собираем известных спикеров
        known_speakers = [s["speaker"] for s in speaker_segments]
        unknown_segments = [i for i, seg in enumerate(segments) if seg["speaker"] == "Unknown"]
        
        if not unknown_segments:
            return segments
            
        print(f"🔧 Исправляем {len(unknown_segments)} Unknown сегментов...")
        
        for idx in unknown_segments:
            segment = segments[idx]
            t_start, t_end = segment["start"], segment["end"]
            t_mid = (t_start + t_end) / 2
            
            # Стратегия 1: Ближайший по времени известный спикер из диаризации
            best_speaker = self._find_nearest_speaker(t_mid, speaker_segments)
            
            # Стратегия 2: Если не нашли, используем контекст соседних сегментов
            if best_speaker == "Unknown":
                best_speaker = self._find_contextual_speaker(idx, segments, known_speakers)
            
            # Стратегия 3: Самый частый спикер в аудио
            if best_speaker == "Unknown" and known_speakers:
                speaker_counts = {}
                for s in segments:
                    if s["speaker"] != "Unknown":
                        speaker_counts[s["speaker"]] = speaker_counts.get(s["speaker"], 0) + 1
                
                if speaker_counts:
                    best_speaker = max(speaker_counts, key=speaker_counts.get)
                else:
                    best_speaker = known_speakers[0] if known_speakers else "Спикер 1"
            
            segments[idx]["speaker"] = best_speaker
        
        return segments
    
    def _find_nearest_speaker(self, t_mid: float, speaker_segments: List[Dict]) -> str:
        """Находит ближайшего по времени спикера"""
        min_distance = float('inf')
        nearest_speaker = "Unknown"
        
        for s_segment in speaker_segments:
            s_start, s_end = s_segment["start"], s_segment["end"]
            s_mid = (s_start + s_end) / 2
            
            # Расстояние до средней точки сегмента диаризации
            distance = abs(t_mid - s_mid)
            
            if distance < min_distance:
                min_distance = distance
                nearest_speaker = s_segment["speaker"]
        
        # Возвращаем только если расстояние разумное (< 10 секунд)
        return nearest_speaker if min_distance < 10.0 else "Unknown"
    
    def _find_contextual_speaker(self, idx: int, segments: List[Dict], known_speakers: List[str]) -> str:
        """Находит спикера на основе контекста соседних сегментов"""
        # Смотрим на соседние сегменты
        before_speaker = None
        after_speaker = None
        
        # Предыдущий известный спикер
        for i in range(idx - 1, -1, -1):
            if segments[i]["speaker"] != "Unknown":
                before_speaker = segments[i]["speaker"]
                break
        
        # Следующий известный спикер  
        for i in range(idx + 1, len(segments)):
            if segments[i]["speaker"] != "Unknown":
                after_speaker = segments[i]["speaker"]
                break
        
        # Если окружен одним спикером, используем его
        if before_speaker and before_speaker == after_speaker:
            return before_speaker
        
        # Иначе используем ближайшего по времени
        if before_speaker:
            return before_speaker
        elif after_speaker:
            return after_speaker
        
        return "Unknown"
    
    def process(self, audio_path: str, output_dir: str = "output", 
                min_speakers: int = 1, max_speakers: int = 10, 
                min_segment_duration: float = 0.5, alignment_strategy: str = "smart",
                time_limit: Optional[float] = None) -> Dict:
        """
        Полная обработка аудио: транскрипция + диаризация
        
        Args:
            audio_path: Путь к аудиофайлу
            output_dir: Директория для сохранения результатов
            min_speakers: Минимальное количество спикеров
            max_speakers: Максимальное количество спикеров  
            min_segment_duration: Минимальная длительность сегмента
            alignment_strategy: Стратегия совмещения ('strict', 'smart', 'aggressive')
            time_limit: Ограничение времени транскрипции в секундах
            
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
            transcription_result = self.transcribe(prepared_audio, time_limit=time_limit)
            transcription_time = time.time() - start_time
            
            # Диаризация с улучшенными параметрами
            start_time = time.time()
            diarization_result = self.diarize(
                prepared_audio, 
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                min_segment_duration=min_segment_duration
            )
            diarization_time = time.time() - start_time
            
            # Совмещаем результаты с выбранной стратегией
            aligned_segments = self._align_transcription_with_speakers(
                transcription_result, 
                diarization_result,
                alignment_strategy=alignment_strategy
            )
            
            # Подготавливаем итоговый результат
            result = {
                "audio_file": str(Path(audio_path).name),
                "transcription": transcription_result.get("text", ""),
                "segments": aligned_segments,
                "language": transcription_result.get("language", "unknown"),
                "has_speaker_diarization": diarization_result is not None,
                "transcription_time": transcription_time,
                "diarization_time": diarization_time,
                "diarization_stats": diarization_result.get("stats", {}) if diarization_result else {},
                "alignment_strategy": alignment_strategy
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

    def test_transcription_with_different_settings(self, audio_path: str) -> Dict:
        """
        Тестирование транскрипции с разными настройками для диагностики проблем
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Словарь с результатами разных настроек
        """
        print("🧪 Тестируем разные настройки транскрипции...")
        
        settings_to_test = [
            {
                "name": "Стандартные настройки",
                "options": {
                    "language": "ru",
                    "word_timestamps": True
                }
            },
            {
                "name": "Пустой initial_prompt",
                "options": {
                    "language": "ru",
                    "word_timestamps": True,
                    "initial_prompt": ""
                }
            },
            {
                "name": "Низкие пороги",
                "options": {
                    "language": "ru",
                    "word_timestamps": True,
                    "initial_prompt": "",
                    "no_speech_threshold": 0.2,
                    "logprob_threshold": -1.5,
                    "temperature": 0.1
                }
            },
            {
                "name": "Без условия предыдущего текста",
                "options": {
                    "language": "ru",
                    "word_timestamps": True,
                    "initial_prompt": "",
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.3,
                    "temperature": 0.0
                }
            }
        ]
        
        results = {}
        
        for setting in settings_to_test:
            print(f"📝 Тестируем: {setting['name']}")
            try:
                result = self.whisper_model.transcribe(audio_path, **setting['options'])
                
                # Анализируем результат
                segments = result.get('segments', [])
                first_segment_start = segments[0]['start'] if segments else 0
                total_duration = segments[-1]['end'] if segments else 0
                
                results[setting['name']] = {
                    'first_segment_start': first_segment_start,
                    'total_segments': len(segments),
                    'total_duration': total_duration,
                    'text_preview': result.get('text', '')[:100] + '...',
                    'full_result': result
                }
                
                print(f"   ✅ Первый сегмент начинается с: {first_segment_start:.1f}с")
                print(f"   📊 Всего сегментов: {len(segments)}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
                results[setting['name']] = {'error': str(e)}
        
        return results


@click.command()
@click.argument('audio_file')
@click.option('--model', '-m', default='large', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Модель Whisper для использования (только для стандартных моделей)')
@click.option('--custom-model', '--custom-whisper-model', 
              help='Путь к кастомной модели Whisper (HuggingFace format) или HF model ID')
@click.option('--output', '-o', default='output', 
              help='Директория для сохранения результатов')
@click.option('--hf-token', envvar='HUGGINGFACE_TOKEN', 
              help='HuggingFace токен (можно задать в переменной HUGGINGFACE_TOKEN)')
@click.option('--local-models', envvar='LOCAL_MODELS_DIR',
              help='Директория с локальными моделями (можно задать в переменной LOCAL_MODELS_DIR)')
@click.option('--device', default=None, type=click.Choice(['cpu', 'cuda', 'mps']), 
              help='Устройство для инференса (cpu, cuda, mps)')
@click.option('--min-speakers', default=1, type=int,
              help='Минимальное количество спикеров (по умолчанию: 1)')
@click.option('--max-speakers', default=10, type=int,
              help='Максимальное количество спикеров (по умолчанию: 10)')
@click.option('--min-segment', default=0.5, type=float,
              help='Минимальная длительность сегмента в секундах (по умолчанию: 0.5)')
@click.option('--alignment-strategy', default='smart', type=click.Choice(['strict', 'smart', 'aggressive']),
              help='Стратегия совмещения (strict, smart, aggressive)')
@click.option('--test-transcription', is_flag=True,
              help='Протестировать разные настройки транскрипции для диагностики проблем')
@click.option('--time-limit', type=float,
              help='Ограничение времени транскрипции в секундах (например, 3500 для транскрипции первых 3500 секунд)')
def main(audio_file: str, model: str, custom_model: Optional[str], output: str, hf_token: Optional[str], 
         local_models: Optional[str], device: Optional[str], min_speakers: int, 
         max_speakers: int, min_segment: float, alignment_strategy: str, test_transcription: bool,
         time_limit: Optional[float]):
    """
    Пайплайн транскрипции и диаризации аудио с улучшенными настройками
    
    AUDIO_FILE: Путь к аудиофайлу для обработки
    """
    
    # Проверяем и корректируем путь к аудиофайлу
    input_dir = Path("input")
    if input_dir.exists() and not Path(audio_file).exists():
        # Ищем файл в input/ директории
        input_file_path = input_dir / audio_file
        if input_file_path.exists():
            audio_file = str(input_file_path)
        else:
            print(f"❌ Файл не найден: {audio_file}")
            print(f"🔍 Искал в: {Path(audio_file).absolute()}")
            print(f"🔍 Искал в: {input_file_path.absolute()}")
            
            # Покажем доступные файлы в input/
            if input_dir.exists():
                input_files = list(input_dir.glob("*"))
                if input_files:
                    print(f"📁 Доступные файлы в input/:")
                    for f in input_files:
                        if f.is_file():
                            print(f"   • {f.name}")
                else:
                    print(f"📁 Папка input/ пуста")
            sys.exit(1)
    elif not Path(audio_file).exists():
        print(f"❌ Файл не найден: {audio_file}")
        sys.exit(1)
    
    print("🎵 Whisper + PyAnnote Audio Pipeline (Улучшенная диаризация + Кастомные модели)")
    print(f"📁 Обрабатываем: {audio_file}")
    
    if custom_model:
        print(f"🧠 Кастомная модель Whisper: {custom_model}")
    else:
        print(f"🧠 Стандартная модель Whisper: {model}")
    
    print(f"📂 Результаты будут сохранены в: {output}")
    print(f"👥 Настройки диаризации: {min_speakers}-{max_speakers} спикеров, мин. сегмент {min_segment}с")
    
    if time_limit:
        print(f"⏱️  Ограничение времени: {time_limit} секунд")
    
    if local_models:
        print(f"🏠 Используем локальные модели из: {local_models}")
    elif not hf_token:
        print("⚠️  HuggingFace токен не найден. Диаризация будет недоступна.")
        print("💡 Установите токен: export HUGGINGFACE_TOKEN=your_token")
        print("💡 Или скачайте модели локально: python download_models.py")
    
    # Проверяем совместимость опций
    if custom_model and not HF_TRANSFORMERS_AVAILABLE:
        print("❌ Для использования кастомных моделей нужна библиотека transformers")
        print("💡 Установите: pip install transformers")
        sys.exit(1)
    
    try:
        # Создаем процессор
        processor = AudioProcessor(
            whisper_model=model, 
            hf_token=hf_token,
            local_models_dir=local_models,
            device=device,
            custom_whisper_model=custom_model
        )
        
        # Если включено тестирование, запускаем диагностику
        if test_transcription:
            print("\n🧪 Режим тестирования настроек транскрипции")
            print("=" * 50)
            
            test_results = processor.test_transcription_with_different_settings(audio_file)
            
            print("\n📊 Результаты тестирования:")
            print("=" * 50)
            
            for setting_name, result in test_results.items():
                print(f"\n🔧 {setting_name}:")
                if 'error' in result:
                    print(f"   ❌ Ошибка: {result['error']}")
                else:
                    print(f"   🕒 Начало первого сегмента: {result['first_segment_start']:.1f}с")
                    print(f"   📊 Количество сегментов: {result['total_segments']}")
                    print(f"   ⏱️  Общая длительность: {result['total_duration']:.1f}с")
                    print(f"   📝 Превью текста: {result['text_preview']}")
            
            # Анализ результатов
            best_start_time = float('inf')
            best_setting = None
            
            for setting_name, result in test_results.items():
                if 'error' not in result and result['first_segment_start'] < best_start_time:
                    best_start_time = result['first_segment_start']
                    best_setting = setting_name
            
            if best_setting:
                print(f"\n🏆 Лучший результат: '{best_setting}' (начинается с {best_start_time:.1f}с)")
                
                if best_start_time > 5.0:
                    print("⚠️  Обнаружена проблема с пропуском начала аудио!")
                    print("💡 Рекомендации:")
                    print("   • Попробуйте использовать model='base' вместо medium/small")
                    print("   • Проверьте качество аудио в первые 30 секунд")
                    print("   • Убедитесь, что аудио не содержит длинных пауз в начале")
                else:
                    print("✅ Проблем с пропуском начала не обнаружено")
            
            return
        
        # Обрабатываем аудио с улучшенными настройками
        result = processor.process(
            audio_file, 
            output,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            min_segment_duration=min_segment,
            alignment_strategy=alignment_strategy,
            time_limit=time_limit
        )
        
        print("\n✅ Обработка завершена!")
        print(f"📊 Найдено сегментов: {len(result['segments'])}")
        
        if result['has_speaker_diarization']:
            speakers = set(seg['speaker'] for seg in result['segments'])
            unknown_count = sum(1 for seg in result['segments'] if seg['speaker'] == 'Unknown')
            
            print(f"👥 Обнаружено спикеров: {len(speakers)} ({', '.join(speakers)})")
            print(f"🎯 Стратегия совмещения: {result['alignment_strategy']}")
            
            if unknown_count > 0:
                print(f"⚠️  Unknown сегментов: {unknown_count} ({unknown_count/len(result['segments'])*100:.1f}%)")
            else:
                print("✅ Все сегменты успешно привязаны к спикерам!")
            
            # Показываем статистику диаризации
            stats = result.get('diarization_stats', {})
            if stats:
                print(f"📈 Статистика диаризации:")
                print(f"   • Сегментов до фильтрации: {stats.get('segments_before_filter', 'N/A')}")
                print(f"   • Сегментов после фильтрации: {stats.get('segments_after_filter', 'N/A')}")
                print(f"   • Уникальных спикеров: {stats.get('unique_speakers', 'N/A')}")
        
        print(f"🔤 Язык: {result['language']}")
        print(f"📝 Полный текст: {len(result['transcription'])} символов")
        print(f"⏱️  Время транскрипции: {result['transcription_time']:.1f}с")
        print(f"⏱️  Время диаризации: {result['diarization_time']:.1f}с")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 