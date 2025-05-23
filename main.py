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
        
        # Пытаемся загрузить модель Whisper
        whisper_device = self.device
        try:
            self.whisper_model = whisper.load_model(
                self.whisper_model_name, 
                device=whisper_device
            )
            print(f"✅ Модель Whisper загружена на {whisper_device}")
        except Exception as e:
            if whisper_device == "mps" and ("SparseMPS" in str(e) or "aten::empty.memory_format" in str(e) or "_sparse_coo_tensor_with_dims_and_tensors" in str(e)):
                print(f"⚠️  Ошибка загрузки Whisper на MPS: {e}")
                print("🔄 Переключаемся на CPU для Whisper...")
                whisper_device = "cpu"
                self.whisper_model = whisper.load_model(
                    self.whisper_model_name, 
                    device=whisper_device
                )
                print("✅ Модель Whisper загружена на CPU")
            else:
                raise e
        
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
        if self.whisper_model_name in ['small', 'medium']:
            print("🔧 Применяем дополнительные настройки для small/medium модели...")
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
                min_segment_duration: float = 0.5, alignment_strategy: str = "smart") -> Dict:
        """
        Полная обработка аудио: транскрипция + диаризация
        
        Args:
            audio_path: Путь к аудиофайлу
            output_dir: Директория для сохранения результатов
            min_speakers: Минимальное количество спикеров
            max_speakers: Максимальное количество спикеров  
            min_segment_duration: Минимальная длительность сегмента
            alignment_strategy: Стратегия совмещения ('strict', 'smart', 'aggressive')
            
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
def main(audio_file: str, model: str, output: str, hf_token: Optional[str], 
         local_models: Optional[str], device: Optional[str], min_speakers: int, 
         max_speakers: int, min_segment: float, alignment_strategy: str, test_transcription: bool):
    """
    Пайплайн транскрипции и диаризации аудио с улучшенными настройками
    
    AUDIO_FILE: Путь к аудиофайлу для обработки
    """
    
    print("🎵 Whisper + PyAnnote Audio Pipeline (Улучшенная диаризация)")
    print(f"📁 Обрабатываем: {audio_file}")
    print(f"🧠 Модель Whisper: {model}")
    print(f"📂 Результаты будут сохранены в: {output}")
    print(f"👥 Настройки диаризации: {min_speakers}-{max_speakers} спикеров, мин. сегмент {min_segment}с")
    
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
            alignment_strategy=alignment_strategy
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