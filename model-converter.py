import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Используем рекомендованные параметры из официальной документации
torch_dtype = torch.bfloat16

print("🔄 Загружаем модель antony66/whisper-large-v3-russian с оптимальными параметрами...")

model = WhisperForConditionalGeneration.from_pretrained(
    "antony66/whisper-large-v3-russian", 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)

processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")

print("💾 Сохраняем модель локально...")
model.save_pretrained("./whisper-large-v3-russian-pt")
processor.save_pretrained("./whisper-large-v3-russian-pt")

print("✅ Модель и процессор сохранены в ./whisper-large-v3-russian-pt")