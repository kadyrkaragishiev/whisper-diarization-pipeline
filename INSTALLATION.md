# 📦 Инструкции по установке

## 🚀 Автоматическая установка (рекомендуется)

### Для всех ОС:
```bash
git clone <repository-url>
cd whisper-diarization-pipeline
./setup-complete.sh
```

## 🖥️ Ручная установка по ОС

### 🐧 Linux (Ubuntu/Debian)

1. **Установка Docker:**
```bash
# Автоматически
./install-docker.sh

# Или вручную
sudo apt update
sudo apt install docker.io docker-compose-plugin
sudo usermod -aG docker $USER
```

2. **NVIDIA GPU поддержка (опционально):**
```bash
# Установите NVIDIA драйверы
sudo apt install nvidia-driver-535  # или новее

# Установите NVIDIA Docker runtime
sudo apt install nvidia-docker2
sudo systemctl restart docker

# Проверка
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

3. **Запуск:**
```bash
./run.sh your_audio.wav
```

### 🍎 macOS

1. **Установка Docker Desktop:**
```bash
# Через Homebrew
brew install --cask docker

# Или скачайте с https://www.docker.com/products/docker-desktop
```

2. **Запустите Docker Desktop** из Applications

3. **Запуск (CPU версия):**
```bash
./run.sh your_audio.wav
```

**Примечание:** На macOS используется CPU версия, так как NVIDIA GPU недоступны.

### 🪟 Windows

#### Автоматическая установка (рекомендуется):
```powershell
# В PowerShell как администратор
git clone <repository-url>
cd whisper-diarization-pipeline
.\setup-complete.ps1
```

#### Ручная установка:

1. **Установка Docker Desktop:**
```powershell
# Автоматически через PowerShell
.\install-docker.ps1

# Или вручную:
# Скачайте с https://www.docker.com/products/docker-desktop
# Установите и перезагрузите систему
```

2. **WSL2 (рекомендуется для лучшей производительности):**
```powershell
# В PowerShell как администратор
wsl --install
# Перезагрузите компьютер
# Установите Ubuntu из Microsoft Store
```

3. **NVIDIA GPU поддержка (для RTX 3080):**
   - Установите актуальные NVIDIA драйверы (версия 472.12+)
   - Docker Desktop автоматически поддерживает NVIDIA GPU
   - Включите WSL2 backend в Docker Desktop

4. **Запуск:**
```powershell
# PowerShell
.\run.ps1 your_audio.wav

# Или в WSL2/Git Bash
./run.sh your_audio.wav
```

## 🔧 Проверка установки

### Проверка Docker:
```bash
docker --version
docker run hello-world
```

### Проверка GPU поддержки:
```bash
# Linux
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Windows с WSL2
nvidia-smi.exe
```

### Проверка системы:
```bash
./run.sh  # Показывает справку и проверяет систему
```

## 🐛 Устранение проблем

### Docker не запускается:
- **Linux:** `sudo systemctl start docker`
- **macOS:** Запустите Docker Desktop
- **Windows:** Запустите Docker Desktop

### Нет прав доступа к Docker:
```bash
sudo usermod -aG docker $USER
# Перезайдите в систему
```

### NVIDIA GPU не работает:
1. Проверьте драйверы: `nvidia-smi`
2. Установите NVIDIA Docker runtime
3. Перезапустите Docker: `sudo systemctl restart docker`

### Ошибки сборки образа:
- Проверьте интернет соединение
- Очистите Docker кэш: `docker system prune -a`
- Попробуйте CPU версию: `docker-compose --profile cpu build`

## 📋 Системные требования

### Минимальные:
- **CPU:** 4 ядра
- **RAM:** 8GB
- **Диск:** 10GB свободного места
- **ОС:** Linux, macOS, Windows 10+

### Рекомендуемые:
- **CPU:** 8+ ядер
- **RAM:** 16GB+
- **GPU:** NVIDIA с 4GB+ VRAM
- **Диск:** SSD с 20GB+ свободного места

### Для GPU ускорения:
- **NVIDIA GPU** с CUDA Compute Capability 3.5+
- **VRAM:** 4GB+ (8GB+ для больших моделей)
- **CUDA:** 12.1+
- **Драйверы:** 525.60.13+

## 🚀 Оптимизация производительности

### CPU версия:
- Используйте модель `base` для баланса скорости/качества
- Ограничьте количество спикеров: `--max-speakers 3`
- Для длинных файлов: `--time-limit 3600`

### GPU версия:
- Используйте модель `large` для лучшего качества
- Увеличьте batch size в настройках
- Мониторьте использование VRAM

### Общие советы:
- Конвертируйте аудио в WAV 16kHz mono
- Разбивайте очень длинные файлы (>2 часов)
- Используйте SSD для временных файлов 