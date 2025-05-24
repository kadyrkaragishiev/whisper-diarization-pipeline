#!/bin/bash

# Complete Whisper Diarization Pipeline Setup
# Полная автоматическая установка системы

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🎤 Whisper Diarization Pipeline - Complete Setup${NC}"
echo -e "${PURPLE}=================================================${NC}"
echo ""
echo -e "${BLUE}Этот скрипт автоматически установит:${NC}"
echo -e "${BLUE}• Docker и Docker Compose${NC}"
echo -e "${BLUE}• NVIDIA Docker runtime (если доступно)${NC}"
echo -e "${BLUE}• Соберет Docker образы${NC}"
echo -e "${BLUE}• Настроит окружение${NC}"
echo ""

# Ask for confirmation
read -p "Продолжить установку? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Установка отменена${NC}"
    exit 0
fi

echo -e "${BLUE}🚀 Начинаем установку...${NC}"

# Step 1: Install Docker
echo -e "${BLUE}📦 Шаг 1: Установка Docker...${NC}"
if [ -f "./install-docker.sh" ]; then
    chmod +x install-docker.sh
    ./install-docker.sh
else
    echo -e "${RED}❌ Файл install-docker.sh не найден${NC}"
    exit 1
fi

# Step 2: Create directories
echo -e "${BLUE}📁 Шаг 2: Создание директорий...${NC}"
mkdir -p input output models

# Step 3: Setup environment
echo -e "${BLUE}⚙️  Шаг 3: Настройка окружения...${NC}"
if [ ! -f ".env" ]; then
    cp env.example .env
    echo -e "${YELLOW}💡 Создан файл .env из шаблона${NC}"
    echo -e "${YELLOW}💡 Отредактируйте .env и добавьте HUGGINGFACE_TOKEN если нужно${NC}"
else
    echo -e "${GREEN}✅ Файл .env уже существует${NC}"
fi

# Step 4: Test Docker
echo -e "${BLUE}🧪 Шаг 4: Проверка Docker...${NC}"
if docker run --rm hello-world > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker работает корректно${NC}"
else
    echo -e "${RED}❌ Проблемы с Docker${NC}"
    echo -e "${YELLOW}💡 Возможно нужно перезагрузиться или добавить пользователя в группу docker${NC}"
    echo -e "${YELLOW}💡 Команда: sudo usermod -aG docker \$USER${NC}"
fi

# Step 5: Check GPU support
echo -e "${BLUE}🎮 Шаг 5: Проверка GPU поддержки...${NC}"
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✅ NVIDIA GPU и Docker runtime доступны${NC}"
        GPU_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️  NVIDIA GPU найдена, но Docker runtime не настроен${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  NVIDIA GPU не обнаружена${NC}"
fi

# Step 6: Build Docker images
echo -e "${BLUE}🔨 Шаг 6: Сборка Docker образов...${NC}"

if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${GREEN}🚀 Собираем GPU версию...${NC}"
    if command -v docker-compose &> /dev/null; then
        docker-compose --profile gpu build whisper-diarization-gpu
    else
        docker compose --profile gpu build whisper-diarization-gpu
    fi
    RECOMMENDED_VERSION="GPU"
else
    echo -e "${YELLOW}🚀 Собираем CPU версию...${NC}"
    if command -v docker-compose &> /dev/null; then
        docker-compose --profile cpu build whisper-diarization-cpu
    else
        docker compose --profile cpu build whisper-diarization-cpu
    fi
    RECOMMENDED_VERSION="CPU"
fi

# Step 7: Test the pipeline
echo -e "${BLUE}🧪 Шаг 7: Тестирование системы...${NC}"
echo -e "${BLUE}Показываем справку...${NC}"
chmod +x run.sh
./run.sh

echo ""
echo -e "${GREEN}🎉 Установка завершена успешно!${NC}"
echo ""
echo -e "${PURPLE}📋 Что дальше:${NC}"
echo ""
echo -e "${BLUE}1. Поместите аудиофайл в директорию input/:${NC}"
echo -e "${GREEN}   cp your_audio.wav input/${NC}"
echo ""
echo -e "${BLUE}2. Запустите обработку:${NC}"
echo -e "${GREEN}   ./run.sh your_audio.wav${NC}"
echo ""
echo -e "${BLUE}3. Результаты будут в директории output/${NC}"
echo ""
echo -e "${BLUE}💡 Рекомендуемая версия: ${RECOMMENDED_VERSION}${NC}"
echo ""
if [ "$GPU_AVAILABLE" = false ]; then
    echo -e "${YELLOW}💡 Для ускорения установите NVIDIA GPU и драйверы${NC}"
fi
echo ""
echo -e "${BLUE}📖 Подробная документация: README.md${NC}"
echo -e "${BLUE}🐛 Проблемы? Проверьте раздел 'Устранение неполадок' в README.md${NC}"
echo ""
echo -e "${PURPLE}Готово к использованию! 🚀${NC}" 