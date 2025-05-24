#!/bin/bash

# Whisper Diarization Pipeline - Universal Launcher
# Автоматически определяет наличие NVIDIA GPU и запускает соответствующую версию

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎤 Whisper Diarization Pipeline${NC}"
echo -e "${BLUE}================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker не установлен. Установите Docker и попробуйте снова.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose не найден. Установите Docker Compose и попробуйте снова.${NC}"
    exit 1
fi

# Function to check NVIDIA GPU availability
check_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo -e "${GREEN}✅ NVIDIA GPU обнаружена${NC}"
            return 0
        fi
    fi
    echo -e "${YELLOW}⚠️  NVIDIA GPU не обнаружена или драйверы не установлены${NC}"
    return 1
}

# Function to check NVIDIA Docker runtime
check_nvidia_docker() {
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✅ NVIDIA Docker runtime доступен${NC}"
        return 0
    fi
    echo -e "${YELLOW}⚠️  NVIDIA Docker runtime не найден${NC}"
    return 1
}

# Create necessary directories
echo -e "${BLUE}📁 Создаем необходимые директории...${NC}"
mkdir -p input output models

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}📝 Создаем .env файл...${NC}"
    cp env.example .env
    echo -e "${YELLOW}💡 Отредактируйте .env файл и добавьте ваш HUGGINGFACE_TOKEN если нужно${NC}"
fi

# Determine which version to use
USE_GPU=false
if check_nvidia_gpu && check_nvidia_docker; then
    USE_GPU=true
    PROFILE="gpu"
    SERVICE="whisper-diarization-gpu"
    echo -e "${GREEN}🚀 Запускаем GPU версию${NC}"
else
    PROFILE="cpu"
    SERVICE="whisper-diarization-cpu"
    echo -e "${YELLOW}🚀 Запускаем CPU версию${NC}"
fi

# Build the image
echo -e "${BLUE}🔨 Собираем Docker образ...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose --profile $PROFILE build $SERVICE
else
    docker compose --profile $PROFILE build $SERVICE
fi

# Function to run the pipeline
run_pipeline() {
    local audio_file="$1"
    shift
    local args="$@"
    
    if [ ! -f "input/$audio_file" ]; then
        echo -e "${RED}❌ Файл input/$audio_file не найден${NC}"
        echo -e "${YELLOW}💡 Поместите аудиофайл в директорию input/${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🎵 Обрабатываем: $audio_file${NC}"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose --profile $PROFILE run --rm $SERVICE "$audio_file" $args
    else
        docker compose --profile $PROFILE run --rm $SERVICE "$audio_file" $args
    fi
}

# Check command line arguments
if [ $# -eq 0 ]; then
    echo -e "${BLUE}📖 Показываем справку...${NC}"
    if command -v docker-compose &> /dev/null; then
        docker-compose --profile $PROFILE run --rm $SERVICE --help
    else
        docker compose --profile $PROFILE run --rm $SERVICE --help
    fi
    echo ""
    echo -e "${YELLOW}Примеры использования:${NC}"
    echo -e "${GREEN}  ./run.sh audio.wav${NC}                          # Базовая обработка"
    echo -e "${GREEN}  ./run.sh audio.wav --model large${NC}            # Использовать большую модель"
    echo -e "${GREEN}  ./run.sh audio.wav --max-speakers 5${NC}         # Максимум 5 спикеров"
    echo -e "${GREEN}  ./run.sh audio.wav --custom-model path/model${NC} # Кастомная модель"
    echo ""
    echo -e "${BLUE}💡 Поместите аудиофайлы в директорию input/${NC}"
    echo -e "${BLUE}📁 Результаты будут сохранены в директории output/${NC}"
else
    run_pipeline "$@"
fi 