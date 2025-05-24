#!/bin/bash

# Docker and NVIDIA Docker Runtime Installer
# Автоматическая установка Docker и NVIDIA Docker runtime

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Docker and NVIDIA Docker Runtime Installer${NC}"
echo -e "${BLUE}===============================================${NC}"

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]] || [[ -n "$WINDIR" ]]; then
    OS="windows"
else
    echo -e "${RED}❌ Неподдерживаемая ОС: $OSTYPE${NC}"
    echo -e "${YELLOW}💡 Попробуйте запустить в WSL2 или Git Bash${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Обнаружена ОС: $OS${NC}"

# Function to install Docker on Linux
install_docker_linux() {
    echo -e "${BLUE}📦 Устанавливаем Docker на Linux...${NC}"
    
    # Update package index
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}✅ Docker установлен${NC}"
}

# Function to install NVIDIA Docker runtime on Linux
install_nvidia_docker_linux() {
    echo -e "${BLUE}🎮 Устанавливаем NVIDIA Docker runtime...${NC}"
    
    # Check if NVIDIA drivers are installed
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠️  NVIDIA драйверы не найдены. Установите их сначала:${NC}"
        echo -e "${YELLOW}   sudo apt install nvidia-driver-XXX${NC}"
        echo -e "${YELLOW}   (где XXX - версия драйвера)${NC}"
        return 1
    fi
    
    # Add NVIDIA Docker repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install NVIDIA Docker runtime
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    
    # Restart Docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✅ NVIDIA Docker runtime установлен${NC}"
}

# Function to install Docker on macOS
install_docker_macos() {
    echo -e "${BLUE}📦 Устанавливаем Docker на macOS...${NC}"
    
    if command -v brew &> /dev/null; then
        echo -e "${BLUE}🍺 Используем Homebrew...${NC}"
        brew install --cask docker
    else
        echo -e "${YELLOW}⚠️  Homebrew не найден${NC}"
        echo -e "${YELLOW}💡 Скачайте Docker Desktop с https://www.docker.com/products/docker-desktop${NC}"
        echo -e "${YELLOW}💡 Или установите Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Docker установлен${NC}"
    echo -e "${YELLOW}💡 Запустите Docker Desktop из Applications${NC}"
}

# Function to install Docker on Windows
install_docker_windows() {
    echo -e "${BLUE}📦 Устанавливаем Docker на Windows...${NC}"
    
    # Check if we're in WSL
    if grep -q Microsoft /proc/version 2>/dev/null; then
        echo -e "${BLUE}🐧 Обнаружен WSL, используем Linux установку...${NC}"
        install_docker_linux
        return
    fi
    
    echo -e "${YELLOW}💡 Для Windows нужно установить Docker Desktop вручную:${NC}"
    echo -e "${BLUE}1. Скачайте Docker Desktop: https://www.docker.com/products/docker-desktop${NC}"
    echo -e "${BLUE}2. Запустите установщик с правами администратора${NC}"
    echo -e "${BLUE}3. Перезагрузите компьютер после установки${NC}"
    echo -e "${BLUE}4. Запустите Docker Desktop${NC}"
    echo ""
    echo -e "${YELLOW}Для автоматической установки используйте PowerShell скрипт:${NC}"
    echo -e "${GREEN}   PowerShell -ExecutionPolicy Bypass -File install-docker.ps1${NC}"
    
    # Check if winget is available for automatic installation
    if command -v winget &> /dev/null; then
        echo ""
        echo -e "${BLUE}💡 Обнаружен winget, можно установить автоматически:${NC}"
        read -p "Установить Docker Desktop через winget? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            winget install Docker.DockerDesktop
            echo -e "${GREEN}✅ Docker Desktop установлен${NC}"
            echo -e "${YELLOW}💡 Перезагрузите компьютер и запустите Docker Desktop${NC}"
        fi
    fi
}

# Function to install NVIDIA Docker runtime on Windows
install_nvidia_docker_windows() {
    echo -e "${BLUE}🎮 Настройка NVIDIA GPU для Windows...${NC}"
    
    # Check if we're in WSL
    if grep -q Microsoft /proc/version 2>/dev/null; then
        echo -e "${BLUE}🐧 WSL обнаружен, проверяем NVIDIA поддержку...${NC}"
        if command -v nvidia-smi &> /dev/null; then
            echo -e "${GREEN}✅ NVIDIA драйверы доступны в WSL${NC}"
        else
            echo -e "${YELLOW}⚠️  NVIDIA драйверы не найдены в WSL${NC}"
            echo -e "${YELLOW}💡 Установите NVIDIA драйверы для Windows с WSL поддержкой${NC}"
        fi
        return
    fi
    
    echo -e "${YELLOW}💡 Для NVIDIA GPU поддержки на Windows:${NC}"
    echo -e "${BLUE}1. Установите актуальные NVIDIA драйверы (версия 450.80.02+)${NC}"
    echo -e "${BLUE}2. Docker Desktop автоматически поддерживает NVIDIA GPU${NC}"
    echo -e "${BLUE}3. В Docker Desktop включите WSL2 backend${NC}"
    echo ""
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA драйверы найдены${NC}"
        nvidia-smi
    else
        echo -e "${YELLOW}⚠️  NVIDIA драйверы не найдены${NC}"
        echo -e "${YELLOW}💡 Скачайте драйверы: https://www.nvidia.com/drivers${NC}"
    fi
}

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ Docker уже установлен${NC}"
    docker --version
else
    if [[ "$OS" == "linux" ]]; then
        install_docker_linux
    elif [[ "$OS" == "macos" ]]; then
        install_docker_macos
    elif [[ "$OS" == "windows" ]]; then
        install_docker_windows
    fi
fi

# Check if Docker Compose is available
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo -e "${GREEN}✅ Docker Compose доступен${NC}"
else
    echo -e "${YELLOW}⚠️  Docker Compose не найден${NC}"
    if [[ "$OS" == "linux" ]]; then
        echo -e "${BLUE}📦 Устанавливаем Docker Compose...${NC}"
        sudo apt-get install -y docker-compose-plugin
    fi
fi

# Install NVIDIA Docker runtime
if [[ "$OS" == "linux" ]]; then
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✅ NVIDIA Docker runtime уже установлен${NC}"
    else
        install_nvidia_docker_linux
    fi
elif [[ "$OS" == "macos" ]]; then
    echo -e "${YELLOW}💡 На macOS используйте CPU версию или Rosetta для совместимости${NC}"
elif [[ "$OS" == "windows" ]]; then
    install_nvidia_docker_windows
fi

echo -e "${GREEN}🎉 Установка завершена!${NC}"
echo -e "${BLUE}💡 Возможно потребуется перезагрузка или повторный вход в систему${NC}"
echo -e "${BLUE}💡 Для проверки запустите: docker run hello-world${NC}"

if [[ "$OS" == "linux" ]] || [[ "$OS" == "windows" ]]; then
    echo -e "${BLUE}💡 Для проверки NVIDIA: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi${NC}"
fi 