# Исправление проблемы с HuggingFace токеном

## Проблема
При запуске `.\rebuild-and-test.ps1` не подтягивался `hf_token` из env файла.

## Причина
PowerShell скрипты не загружали переменные окружения из `.env` файла перед запуском Docker Compose.

## Исправления

### 1. Обновлены PowerShell скрипты
- `run.ps1` - добавлена функция `Load-EnvFile`
- `rebuild-and-test.ps1` - добавлена функция `Load-EnvFile`

### 2. Обновлен docker-compose.yml
- Добавлена секция `env_file: - .env` для каждого сервиса
- Теперь переменные загружаются двумя способами:
  - Через `env_file` (Docker Compose автоматически)
  - Через `environment: HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}` (из системных переменных)

### 3. Исправлен .env файл
- Убран лишний текст, который мог мешать парсингу
- Оставлена только строка: `HUGGINGFACE_TOKEN=hf_your_token`

## Проверка исправления

1. **Проверить .env файл:**
   ```bash
   cat .env
   ```
   Должен содержать: `HUGGINGFACE_TOKEN=hf_ваш_токен`

2. **Запустить тест загрузки переменных:**
   ```powershell
   .\test-env-loading.ps1
   ```

3. **Проверить диагностику токена:**
   ```powershell
   .\check-hf-token.ps1
   ```

4. **Запустить пересборку:**
   ```powershell
   .\rebuild-and-test.ps1
   ```

## Функция Load-EnvFile

Добавленная функция загружает переменные из `.env` файла:

```powershell
function Load-EnvFile {
    param([string]$FilePath = ".env")
    
    if (Test-Path $FilePath) {
        Get-Content $FilePath | ForEach-Object {
            if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)\s*$") {
                $name = $matches[1].Trim()
                $value = $matches[2].Trim()
                $value = $value -replace '^["'']|["'']$', ''
                Set-Item -Path "env:$name" -Value $value
            }
        }
    }
}
```

## Приоритет переменных окружения

1. Аргументы командной строки: `--hf-token your_token`
2. Системные переменные окружения: `$env:HUGGINGFACE_TOKEN`
3. Файл .env: `HUGGINGFACE_TOKEN=your_token`

Теперь все три способа работают корректно! 