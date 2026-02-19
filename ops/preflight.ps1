Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $rootDir

if (-not (Test-Path ".env")) {
    throw ".env is missing. Copy .env.example to .env first."
}

docker --version | Out-Null
docker compose version | Out-Null

function Get-EnvValue {
    param([string]$Key)
    $line = Select-String -Path ".env" -Pattern "^$Key=" | Select-Object -First 1
    if ($null -eq $line) { return "" }
    return ($line.Line -replace "^$Key=", "")
}

function Require-NonEmpty {
    param([string]$Key)
    $value = Get-EnvValue $Key
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing required .env value: $Key"
    }
}

function Reject-Placeholder {
    param([string]$Key)
    $value = Get-EnvValue $Key
    if ($value -in @("change_me", "change_me_admin_key", "change_me_user_key")) {
        throw "Placeholder value detected for $Key; set a real value first."
    }
}

function Ensure-PathWritable {
    param(
        [string]$Label,
        [string]$PathValue
    )
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return
    }
    New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
}

Require-NonEmpty "POSTGRES_PASSWORD"
Require-NonEmpty "DATABASE_URL"
Require-NonEmpty "ADMIN_API_KEYS"
Require-NonEmpty "USER_API_KEYS"
Require-NonEmpty "FRONTEND_PROXY_API_KEY"

Reject-Placeholder "POSTGRES_PASSWORD"
Reject-Placeholder "ADMIN_API_KEYS"
Reject-Placeholder "USER_API_KEYS"
Reject-Placeholder "FRONTEND_PROXY_API_KEY"

foreach ($pathKey in @("POSTGRES_DATA_PATH", "QDRANT_STORAGE_PATH", "DROP_FOLDER", "OBSIDIAN_VAULT_PATH", "BACKUP_PATH", "LOG_PATH", "LOCALAI_MODELS_PATH")) {
    Ensure-PathWritable $pathKey (Get-EnvValue $pathKey)
}

$frontendPort = Get-EnvValue "FRONTEND_HOST_PORT"
if ([string]::IsNullOrWhiteSpace($frontendPort)) { $frontendPort = "3000" }
$backendPort = Get-EnvValue "BACKEND_HOST_PORT"
if ([string]::IsNullOrWhiteSpace($backendPort)) { $backendPort = "8001" }
if ($frontendPort -eq $backendPort) {
    throw "FRONTEND_HOST_PORT and BACKEND_HOST_PORT cannot be the same ($frontendPort)."
}

$listening = netstat -ano -p tcp
if ($listening -match "LISTENING\s+\S+:$frontendPort\s") {
    Write-Warning "FRONTEND_HOST_PORT $frontendPort is currently in use."
}
if ($listening -match "LISTENING\s+\S+:$backendPort\s") {
    Write-Warning "BACKEND_HOST_PORT $backendPort is currently in use."
}

Write-Host "Preflight checks passed."
