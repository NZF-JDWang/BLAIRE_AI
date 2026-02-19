Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string[]]$Profiles = @()
)

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $rootDir

if (-not (Test-Path ".env")) {
    throw ".env is missing. Copy .env.example to .env first."
}

function Get-EnvValue {
    param([string]$Key)
    $line = Select-String -Path ".env" -Pattern "^$Key=" | Select-Object -First 1
    if ($null -eq $line) { return "" }
    return ($line.Line -replace "^$Key=", "")
}

function Invoke-Endpoint {
    param(
        [string]$Path,
        [string]$ApiKey = "",
        [string]$Method = "GET"
    )
    $cmd = @("compose")
    foreach ($profile in $Profiles) {
        if (-not [string]::IsNullOrWhiteSpace($profile)) {
            $cmd += @("--profile", $profile)
        }
    }
    $cmd += @("exec", "-T", "backend", "python", "-c")
    $payload = "import urllib.request; headers={}; key='$ApiKey'; headers.update({'X-API-Key': key} if key else {}); req=urllib.request.Request('http://localhost:8000$Path', method='$Method', headers=headers); urllib.request.urlopen(req, timeout=20).read(); print('$Path ok')"
    $cmd += $payload
    & docker @cmd
}

$adminKey = (Get-EnvValue "ADMIN_API_KEYS").Split(",")[0].Trim()
$userKey = (Get-EnvValue "USER_API_KEYS").Split(",")[0].Trim()
if ([string]::IsNullOrWhiteSpace($adminKey) -or [string]::IsNullOrWhiteSpace($userKey)) {
    throw "ADMIN_API_KEYS and USER_API_KEYS must be set for smoke tests."
}

Write-Host "Running smoke tests..."
Invoke-Endpoint -Path "/health"
Invoke-Endpoint -Path "/health/dependencies" -ApiKey $userKey
Invoke-Endpoint -Path "/runtime/options" -ApiKey $userKey
Invoke-Endpoint -Path "/models" -ApiKey $userKey
Invoke-Endpoint -Path "/ops/status" -ApiKey $adminKey
Write-Host "Smoke tests passed."
