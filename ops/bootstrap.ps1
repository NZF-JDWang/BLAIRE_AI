Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$rootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $rootDir

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

function New-SecretHex {
    param([int]$Bytes = 24)
    $data = New-Object byte[] $Bytes
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($data)
    -join ($data | ForEach-Object { $_.ToString("x2") })
}

function Ensure-EnvValue {
    param(
        [string]$Key,
        [string]$Value
    )
    $lines = Get-Content ".env"
    $prefix = "$Key="
    $index = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].StartsWith($prefix)) {
            $index = $i
            break
        }
    }

    if ($index -ge 0) {
        $current = $lines[$index].Substring($prefix.Length)
        if (-not [string]::IsNullOrWhiteSpace($current)) {
            return
        }
        $lines[$index] = "$Key=$Value"
    } else {
        $lines += "$Key=$Value"
    }

    Set-Content ".env" $lines
}

function Get-EnvValue {
    param([string]$Key)
    $line = Select-String -Path ".env" -Pattern "^$Key=" | Select-Object -First 1
    if ($null -eq $line) { return "" }
    return ($line.Line -replace "^$Key=", "")
}

function Sync-DatabaseUrlPassword {
    $postgresPassword = Get-EnvValue "POSTGRES_PASSWORD"
    if ([string]::IsNullOrWhiteSpace($postgresPassword)) {
        return
    }

    $lines = Get-Content ".env"
    $dbIndex = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].StartsWith("DATABASE_URL=")) {
            $dbIndex = $i
            break
        }
    }
    if ($dbIndex -lt 0) { return }

    $dbUrl = $lines[$dbIndex].Substring("DATABASE_URL=".Length).Trim()
    try {
        $uri = [System.Uri]$dbUrl
    } catch {
        return
    }

    if (-not $uri.Scheme.StartsWith("postgresql")) { return }
    if ([string]::IsNullOrWhiteSpace($uri.UserInfo) -or -not $uri.UserInfo.Contains(":")) { return }

    $parts = $uri.UserInfo.Split(":", 2)
    $user = [System.Uri]::UnescapeDataString($parts[0])
    $pass = [System.Uri]::UnescapeDataString($parts[1])
    if ($pass -ne "" -and $pass -ne "change_me") { return }

    $encUser = [System.Uri]::EscapeDataString($user)
    $encPass = [System.Uri]::EscapeDataString($postgresPassword)
    $authority = "$encUser`:$encPass@$($uri.Host)"
    if ($uri.Port -gt 0 -and $uri.Port -ne 5432) {
        $authority = "$authority`:$($uri.Port)"
    } elseif ($uri.Port -eq 5432 -and $dbUrl.Contains(":5432")) {
        $authority = "$authority`:5432"
    }

    $builder = New-Object System.UriBuilder($uri)
    $builder.UserName = $encUser
    $builder.Password = $encPass
    $updated = $builder.Uri.AbsoluteUri
    if ($dbUrl.StartsWith("postgresql+psycopg://")) {
        $updated = $updated -replace "^postgresql://", "postgresql+psycopg://"
    }
    $updated = $updated.TrimEnd("/")
    $lines[$dbIndex] = "DATABASE_URL=$updated"
    Set-Content ".env" $lines
    Write-Host "Updated DATABASE_URL password to match POSTGRES_PASSWORD"
}

Ensure-EnvValue "POSTGRES_PASSWORD" (New-SecretHex)
Ensure-EnvValue "ADMIN_API_KEYS" (New-SecretHex)
Ensure-EnvValue "USER_API_KEYS" (New-SecretHex)
Ensure-EnvValue "FRONTEND_PROXY_API_KEY" (New-SecretHex)
Sync-DatabaseUrlPassword

$frontendPort = Get-EnvValue "FRONTEND_HOST_PORT"
if ([string]::IsNullOrWhiteSpace($frontendPort)) { $frontendPort = "3000" }
$backendPort = Get-EnvValue "BACKEND_HOST_PORT"
if ([string]::IsNullOrWhiteSpace($backendPort)) { $backendPort = "8001" }

$listening = netstat -ano -p tcp
if ($listening -match "LISTENING\s+\S+:$frontendPort\s") {
    Write-Warning "FRONTEND_HOST_PORT $frontendPort is already in use"
}
if ($listening -match "LISTENING\s+\S+:$backendPort\s") {
    Write-Warning "BACKEND_HOST_PORT $backendPort is already in use"
}

docker compose up -d

$adminKey = (Get-EnvValue "ADMIN_API_KEYS").Split(",")[0].Trim()
if ([string]::IsNullOrWhiteSpace($adminKey)) {
    throw "ADMIN_API_KEYS is empty; cannot call /ops/init"
}

for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        docker compose exec -T backend python -c "import urllib.request; req=urllib.request.Request('http://localhost:8000/health', headers={'X-API-Key':'$adminKey'}); urllib.request.urlopen(req, timeout=5).read()"
        break
    } catch {
        if ($attempt -eq 30) { throw "Backend did not become ready in time." }
        Start-Sleep -Seconds 2
    }
}

for ($attempt = 1; $attempt -le 10; $attempt++) {
    try {
        docker compose exec -T backend python -c "import urllib.request; req=urllib.request.Request('http://localhost:8000/ops/init', method='POST', headers={'X-API-Key':'$adminKey'}); urllib.request.urlopen(req, timeout=30).read(); print('ops/init completed')"
        break
    } catch {
        if ($attempt -eq 10) { throw "Failed to call /ops/init after retries." }
        Start-Sleep -Seconds 2
    }
}

Write-Host "Bootstrap completed successfully."
