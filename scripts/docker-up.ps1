[CmdletBinding()]
param(
    [ValidateSet('none', 'cpu-music', 'gpu')]
    [string]$MusicProfile = 'none'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repo '.env'
$exampleFile = Join-Path $repo '.env.example'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI를 찾을 수 없습니다. Docker Desktop을 먼저 설치하십시오.'
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker 엔진이 실행 중이 아닙니다. Docker Desktop을 시작하십시오.'
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath $exampleFile -Destination $envFile
    $secretBytes = [System.Security.Cryptography.RandomNumberGenerator]::GetBytes(64)
    $secret = [Convert]::ToHexString($secretBytes).ToLowerInvariant()
    $content = Get-Content -Raw -LiteralPath $envFile
    $content = [regex]::Replace($content, '(?m)^SECRET_KEY=.*$', "SECRET_KEY=$secret")
    Set-Content -LiteralPath $envFile -Value $content -Encoding utf8NoBOM -NoNewline
    Write-Host '.env를 생성하고 SECRET_KEY를 무작위 값으로 설정했습니다.'
}

$composeFile = Join-Path $repo 'compose.yaml'
$arguments = @('compose', '--project-directory', $repo, '--env-file', $envFile, '-f', $composeFile)
if ($MusicProfile -ne 'none') {
    $arguments += @('--profile', $MusicProfile)
}

& docker @arguments config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose 설정 검증에 실패했습니다.' }

& docker @arguments up --build --detach --wait --wait-timeout 300
if ($LASTEXITCODE -ne 0) { throw 'EBookStudio 컨테이너 시작 또는 준비 확인에 실패했습니다.' }

& docker @arguments ps
$apiPort = 5000
$portMatch = [regex]::Match((Get-Content -Raw -LiteralPath $envFile), '(?m)^API_PORT=(\d+)\s*$')
if ($portMatch.Success) { $apiPort = [int]$portMatch.Groups[1].Value }
$healthUrl = "http://127.0.0.1:$apiPort/health"
$health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 10
if ($health.status -ne 'ok' -or $health.persistence -ne 'postgresql') {
    throw 'API health 응답이 예상한 PostgreSQL 구성과 다릅니다.'
}
Write-Host "EBookStudio가 준비되었습니다: $healthUrl"

