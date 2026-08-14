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

$arguments = @('compose', '--env-file', $envFile)
if ($MusicProfile -ne 'none') {
    $arguments += @('--profile', $MusicProfile)
}

& docker @arguments config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose 설정 검증에 실패했습니다.' }

& docker @arguments up --build --detach
if ($LASTEXITCODE -ne 0) { throw 'EBookStudio 컨테이너 시작에 실패했습니다.' }

& docker @arguments ps
Write-Host 'API health: http://127.0.0.1:5000/health'

