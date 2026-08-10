param([string]$BackupDirectory = "..\backups")
$ErrorActionPreference = "Stop"
if (-not $env:MONGO_URI) { throw "MONGO_URI is not configured." }
if (-not $env:MONGO_DATABASE) { throw "MONGO_DATABASE is not configured." }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path $BackupDirectory | Out-Null
$archivePath = Join-Path $BackupDirectory "$($env:MONGO_DATABASE)_$timestamp.archive.gz"
mongodump --uri="$($env:MONGO_URI)" --db="$($env:MONGO_DATABASE)" --archive="$archivePath" --gzip
if ($LASTEXITCODE -ne 0) { throw "MongoDB backup failed." }
$hash = Get-FileHash $archivePath -Algorithm SHA256
"$($hash.Hash)  $($hash.Path)" | Set-Content "$archivePath.sha256"
Write-Host "`nBackup completed:`n$archivePath`nSHA-256:`n$($hash.Hash)"
