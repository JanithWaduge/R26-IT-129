param(
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [string]$SourceDatabase = "slsl_bams",
    [string]$TargetDatabase = "slsl_bams_restore_test"
)
$ErrorActionPreference = "Stop"
if (-not $env:MONGO_URI) { throw "MONGO_URI is not configured." }
if (-not (Test-Path $ArchivePath)) { throw "Backup archive was not found." }
mongorestore --uri="$($env:MONGO_URI)" --archive="$ArchivePath" --gzip --drop `
    --nsFrom="$SourceDatabase.*" --nsTo="$TargetDatabase.*"
if ($LASTEXITCODE -ne 0) { throw "MongoDB restore failed." }
Write-Host "`nRestore completed into $TargetDatabase"
