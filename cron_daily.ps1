# cron_daily.ps1 — daily CrunchDAO auto-submit via `crunch push`.
# Scheduled by Task Scheduler: CrunchDAODailyPipeline, daily 23:00.

$ErrorActionPreference = "Stop"
$BaseDir = "C:\Users\Admin\earn5usd\crunchdao"
$ProjectDir = "$BaseDir\datacrunch-2-curly-crayfishwetminh-v2"
$LogFile = "$BaseDir\cron_log.txt"

# PATH for `crunch.exe`
$env:PATH = "$env:APPDATA\Python\Python314\Scripts;$env:PATH"
$env:PYTHONIOENCODING = "utf-8"

function Log-Line {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Out-File -FilePath $LogFile -Append -Encoding utf8
    Write-Host "$ts $msg"
}

Log-Line "=== cron_daily start ==="

if (-not (Test-Path $ProjectDir)) {
    Log-Line "ERROR: project dir $ProjectDir not found"
    exit 1
}

Set-Location $ProjectDir
Log-Line "cwd = $ProjectDir"

# Run crunch push — handles data update + retrain + submit
$msg = "cron auto-resub $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Log-Line "Running: crunch push --message '$msg'"
& crunch push --message $msg 2>&1 | ForEach-Object { Log-Line $_ }
$rc = $LASTEXITCODE

if ($rc -eq 0) {
    Log-Line "OK: submission pushed"
} else {
    Log-Line "ERROR: crunch push exit $rc"
}

Log-Line "=== cron_daily end ==="
exit $rc
