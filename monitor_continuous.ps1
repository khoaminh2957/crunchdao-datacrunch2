# monitor_continuous.ps1
# Continuous DOM/API monitor for CrunchDAO datacrunch-2 runs.
# Polls every 60s for 5 hours via monitor_dom.py (CDP on :9222).
# Outputs:
#   monitor_runs.csv  - one row per (poll, run): timestamp,submission,run_id,status,duration,error,source
#   alerts.log        - one line per status change or new error
#   monitor.log       - raw stdout/stderr per poll (rotated by overwrite each poll)

$ErrorActionPreference = 'Continue'

$Root      = 'C:\Users\Admin\earn5usd\crunchdao'
$PyScript  = Join-Path $Root 'monitor_dom.py'
$CsvFile   = Join-Path $Root 'monitor_runs.csv'
$AlertFile = Join-Path $Root 'alerts.log'
$LogFile   = Join-Path $Root 'monitor.log'
$StateFile = Join-Path $Root 'monitor_state.json'

# Resolve python: prefer `python`, fall back to `py -3`.
$PyExe = $null
foreach ($cand in @('python', 'py')) {
    try {
        $cmd = Get-Command $cand -ErrorAction Stop
        $PyExe = $cmd.Source
        if ($cand -eq 'py') { $PyArgs0 = @('-3') } else { $PyArgs0 = @() }
        break
    } catch {}
}
if (-not $PyExe) {
    "$(Get-Date -Format o) FATAL: no python on PATH" | Out-File -FilePath $AlertFile -Append -Encoding utf8
    exit 1
}

# Initialize CSV header if missing.
if (-not (Test-Path $CsvFile)) {
    'timestamp,submission,run_id,status,duration,error,source' | Out-File -FilePath $CsvFile -Encoding utf8
}

# Load previous state for change-detection.
$prev = @{}
if (Test-Path $StateFile) {
    try {
        $raw = Get-Content -Path $StateFile -Raw -Encoding utf8
        if ($raw) {
            $obj = $raw | ConvertFrom-Json
            foreach ($p in $obj.PSObject.Properties) { $prev[$p.Name] = $p.Value }
        }
    } catch {}
}

$startTs  = Get-Date
$endTs    = $startTs.AddHours(5)
$interval = 60
$iter     = 0

"$(Get-Date -Format o) START monitor_continuous (until $($endTs.ToString('o')))" |
    Out-File -FilePath $AlertFile -Append -Encoding utf8

function CsvEscape([string]$s) {
    if ($null -eq $s) { return '' }
    $s = $s -replace "`r", ' ' -replace "`n", ' '
    if ($s -match '[",]') { return '"' + ($s -replace '"', '""') + '"' }
    return $s
}

while ((Get-Date) -lt $endTs) {
    $iter++
    $iterStart = Get-Date
    $iso = $iterStart.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    # Invoke python; capture stdout (one JSON line) and stderr.
    $stdout = ''
    $stderr = ''
    try {
        if ($PyArgs0.Count -gt 0) {
            $stdout = & $PyExe $PyArgs0 $PyScript 2>&1 | Out-String
        } else {
            $stdout = & $PyExe $PyScript 2>&1 | Out-String
        }
    } catch {
        $stderr = $_.Exception.Message
    }

    # Persist raw log (overwrite).
    "ITER $iter @ $iso`n$stdout`nSTDERR:$stderr" | Out-File -FilePath $LogFile -Encoding utf8

    # Find the JSON line (last { ... } block).
    $jsonLine = $null
    foreach ($line in ($stdout -split "`r?`n")) {
        $t = $line.Trim()
        if ($t.StartsWith('{') -and $t.EndsWith('}')) { $jsonLine = $t }
    }

    if (-not $jsonLine) {
        $msg = "$iso ALERT no_json_output iter=$iter raw=$([string]::Join(' ', ($stdout -split "`r?`n")[0..3]))"
        $msg | Out-File -FilePath $AlertFile -Append -Encoding utf8
    } else {
        try {
            $data = $jsonLine | ConvertFrom-Json
        } catch {
            "$iso ALERT json_parse_error iter=$iter err=$($_.Exception.Message)" |
                Out-File -FilePath $AlertFile -Append -Encoding utf8
            $data = $null
        }

        if ($data) {
            $src = if ($data.PSObject.Properties.Name -contains 'source') { $data.source } else { '' }

            if (-not $data.ok) {
                $fatal = if ($data.PSObject.Properties.Name -contains 'fatal') { $data.fatal } else { 'unknown' }
                "$iso ALERT monitor_not_ok iter=$iter fatal=$fatal" |
                    Out-File -FilePath $AlertFile -Append -Encoding utf8
            }

            if ($data.runs) {
                foreach ($r in $data.runs) {
                    $sub = "$($r.submission)"
                    $rid = "$($r.run_id)"
                    $stt = "$($r.status)"
                    $dur = "$($r.duration)"
                    $err = "$($r.error)"

                    # CSV row
                    $row = @(
                        (CsvEscape $iso),
                        (CsvEscape $sub),
                        (CsvEscape $rid),
                        (CsvEscape $stt),
                        (CsvEscape $dur),
                        (CsvEscape $err),
                        (CsvEscape $src)
                    ) -join ','
                    $row | Out-File -FilePath $CsvFile -Append -Encoding utf8

                    # Change detection
                    $key = "sub${sub}_${rid}"
                    $prevStatus = ''
                    $prevErr = ''
                    if ($prev.ContainsKey($key)) {
                        $prevStatus = "$($prev[$key].status)"
                        $prevErr    = "$($prev[$key].error)"
                    }
                    if ($prevStatus -ne $stt) {
                        "$iso CHANGE sub=$sub run=$rid status: '$prevStatus' -> '$stt' dur=$dur" |
                            Out-File -FilePath $AlertFile -Append -Encoding utf8
                    }
                    if ($err -and $err -ne $prevErr) {
                        "$iso ERROR  sub=$sub run=$rid err=$err" |
                            Out-File -FilePath $AlertFile -Append -Encoding utf8
                    }
                    $prev[$key] = @{ status = $stt; error = $err; duration = $dur }
                }
            }
        }
    }

    # Persist state.
    try {
        ($prev | ConvertTo-Json -Compress -Depth 6) | Out-File -FilePath $StateFile -Encoding utf8
    } catch {}

    # Sleep to next 60-second boundary.
    $elapsed = (Get-Date) - $iterStart
    $sleepFor = $interval - [int]$elapsed.TotalSeconds
    if ($sleepFor -lt 1) { $sleepFor = 1 }
    if ((Get-Date).AddSeconds($sleepFor) -ge $endTs) { break }
    Start-Sleep -Seconds $sleepFor
}

"$(Get-Date -Format o) STOP monitor_continuous after $iter iters" |
    Out-File -FilePath $AlertFile -Append -Encoding utf8
