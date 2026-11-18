$ErrorActionPreference = 'Stop'
$appDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $appDirectory '.venv\Scripts\python.exe'
$serverPath = Join-Path $appDirectory 'sign_server.py'
$serviceUrl = 'http://127.0.0.1:5000/api/status'
$pageUrl = 'http://127.0.0.1:5000/'
$logDirectory = Join-Path $env:LOCALAPPDATA 'ChatbotGesture'
$logPath = Join-Path $logDirectory 'startup.log'

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Test-ModelService {
    try {
        $status = Invoke-RestMethod -Uri $serviceUrl -TimeoutSec 3
        return [bool]$status.ready
    }
    catch {
        return $false
    }
}

try {
    if (-not (Test-ModelService)) {
        if (-not (Test-Path $pythonPath)) {
            throw "Python environment not found at $pythonPath. Follow GESTURE_SETUP.md."
        }

        $runningServer = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
            Where-Object { $_.CommandLine -like '*sign_server.py*' } |
            Select-Object -First 1

        if (-not $runningServer) {
            Start-Process `
                -FilePath $pythonPath `
                -ArgumentList 'sign_server.py' `
                -WorkingDirectory $appDirectory `
                -WindowStyle Minimized
        }

        $serviceReady = $false
        for ($attempt = 0; $attempt -lt 180; $attempt++) {
            Start-Sleep -Seconds 2
            if (Test-ModelService) {
                $serviceReady = $true
                break
            }
        }

        if (-not $serviceReady) {
            throw 'The sign server did not become ready within six minutes.'
        }
    }

    Start-Process $pageUrl
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) Chatbot gesture service and browser started."
}
catch {
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) $($_.Exception.Message)"
    throw
}
