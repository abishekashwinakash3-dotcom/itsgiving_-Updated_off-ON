<#
setup.ps1 — one command to get the meme cam ready. Windows.

    powershell -ExecutionPolicy Bypass -File setup.ps1

Builds a virtualenv, installs the pinned dependencies, then runs doctor.py and
tells you exactly what (if anything) is still wrong.
#>

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Step($m) { Write-Host "`n==> $m" -ForegroundColor White }
function Note($m) { Write-Host "    $m" }
function Warn($m) { Write-Host "    $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "`nSetup stopped: $m" -ForegroundColor Red; exit 1 }

Write-Host "it's giving - setup" -ForegroundColor White

# ------------------------------------------------------------------ 1. python
# mediapipe 0.10.21 has no wheel for 3.13+, so prefer 3.12 and walk down.
Step "Looking for a usable Python"
$py = $null
$probe = 'import sys; sys.exit(0 if (3,9) <= sys.version_info < (3,13) else 1)'

foreach ($v in @("3.12", "3.11", "3.10", "3.9")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py "-$v" -c $probe 2>$null
        if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
    }
}
if (-not $py) {
    foreach ($c in @("python3", "python")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) {
            & $c -c $probe 2>$null
            if ($LASTEXITCODE -eq 0) { $py = @($c); break }
        }
    }
}
if (-not $py) {
    Warn "No Python between 3.9 and 3.12 found (mediapipe 0.10.21 has no wheel for 3.13+)."
    Note "Install Python 3.12 from https://www.python.org/downloads/ - tick"
    Note "'Add python.exe to PATH' in the installer - then run this again."
    Die "no usable Python"
}
# Split into exe + args. Don't slice with $py[1..($py.Length-1)]: for a
# one-element array that is $py[1..0], which yields $py[0] again and runs
# "python python -V".
$pyExe  = $py[0]
$pyArgs = @($py | Select-Object -Skip 1)
$pyVer = & $pyExe @pyArgs -V 2>&1
Note "using $($py -join ' ')  ($pyVer)"

# ------------------------------------------------------------------- 2. venv
Step "Creating the virtualenv"
if (Test-Path "venv") {
    Note "venv\ already exists, reusing it"
} else {
    & $py[0] $py[1..($py.Length - 1)] -m venv venv
    if ($LASTEXITCODE -ne 0) { Die "could not create venv" }
    Note "created venv\"
}

$vpy = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Die "venv\Scripts\python.exe is missing - delete venv\ and retry" }

# --------------------------------------------------------------- 3. installs
Step "Installing dependencies (a minute or two the first time)"
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Die "pip install failed - scroll up for the reason. Do not unpin requirements.txt
    to get past it; those pins are what keep mediapipe from crashing at runtime."
}
Note "done"

# ------------------------------------------------------ 4. virtual-cam backend
Step "Checking the virtual-camera backend"
$obs = @(
    "$env:ProgramFiles\obs-studio",
    "${env:ProgramFiles(x86)}\obs-studio"
) | Where-Object { Test-Path $_ }
if ($obs) {
    Note "OBS Studio is installed"
} else {
    Warn "OBS Studio not found. The virtual camera needs it on Windows:"
    Warn "  install from https://obsproject.com, open OBS once, then quit it."
}

# ------------------------------------------------------------------ 5. doctor
Step "Running the full check"
& $vpy doctor.py
$doctor = $LASTEXITCODE

# ------------------------------------------------------------- 6. calibration
if ($doctor -eq 0) {
    $needs = & $vpy -c @"
import hashlib, os
p = 'calibration.json'
S = '9daa39e0d253d36c02faa44ba3bee8c130291aad239ed0f965ff967cdf9fa166'
print('yes' if (not os.path.exists(p) or hashlib.sha256(open(p,'rb').read().replace(b'\r\n', b'\n')).hexdigest() == S) else 'no')
"@
    if ($needs -eq "yes") {
        Step "Calibration"
        Note "The calibration.json in this repo is somebody else's face. Yours takes"
        Note "seven seconds: sit still with a neutral expression when the window opens."
        $ans = Read-Host "    Calibrate now? [Y/n]"
        if ($ans -match '^[Nn]') {
            Warn "skipped - run 'venv\Scripts\python its_giving_v2.py --calibrate' before you rely on it"
        } else {
            & $vpy its_giving_v2.py --calibrate
        }
    }
}

# -------------------------------------------------------------------- 7. done
Write-Host ""
if ($doctor -ne 0) {
    Write-Host "Setup finished, but the check above found blocking problems." -ForegroundColor White
    Write-Host "Fix those, then re-run:  venv\Scripts\python doctor.py"
    exit 1
}

Write-Host "Ready." -ForegroundColor Green
Write-Host @"

Every new terminal, first:
    venv\Scripts\activate

Then:
    python its_giving_v2.py --hotkeys      # starts OFF - your plain webcam

In Meet: settings > Video > Camera > the virtual camera named above,
and turn Meet's own effects off (... > Apply visual effects > None).

Type into the terminal running the script:
    manual        arm it, memes only when you ask
    heart         show that meme for a couple of seconds
    auto 60       auto for 60 seconds, then it disarms ITSELF
    <blank Enter> panic - straight back to a plain webcam

Leave it running in 'off' for serious calls. Quitting removes the camera
device and Meet goes black, which is worse than a meme.
"@
