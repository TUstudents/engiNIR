# Install the engiNIR viewer on Windows: one command, no admin rights.
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/TUstudents/engiNIR/releases/latest/download/install-enginir-for-windows.ps1 | iex"
#
# What it does (all inside your user profile):
#   1. installs uv (https://docs.astral.sh/uv/) with Astral's installer if it is missing;
#   2. installs engiNIR as a uv tool on its own Python 3.13 (uv downloads it;
#      other Pythons on the machine are not touched);
#   3. puts the tool folder on your PATH (for enginir-viewer-console);
#   4. adds "engiNIR Viewer" to the Start menu (-Desktop: also to the desktop).
#
# Running it again updates engiNIR. To uninstall:
#   enginir-viewer-console --remove-shortcut
#   uv tool uninstall enginir
#
# Parameters (when run as a file, e.g. powershell -ExecutionPolicy ByPass -File <this file> -Desktop):
#   -Desktop      also put the shortcut on the desktop
#   -Version X    install engiNIR version X instead of the newest
#   -Wheel PATH   install from a local wheel (for testing)
#   -NoShortcut   skip step 4
#   -NoPathChange skip step 3's change of the user PATH (for testing)

param(
    [switch]$Desktop,
    [string]$Version = "",
    [string]$Wheel = "",
    [switch]$NoShortcut,
    [switch]$NoPathChange
)

# A script block of its own, so nothing (e.g. $ErrorActionPreference) leaks into
# the caller's session when the script runs through "irm ... | iex".
& {
    $ErrorActionPreference = "Stop"
    $python = "3.13"

    function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

    function Invoke-Checked($exe, [string[]]$arguments) {
        & $exe @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "'$exe $($arguments -join ' ')' failed with exit code $LASTEXITCODE"
        }
    }

    # 1. uv
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -eq $uv) {
        Step "Installing uv (Astral's installer)"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # The installer adds its folder to the user PATH for new windows; this one needs it now.
        foreach ($folder in @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")) {
            if ((Test-Path "$folder\uv.exe") -and ($env:Path -notlike "*$folder*")) {
                $env:Path = "$folder;$env:Path"
            }
        }
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($null -eq $uv) {
            throw "uv was installed but is not on PATH; open a new PowerShell window and run this command again"
        }
    } else {
        Step "Using uv at $($uv.Source)"
    }
    $uv = $uv.Source

    # 2. engiNIR as a uv tool
    if ($Wheel) {
        $package = (Resolve-Path $Wheel).Path
        Step "Installing engiNIR from $package"
        Invoke-Checked $uv @("tool", "install", "--python", $python, "--reinstall", $package)
    } else {
        $package = if ($Version) { "enginir==$Version" } else { "enginir" }
        Step "Installing $package (Python $python)"
        Invoke-Checked $uv @("tool", "install", "--python", $python, "--upgrade", $package)
    }

    # 3. PATH
    if (-not $NoPathChange) {
        Step "Adding the tool folder to PATH"
        Invoke-Checked $uv @("tool", "update-shell")
    }
    $bin = (& $uv tool dir --bin).Trim()
    if ($env:Path -notlike "*$bin*") { $env:Path = "$bin;$env:Path" }
    $console = Join-Path $bin "enginir-viewer-console.exe"
    if (-not (Test-Path $console)) { throw "The launcher $console is missing after the install" }
    Invoke-Checked $console @("--version")

    # 4. Shortcuts
    if (-not $NoShortcut) {
        Step "Adding engiNIR Viewer to the Start menu"
        $arguments = @("--install-shortcut")
        if ($Desktop) { $arguments += "--desktop" }
        Invoke-Checked $console $arguments
    }

    Write-Host ""
    Write-Host "engiNIR is installed." -ForegroundColor Green
    if (-not $NoShortcut) { Write-Host "  Start:     Start menu > engiNIR Viewer" }
    Write-Host "  Console:   enginir-viewer-console   (in a new terminal window)"
    Write-Host "  Update:    run this command again, or: uv tool upgrade enginir"
    Write-Host "  Uninstall: enginir-viewer-console --remove-shortcut; uv tool uninstall enginir"
    Write-Host "  Log file:  $env:USERPROFILE\.enginir\enginir.log"
}
