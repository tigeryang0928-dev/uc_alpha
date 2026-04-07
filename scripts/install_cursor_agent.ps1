# Install Cursor standalone `agent` CLI when missing.
# Official script supports Linux/macOS only. On Windows, use WSL or the Cursor IDE `cursor` command (FAMOSE uses that path).

$ErrorActionPreference = "Continue"

function Test-AgentOnPath {
    return [bool](Get-Command agent -ErrorAction SilentlyContinue)
}

if (Test-AgentOnPath) {
    Write-Host "agent is already on PATH."
    & agent --version 2>&1
    exit 0
}

$hasWsl = [bool](Get-Command wsl.exe -ErrorAction SilentlyContinue)
if (-not $hasWsl) {
    Write-Host @"
No standalone 'agent' found and WSL is not available.

Options:
  1) Install WSL (Admin PowerShell):  wsl --install
     Reboot, open Ubuntu, then run:
       curl -fsSL https://cursor.com/install | bash
     Add to PATH in that shell:  export PATH=`$HOME/.local/bin:`$PATH

  2) Use Cursor IDE: Command Palette -> Install 'cursor' command in PATH.
     FAMOSE (llm_provider: cursor) uses Cursor.exe + cli.js; it does not require standalone agent.

  3) On macOS/Linux, run:  bash scripts/install_cursor_agent.sh
"@
    exit 1
}

$distros = @()
try {
    $raw = & wsl.exe -l -q 2>$null
    if ($raw) { $distros = @($raw | Where-Object { $_.Trim() -ne "" }) }
} catch { }

if ($distros.Count -eq 0) {
    Write-Host @"
WSL is installed but no Linux distribution is registered.

Install a distro (Admin PowerShell):  wsl --install -d Ubuntu
Then reboot, open Ubuntu, and run:
  curl -fsSL https://cursor.com/install | bash

Until then, use the Cursor IDE 'cursor' command for FAMOSE on Windows.
"@
    exit 1
}

Write-Host "Installing Cursor Agent inside WSL (first distro) via official curl | bash ..."
$bashCmd = 'set -e; if command -v agent >/dev/null 2>&1; then echo "agent already installed"; agent --version; else curl -fsSL https://cursor.com/install | bash; fi'
& wsl.exe -e bash -lc $bashCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL install failed (exit $LASTEXITCODE). Run manually inside WSL: curl -fsSL https://cursor.com/install | bash"
    exit $LASTEXITCODE
}
Write-Host "Done. Use agent from your WSL terminal (add ~/.local/bin to PATH if prompted)."
