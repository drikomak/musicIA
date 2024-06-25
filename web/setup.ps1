function Install-Chocolatey {
    Set-ExecutionPolicy Bypass -Scope Process -Force; 
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor  [System.Net.SecurityProtocolType]::Tls12; 
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

function Invoke-Utility {
    <#
    .SYNOPSIS
    Invokes an external utility, ensuring successful execution.
    
    .DESCRIPTION
    Invokes an external utility (program) and, if the utility indicates failure by 
    way of a nonzero exit code, throws a script-terminating error.
    
    * Pass the command the way you would execute the command directly.
    * Do NOT use & as the first argument if the executable name is not a literal.
    
    .EXAMPLE
    Invoke-Utility git push
    
    Executes `git push` and throws a script-terminating error if the exit code
    is nonzero.
    #>
      $exe, $argsForExe = $Args
      # Workaround: Prevents 2> redirections applied to calls to this function
      #             from accidentally triggering a terminating error.
      #             See bug report at https://github.com/PowerShell/PowerShell/issues/4002
      $ErrorActionPreference = 'Continue'
      try { & $exe $argsForExe } catch { Throw } # catch is triggered ONLY if $exe can't be found, never for errors reported by $exe itself
      if ($LASTEXITCODE) { Throw "$exe indicated failure (exit code $LASTEXITCODE; full command: $Args)." }
}

function Write-ConsoleExit {
    Write-Host -NoNewLine $Args -ForegroundColor Red;
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');
    exit
}

function Write-Grey {
    Write-Host $Args -ForegroundColor DarkGray
}

Set-Alias iu Invoke-Utility
Set-Alias err Write-ConsoleExit
Set-Alias log Write-Grey
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

log "/~/ MusicIA 1.1 Installer /~/ `n"
log "/~/ Installation des dépendances de MusicIA. /~/"

# Installation de Chocolatey, FFMPEG, GIT et Python (https://chocolatey.org/)
Install-Chocolatey
choco feature enable -n=allowGlobalConfirmation
choco install ffmpeg git
choco install python --version=3.10.11
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User") 

# Copie de l'application depuis Git vers le dossier Program Files (détection automatique)
Set-Location $Env:Programfiles
if (Test-Path "musicIA") {      # Si un dossier musicIA existe déjà, suppression et réinstallation
    log " /~/ Installation de MusicIA déjà trouvée, réinstallation totale en cours. /~/"
    Remove-Item musicIA -Recurse -Force
}

try {
    iu git clone https://github.com/drikomak/musicIA.git
    Set-Location musicIA
} catch {
    err "Impossible de trouver le repo git.";
}

log " /~/ Installation des Package Python /~/"

# Création d'un environnement virtuel Python et activation
try {
    iu python3.10 -m venv venv
    venv/Scripts/Activate.ps1
} catch {
    err "Impossible d'utiliser Python. Vérifiez que votre antivirus ne bloque pas le programme."
}

# Installation des Packages, selon si la machine possède un GPU NVIDIA ou non (pour profiter de CUDA)
$gpu = (Get-WmiObject Win32_VideoController).Name
If ($gpu -like '*NVIDIA*') {
    python3.10 -m pip install -r requirementsCUDA.txt
} Else {
    python3.10 -m pip install -r requirementsCPU.txt
}

Set-Location .\web

log " /~/ Installation du modèle MusicGen /~/"
python3.10 .\installmodel.py
deactivate

log "/~/ MusicIA a bien été installé. Vous pouvez le lancer depuis le programme run.exe. /~/";
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');