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
Set-Alias iu Invoke-Utility

# taskkill /IM "powershell.exe" /F

Install-Chocolatey
choco feature enable -n=allowGlobalConfirmation
choco install ffmpeg
choco install git
choco install python --version=3.10.11

try {
    Set-Location $Env:Programfiles
    Remove-Item musicIA -Recurse -Force
    iu git clone https://github.com/drikomak/musicIA.git
    Set-Location musicIA
    py -m venv venv
    venv/Scripts/Activate.ps1
} catch {
    Write-Host -NoNewLine "Impossible de trouver le repo git.";
    exit
}


$gpu = (Get-WmiObject Win32_VideoController).Name
If ($gpu -like '*NVIDIA*') {
    pip install -r requirementsCUDA.txt
} Else {
    pip install -r requirementsCPU.txt
}

Set-Location .\web
py .\installmodel.py

Write-Host -NoNewLine "MusicIA a bien été installé. Vous pouvez le lancer depuis le programme run.exe .";
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');
exit