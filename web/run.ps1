Set-Location $Env:Programfiles
if (Test-Path "musicIA") {
    Set-Location .\musicIA
    "/~/ MusicIA 1.1 /~/ \n"
    venv/Scripts/Activate.ps1
    Set-Location .\web
    py .\manage.py runserver
} else {
    Write-Host -NoNewLine "MusicIA n'est pas installé sur cet ordinateur. Appuyez sur n'importe quelle touche pour fermer cette fenêtre.";
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');
}