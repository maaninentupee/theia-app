# Tiedostojen pakkaus Mac-siirtoa varten

$files = @(
    ".env.example",
    ".pre-commit-config.yaml",
    "install_mac.sh",
    "model_config.json",
    "pyproject.toml",
    "settings.json",
    "test_metal.py"
)

# Luo temp-kansio
$tempDir = "mac_transfer"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

# Kopioi tiedostot
foreach ($file in $files) {
    Copy-Item $file $tempDir -Force
    Write-Host "Kopioitu: $file" -ForegroundColor Green
}

# Muuta shell-skriptien rivinvaihdot Unix-muotoon
$shellScripts = @("install_mac.sh")
foreach ($script in $shellScripts) {
    $content = Get-Content "$tempDir\$script" -Raw
    $content = $content -replace "`r`n", "`n"
    Set-Content "$tempDir\$script" -Value $content -NoNewline -Encoding UTF8
    Write-Host "Muunnettu rivinvaihdot: $script" -ForegroundColor Yellow
}

# Pakkaa tiedostot
Compress-Archive -Path $tempDir\* -DestinationPath "theia_mac_setup.zip" -Force
Write-Host "Luotu paketti: theia_mac_setup.zip" -ForegroundColor Cyan

# Siivoa temp-kansio
Remove-Item $tempDir -Recurse -Force
Write-Host "Siivottu väliaikaistiedostot" -ForegroundColor Green

Write-Host "`nPaketti on valmis siirrettäväksi Maciin!" -ForegroundColor Green
Write-Host "Macissa suorita seuraavat komennot:" -ForegroundColor Yellow
Write-Host "1. unzip theia_mac_setup.zip" -ForegroundColor White
Write-Host "2. chmod +x install_mac.sh" -ForegroundColor White
Write-Host "3. ./install_mac.sh" -ForegroundColor White
