# Theia Mac -paketin tarkistusskripti
$ErrorActionPreference = "Stop"

# Värit
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Test-JsonFile {
    param (
        [string]$FilePath,
        [string]$Description
    )
    try {
        $content = Get-Content $FilePath -Raw
        $null = ConvertFrom-Json $content
        Write-ColorOutput Green "[OK] $Description on validi JSON"
        return $true
    }
    catch {
        Write-ColorOutput Red "[VIRHE] $Description tiedostossa: $_"
        return $false
    }
}

Write-ColorOutput Cyan "`nTheia Mac -paketin tarkistus"
Write-ColorOutput Cyan "========================="

# 1. Tarkista vaaditut tiedostot
$requiredFiles = @(
    @{Path=".env.example"; Desc="Ympäristömuuttujat"},
    @{Path="install_mac.sh"; Desc="Mac-asennusskripti"},
    @{Path="model_config.json"; Desc="Mallin konfiguraatio"},
    @{Path="pyproject.toml"; Desc="Python-projektin asetukset"},
    @{Path="settings.json"; Desc="Theia-asetukset"},
    @{Path="test_metal.py"; Desc="Metal-testaaja"},
    @{Path=".pre-commit-config.yaml"; Desc="Pre-commit konfiguraatio"},
    @{Path="package.json"; Desc="Node.js projektin asetukset"},
    @{Path="tsconfig.json"; Desc="TypeScript konfiguraatio"},
    @{Path="src/backend/main.ts"; Desc="Theia backend"}
)

$allFilesExist = $true
Write-ColorOutput Yellow "`n1. Tarkistetaan tiedostot:"
foreach ($file in $requiredFiles) {
    if (Test-Path $file.Path) {
        Write-ColorOutput Green "[OK] $($file.Desc) löytyi"
    }
    else {
        Write-ColorOutput Red "[VIRHE] $($file.Desc) puuttuu!"
        $allFilesExist = $false
    }
}

# 2. Tarkista JSON-tiedostojen validius
Write-ColorOutput Yellow "`n2. Tarkistetaan JSON-tiedostot:"
$jsonValid = $true
$jsonValid = $jsonValid -and (Test-JsonFile "settings.json" "Theia-asetukset")
$jsonValid = $jsonValid -and (Test-JsonFile "model_config.json" "Mallin konfiguraatio")
$jsonValid = $jsonValid -and (Test-JsonFile "package.json" "Node.js projektin asetukset")
$jsonValid = $jsonValid -and (Test-JsonFile "tsconfig.json" "TypeScript konfiguraatio")

# 3. Tarkista rivinvaihdot
Write-ColorOutput Yellow "`n3. Tarkistetaan rivinvaihdot:"
$files = Get-ChildItem -Path "." -File -Recurse | Where-Object { $_.Extension -in ".sh",".py",".json",".yaml",".toml",".ts" }
foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    if ($content -match "`r`n") {
        Write-ColorOutput Red "[VIRHE] $($file.Name) käyttää Windows-rivinvaihtoja (CRLF)"
        # Korjaa rivinvaihdot
        $content = $content -replace "`r`n","`n"
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-ColorOutput Green "[OK] Korjattu $($file.Name) käyttämään Unix-rivinvaihtoja (LF)"
    }
    else {
        Write-ColorOutput Green "[OK] $($file.Name) käyttää Unix-rivinvaihtoja (LF)"
    }
}

# 4. Luo uusi paketti
Write-ColorOutput Yellow "`n4. Luodaan paketti:"
if ($allFilesExist -and $jsonValid) {
    # Luo temp-kansio
    $tempDir = "theia_mac_setup"
    if (Test-Path $tempDir) {
        Remove-Item $tempDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    New-Item -ItemType Directory -Force -Path "$tempDir/src/backend" | Out-Null

    # Kopioi tiedostot
    foreach ($file in $requiredFiles) {
        $targetPath = Join-Path $tempDir $file.Path
        $targetDir = Split-Path $targetPath -Parent
        if (!(Test-Path $targetDir)) {
            New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        }
        Copy-Item $file.Path $targetPath -Force
    }

    # Pakkaa tiedostot
    Compress-Archive -Path "$tempDir\*" -DestinationPath "theia_mac_setup.zip" -Force
    Remove-Item $tempDir -Recurse -Force

    Write-ColorOutput Green "`n[OK] Paketti luotu onnistuneesti: theia_mac_setup.zip"
    
    # Näytä paketin koko
    $fileSize = (Get-Item "theia_mac_setup.zip").Length / 1MB
    Write-ColorOutput Cyan "Paketin koko: $([math]::Round($fileSize, 2)) MB"
}
else {
    Write-ColorOutput Red "[VIRHE] Pakettia ei voitu luoda virheiden takia!"
}

# 5. Yhteenveto ja ohjeet
Write-ColorOutput Yellow "`nYhteenveto ja seuraavat vaiheet:"
if ($allFilesExist -and $jsonValid) {
    Write-ColorOutput Green "[OK] Kaikki tiedostot löytyvät"
    Write-ColorOutput Green "[OK] JSON-tiedostot ovat valideja"
    Write-ColorOutput Green "[OK] Rivinvaihdot on tarkistettu"
    Write-ColorOutput Green "[OK] Paketti on valmis siirrettäväksi"
    
    Write-ColorOutput Cyan "`nSiirto Maciin:"
    Write-ColorOutput White "1. Kopioi theia_mac_setup.zip Maciin"
    Write-ColorOutput White "2. Macissa suorita:"
    Write-ColorOutput White "   unzip theia_mac_setup.zip"
    Write-ColorOutput White "   chmod +x install_mac.sh"
    Write-ColorOutput White "   ./install_mac.sh"
}
else {
    Write-ColorOutput Red "[VIRHE] Korjaa virheet ennen siirtoa Maciin!"
}
