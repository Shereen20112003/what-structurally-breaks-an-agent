$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $compiler = Join-Path $PSScriptRoot "tools\tectonic.exe"
    if (-not (Test-Path $compiler)) {
        $installed = Get-Command tectonic -ErrorAction SilentlyContinue
        if (-not $installed) {
            throw "Tectonic is missing. Install it or place tectonic.exe in tools."
        }
        $compiler = $installed.Source
    }
    foreach ($document in @("paper.tex", "workshop.tex", "review.tex", "supplement.tex")) {
        & $compiler --keep-logs $document
        if ($LASTEXITCODE -ne 0) {
            throw "Compilation failed: $document"
        }
    }
} finally {
    Pop-Location
}
