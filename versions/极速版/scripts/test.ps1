$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    & python -m unittest discover -s (Join-Path $repoRoot "tests") -v
    if ($LASTEXITCODE -ne 0) {
        throw "Unit tests failed."
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
