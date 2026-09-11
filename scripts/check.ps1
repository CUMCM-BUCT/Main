$ErrorActionPreference = "Stop"

$conflictMarkers = & git grep -n -E '^(<<<<<<< |>>>>>>> )' -- . 2>$null
if ($LASTEXITCODE -eq 0) {
    $conflictMarkers | Write-Host
    throw "Unresolved Git conflict markers found."
}
if ($LASTEXITCODE -ne 1) {
    throw "Unable to check Git conflict markers."
}

$forbiddenPatterns = @(
    '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.venv|venv|outputs|tmp|temp)/',
    '(^|/)\.env($|\.)',
    '(^|/)~\$'
)
$trackedFiles = & git ls-files
$forbiddenFiles = $trackedFiles | Where-Object {
    $file = $_
    $forbiddenPatterns | Where-Object { $file -match $_ }
}
if ($forbiddenFiles) {
    $forbiddenFiles | Write-Host
    throw "Tracked cache, temporary, or local secret files found."
}

$pythonFiles = & git ls-files '*.py'
if ($pythonFiles) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python files exist, but Python is not available."
    }
    foreach ($pythonFile in $pythonFiles) {
        & python -m py_compile $pythonFile
        if ($LASTEXITCODE -ne 0) {
            throw "Python syntax check failed: $pythonFile"
        }
    }
}

Write-Host "Main repository checks passed."
