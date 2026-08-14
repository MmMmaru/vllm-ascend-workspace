# Run ruff and Markdownlint directly for all tracked files.

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ascendDirectory = (Resolve-Path (Join-Path $scriptDirectory "..\vllm-ascend")).Path

function Invoke-Tool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE."
    }
}

foreach ($command in @("ruff", "markdownlint")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        Write-Error "$command is required but was not found on PATH." -ErrorAction Continue
        exit 1
    }
}

$originalDirectory = Get-Location
try {
    Set-Location $ascendDirectory

    Invoke-Tool "ruff" @("format", ".")
    Invoke-Tool "ruff" @("check", ".", "--output-format", "github", "--fix")

    $markdownFiles = @(
        git ls-files -- "*.md" |
            Where-Object {
                $_ -notmatch '^(\.agents|\.claude|\.gemini)/' -and
                $_ -notmatch '\.inc\.md$' -and
                $_ -notmatch 'report_template\.md$' -and
                $_ -notmatch 'contributors\.md$' -and
                $_ -notmatch 'PULL_REQUEST_TEMPLATE\.md$'
            }
    )

    if ($markdownFiles.Count -gt 0) {
        Invoke-Tool "markdownlint" (@("--fix") + $markdownFiles)
    }
}
catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
finally {
    Set-Location $originalDirectory
}
