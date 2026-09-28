$uvRoot = "D:\software\tool\uv"

$uvBin = "$uvRoot\bin"
$uvPython = "$uvRoot\python"
$uvCache = "$uvRoot\cache"
$uvTools = "$uvRoot\tools"

# 创建目录
New-Item -ItemType Directory -Force -Path $uvBin | Out-Null
New-Item -ItemType Directory -Force -Path $uvPython | Out-Null
New-Item -ItemType Directory -Force -Path $uvCache | Out-Null
New-Item -ItemType Directory -Force -Path $uvTools | Out-Null

# 设置用户级环境变量
[Environment]::SetEnvironmentVariable(
    "UV_INSTALL_DIR",
    $uvBin,
    "User"
)

[Environment]::SetEnvironmentVariable(
    "UV_PYTHON_INSTALL_DIR",
    $uvPython,
    "User"
)

[Environment]::SetEnvironmentVariable(
    "UV_CACHE_DIR",
    $uvCache,
    "User"
)

[Environment]::SetEnvironmentVariable(
    "UV_TOOL_DIR",
    $uvTools,
    "User"
)

# 添加 uv/bin 到用户 PATH
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if (-not (($userPath -split ';') -contains $uvBin)) {
    $newPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
        $uvBin
    } else {
        "$userPath;$uvBin"
    }

    [Environment]::SetEnvironmentVariable(
        "Path",
        $newPath,
        "User"
    )
}

Write-Host ""
Write-Host "UV environment configured:"
Write-Host "  UV_INSTALL_DIR        = $uvBin"
Write-Host "  UV_PYTHON_INSTALL_DIR = $uvPython"
Write-Host "  UV_CACHE_DIR          = $uvCache"
Write-Host "  UV_TOOL_DIR           = $uvTools"
Write-Host ""
Write-Host "Please restart PowerShell / Windows Terminal."