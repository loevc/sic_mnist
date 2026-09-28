powershell -ExecutionPolicy Bypass -File D:\software\tool\setup_uv.ps1


# restart this window

[Environment]::GetEnvironmentVariable("UV_INSTALL_DIR", "User")

[Environment]::GetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", "User")

[Environment]::GetEnvironmentVariable("UV_CACHE_DIR", "User")

[Environment]::GetEnvironmentVariable("UV_TOOL_DIR", "User")

conda deactivate

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 记得powershell 左上角如果是选择，那么看不到执行变化哦

uv --version


