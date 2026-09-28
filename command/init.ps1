# open .
# open special path


uv init

# 不会真正下载
uv python pin 3.12

uv python install 3.12
# 3.12.14

uv python list --only-installed
uv python list

# reset python version
Remove-Item -Recurse -Force .venv
uv python pin 3.12.14
uv sync --python 3.12.14

# deactivate conda env
conda deactivate

# use current proj python
uv run python


uv run python src/sic_mnist/__init__.py