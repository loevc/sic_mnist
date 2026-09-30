uv add numpy matplotlib

uv run python

uv run python .\src\sic_mnist\00_dependency.py

uv sync
uv lock

uv add torch

uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"