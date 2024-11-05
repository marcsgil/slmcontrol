#!/bin/bash

python_versions=("3.10" "3.11")

# Loop through the combinations and run the command
for python_version in "${python_versions[@]}"; do
    echo uv run --python "$python_version" --with "numpy==1.25" pytest
    uv run --python "$python_version" --with "numpy==1.25" pytest
done

numpy_version = "1.26"
python_versions=("3.10" "3.11" "3.12")

# Loop through the combinations and run the command
for python_version in "${python_versions[@]}"; do
    echo uv run --python "$python_version" --with "numpy==1.26" pytest
    uv run --python "$python_version" --with "numpy==1.26" pytest
done

numpy_version = "2.0"
python_versions=("3.10" "3.11" "3.12")

# Loop through the combinations and run the command
for python_version in "${python_versions[@]}"; do
    echo uv run --python "$python_version" --with "numpy==2.0" pytest
    uv run --python "$python_version" --with "numpy==2.0" pytest
done

numpy_version = "2.1"
python_versions=("3.10" "3.11" "3.12" "3.13")

# Loop through the combinations and run the command
for python_version in "${python_versions[@]}"; do
    echo uv run --python "$python_version" --with "numpy==2.1" pytest
    uv run --python "$python_version" --with "numpy==2.1" pytest
done