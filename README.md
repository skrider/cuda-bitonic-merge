# CUDA bitonic merge

Experiments writing a bitonic merge for CUDA to quickly sort/argsort vectors on the order of 10^6 elements.

## run

```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd):/workspace/cbm -v /tmp:/tmp nvcr.io/nvidia/pytorch:24.01-py3
cd cbm
pip install -e .
pytest tests/test_sort.py
```
