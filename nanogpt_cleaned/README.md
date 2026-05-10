
This is a fork of Karpathy's excellent https://github.com/karpathy/nanogpt (MIT License), I continue the MIT license in this repo. I've simplified the code by a) removing alternate configurations and b) removing GPT2 architectural options. This fork works just for Shakespeare-character level tokens, on CPU or GPU/MPS.

## Setting up with venv

Create a `.venv` folder with uv, then install the `nanogpt` dependencies.

This project allows Python 3.11, 3.12, and 3.13 except for Python 3.13.8. Avoid Python 3.13.8 with PyTorch 2.11.0 for now; that combination can fail during `import torch` with an `IndentationError` in PyTorch's overload parsing.

```
uv sync
. .venv/bin/activate
```

## Training shakespeare

```
python shakespeare_char/prepare.py
... outputs some vocab info
python train.py # by default this reads the shakespeare_char prepared data, using CPU (i.e. stupid + quick + debuggable), circa 3 mins for 2000 iterations
```

### If you're on Mac or you have a CUDA device

If you run the GPU config it'll take 20 minutes, building a larger model that's better at producing Shakespeare but it might be harder to debug as the data might not be easily accessible to the Python debugger.

```
python train.py --device=mps # use on a Mac
python train.py --device=cuda # win/lin with CUDA
```

In `train.py` there's a flag `TRAIN_CONFIG` which by default is `cpu` which makes it easy to debug, at least with CUDA it is hard to debug as data lives on the GPU. If you flick this switch to `gpu` it'll run with a bigger model, much slower and it'll make a better model.

## Inference on the trained model

```
python sample.py
```
