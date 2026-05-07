

## Setting up with venv

Create a `.venv` folder with your system Python, *activate* it, then install the `nanogpt` dependencies.

```
python -m venv .venv
. .venv/bin/activate
pip install torch numpy transformers wandb tqdm
```

## Training shakespeare

```
python shakespeare_char/prepare.py
# outputs some vocab info
python train.py # by default this reads the shakespeare_char prepared data, using CPU (i.e. stupid + quick + debuggable)
```

The above has settings hard-coded that are equivalent to Karpathy's 
```
python train.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 
```


```
### If you're on Mac or you have a CUDA device

If you run the GPU config it'll take 5-20 minutes, building a larger model that's better at producing Shakespeare but it might be harder to debug as the data might not be easily accessible to the Python debugger.

```
python train.py --device=mps # use on a Mac
python train.py --device=cuda # win/lin with CUDA
```


In `train.py` there's a flag `TRAIN_CONFIG` which by default is `cpu` which makes it easy to debug, at least with CUDA it is hard to debug as data lives on the GPU. If you flick this switch to `gpu` it'll run with a bigger model, much slower and it'll make a better model.

## Inference on the trained model

```
python sample.py --out_dir=shakespeare_char
```
```
```
```
```
```
```




```
```
