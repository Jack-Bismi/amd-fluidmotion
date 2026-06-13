# AMD FluidMotion

I wanted to watch anime at 120fps on my RX 7800 XT. There was nothing that did frame interpolation natively on AMD. Everything was CUDA.

So I started building this.

## what it does

Takes a video, generates intermediate frames using optical flow + neural networks, outputs a higher framerate video. Also does super-resolution if you want.

It runs on ROCm. No CUDA. No translation layers. Just native HIP kernels.

```
python -m fluidmotion interpolate --input in.mp4 --output out.mp4 --target-fps 60
```

## where i'm at

This is pre-alpha. It works on my machine (RX 7800 XT, ROCm 5.7). Things will break on other cards. The super-resolution model is barely trained.

What works:
- Frame interpolation from 24/30fps to 60fps
- Basic super-resolution (480p -> 1080p)
- Batch processing for video folders

What doesn't work yet:
- Real-time playback (too slow)
- Audio sync is janky sometimes
- RDNA2 cards have issues with WMMA ops

## why not just use NVIDIA?

I don't have an NVIDIA card. Also, AMD's RDNA3 has some interesting matrix core capabilities that nobody's really explored for video processing yet. The Infinity Cache bandwidth is actually great for optical flow computation.

## setup

```bash
# need ROCm 5.7+ and a supported GPU
pip install -r requirements.txt
python setup.py develop
```

Check `configs/` for model configs. Default is optimized for 8GB VRAM.

## contributing

Sure, open an issue or PR. Especially if you have a different AMD GPU than mine — I can only test on the 7800 XT.

## license

MIT. Do whatever you want with it.


## Hardware Tested
- AMD RX 7800 XT (RDNA3)
- AMD RX 7900 XTX (RDNA3)