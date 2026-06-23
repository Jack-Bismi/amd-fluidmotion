# amd-fluidmotion

I started this as a weekend experiment to make short anime clips smoother on my own AMD setup. The goal is not to beat commercial interpolation tools — it's to understand how far a small ROCm/HIP pipeline can go for frame interpolation.

## Why I built this

I had a bunch of old anime clips at 24fps that looked choppy on my 144Hz monitor. Most interpolation tools are CUDA-only, and I didn't want to dual-boot just for this. So I started poking around RIFE and OpenCV on ROCm to see what's possible.

## Current state

- Basic RIFE inference working on ROCm 6.x
- CLI tool for 24→60fps conversion
- Tested on short clips (< 2 min) at 1080p
- Audio sync is still WIP (see `docs/audio_sync_debug.md`)

## What's next

- Batch processing for folders of clips
- Better audio stream handling (currently re-muxes with ffmpeg)
- Testing on RDNA3 consumer GPUs (see `docs/rdna3_kernel_notes.md`)

## Quick start

```bash
pip install -r requirements.txt
python interpolate.py --input clip.mp4 --output clip_60fps.mp4 --target-fps 60
```

## Limitations

- Slow on first run (kernel compilation)
- 4K clips need a lot of VRAM — 1080p recommended for now
- Audio sync drifts on clips longer than 90 seconds


## Recent Updates
- Performance improvements for batch processing
- Better error messages for common issues

## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.