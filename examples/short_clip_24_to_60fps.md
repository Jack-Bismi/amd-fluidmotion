# Example: Convert 24fps anime clip to 60fps

## Input

- File: `naruto_ep1_scene3.mp4` (24fps, 1080p, 12 seconds)
- 288 frames → 720 frames

## Command

```bash
python interpolate.py \
    --input naruto_ep1_scene3.mp4 \
    --output naruto_60fps.mp4 \
    --target-fps 60 \
    --model rife-v4.14 \
    --fp16
```

## Results

- Processing time: ~6.5 seconds on RX 7900 XTX
- Output: 720 frames, smooth playback
- Audio: Re-muxed with ffmpeg (no drift on 12s clip)

## Before/After

The difference is most visible during:
- Panning shots (background moves smoothly)
- Character dialogue (mouth movements more natural)
- Subtle camera drift

Less visible on:
- Static scenes (no difference)
- Fast action (still some artifacts)

## Gotchas

- Make sure input clip has CFR (constant frame rate). VFR clips produce wrong output duration.
- If you get a black screen, try `--half` flag to force FP16
- Audio sync works fine for clips under 30 seconds. Beyond that, see `docs/audio_sync_debug.md`
