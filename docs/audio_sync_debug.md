# Audio Sync Debug Log

## The problem

When interpolating a 24fps clip to 60fps, the audio stream gets out of sync by ~0.5-1.5 seconds over a 90-second clip. The video frames are correct, but the audio drifts forward.

## What I've tried

### Approach 1: Simple ffmpeg re-mux
```bash
ffmpeg -i interp_video.mp4 -i original_audio.mp3 -c copy output.mp4
```
Result: Works for short clips (< 30s). Drifts on longer ones.

### Approach 2: Adjust audio tempo
```bash
ffmpeg -i audio.mp3 -atempo 0.4 -i interp.mp4 -c:v copy output.mp4
```
Result: Closer but still ~0.3s drift at 90s. The tempo factor isn't exact.

### Approach 3: Frame-accurate trimming
Count exact frames, calculate expected audio duration, trim precisely.
```python
original_frames = 2160  # 24fps × 90s
interp_frames = 5400    # 60fps × 90s
audio_duration = 90.0   # seconds
expected_interp_duration = original_frames / 60.0  # = 36s ??? no...
```

## Current status

The issue seems to be that ffmpeg's `-c copy` doesn't handle PTS (presentation timestamps) correctly when frame count changes. Need to either:
1. Re-encode audio with correct timestamps
2. Use `-shortest` flag and manually trim

Still investigating. See `examples/short_clip_24_to_60fps.md` for working example with short clips.
