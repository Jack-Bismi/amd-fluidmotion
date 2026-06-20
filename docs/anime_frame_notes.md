# Anime Frame Interpolation Notes

## Why anime is different

Anime has sharp edges, flat color regions, and very different motion characteristics compared to real video. Standard optical flow models trained on real footage often produce:

- Ghosting around fast-moving characters
- Artifacts on thin lines (hair, weapon edges)
- Color bleeding in flat shading regions

## What I've tested

### RIFE (Real-Time Intermediate Flow Estimation)
- Works surprisingly well for slow pans and dialogue scenes
- Struggles with action sequences — lots of warping on sword swings
- Model: `rife-v4.14` (small enough for 8GB VRAM)

### OpenCV-based approaches
- Farneback optical flow: too blurry for anime
- DIS optical flow: better but still loses thin lines
- Lucas-Kanade: only works for sparse features

## Patterns that break interpolation

1. **Speed lines**: Background blurs with character in foreground — model gets confused about what's moving
2. **Camera shake**: Intentional shake in action scenes gets amplified
3. **Quick cuts**: Transition frames between scenes produce garbage

## What seems to work

- Pre-processing: Separate foreground/background detection
- Post-processing: Blend weight tuning for edge regions
- Short clips (< 30s) process better than long ones (less drift)
