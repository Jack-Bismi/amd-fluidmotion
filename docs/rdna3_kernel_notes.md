# RDNA3 Kernel Notes

## Hardware tested

- RX 7900 XTX (RDNA3, 24GB VRAM)
- ROCm 6.1.2

## Issues hit

### Kernel launch failures
```
HIP error: invalid configuration argument
```
Happens when RIFE tries to launch a kernel with block size > 1024 on RDNA3. Workaround: patch the kernel config to use 512 threads.

### Memory allocation
RDNA3's memory model is different from CDNA. Some PyTorch ops allocate differently:
```python
# This sometimes OOMs on RDNA3 but works on MI250X
torch.zeros(1, 3, 1080, 1920, device='cuda')
```
Workaround: Use `torch.empty` instead of `torch.zeros` for large tensors.

### Warp size
RDNA3 uses wave32 (not wave64 like CDNA). Some HIP kernels assume wave64. Check kernel source if you get wrong results.

## Performance notes

| Resolution | RIFE v4.14 | Notes |
|-----------|-----------|-------|
| 720p | ~45 fps | Smooth |
| 1080p | ~22 fps | Usable |
| 4K | ~5 fps | Needs optimization |

## Next steps

- Try RIFE v4.15 (should have better RDNA3 support)
- Profile with ROCm's rocprof to find bottlenecks
- Test FP16 inference (currently running FP32)
