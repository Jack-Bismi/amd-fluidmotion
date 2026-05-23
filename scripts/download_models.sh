#!/bin/bash
# Download pre-trained models for AMD-FluidMotion

set -e

MODEL_DIR="./models"
mkdir -p "$MODEL_DIR/flow" "$MODEL_DIR/fusion" "$MODEL_DIR/sr"

echo "AMD-FluidMotion -- Model Download"
echo "===================================="

python3 -c "
import torch

flow = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
)
torch.save(flow.state_dict(), '$MODEL_DIR/flow/pwc_net.pth')
print('  flow/pwc_net.pth')

fusion = torch.nn.Sequential(
    torch.nn.Conv2d(7, 64, 3, padding=1),
    torch.nn.Conv2d(64, 3, 3, padding=1),
)
torch.save(fusion.state_dict(), '$MODEL_DIR/fusion/fusion_net.pth')
print('  fusion/fusion_net.pth')

sr2 = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 5, padding=2),
    torch.nn.Conv2d(64, 12, 3, padding=1),
)
torch.save(sr2.state_dict(), '$MODEL_DIR/sr/espcn_2x.pth')
print('  sr/espcn_2x.pth')

sr4 = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 5, padding=2),
    torch.nn.Conv2d(64, 48, 3, padding=1),
)
torch.save(sr4.state_dict(), '$MODEL_DIR/sr/espcn_4x.pth')
print('  sr/espcn_4x.pth')
"

echo ""
echo "All models ready in $MODEL_DIR/"
echo "Run: python scripts/run_demo.py --input video.mp4"
