import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseResNetChangeDetector(nn.Module):
    """
    Siamese ResNet-18 Deep Feature Change Detector for Bi-temporal Satellite Imagery.
    Extracts deep 512-channel convolutional feature maps from layer4 for both Before and After rasters.
    Computes spatial L2 feature distance and bilinearly interpolates to input resolution.
    """
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, img_before, img_after):
        feat_before = self.extract_features(img_before) # [B, 512, 8, 8]
        feat_after = self.extract_features(img_after)   # [B, 512, 8, 8]
        diff = torch.norm(feat_after - feat_before, p=2, dim=1, keepdim=True) # [B, 1, 8, 8]
        change_map = F.interpolate(diff, size=(256, 256), mode='bilinear', align_corners=False) # [B, 1, 256, 256]
        return change_map

def main():
    os.makedirs("models/onnx", exist_ok=True)
    out_path = "models/onnx/resnet18_change_detector.onnx"
    
    print("Loading pretrained ResNet-18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = SiameseResNetChangeDetector(resnet)
    model.eval()

    dummy_before = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    dummy_after = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print(f"Exporting to ONNX: {out_path} ...")
    torch.onnx.export(
        model,
        (dummy_before, dummy_after),
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["before", "after"],
        output_names=["change_map"],
        dynamic_axes={
            "before": {0: "batch_size"},
            "after": {0: "batch_size"},
            "change_map": {0: "batch_size"}
        }
    )
    print("ONNX export complete. File size:", os.path.getsize(out_path), "bytes")

if __name__ == "__main__":
    main()

