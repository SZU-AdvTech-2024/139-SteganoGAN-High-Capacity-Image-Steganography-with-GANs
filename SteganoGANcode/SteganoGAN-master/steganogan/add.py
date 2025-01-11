import torch
import torch.nn as nn


class MultiScaleEncoder(nn.Module):
    def __init__(self, input_channels=3, message_channels=1):
        super(MultiScaleEncoder, self).__init__()
        # Multi-scale convolution layers
        self.conv_3x3 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)

        # Feature fusion and embedding
        self.feature_fusion = nn.Conv2d(128, 64, kernel_size=1)
        self.embed_message = nn.Conv2d(64 + message_channels, 3, kernel_size=3, padding=1)

    def forward(self, x, message):
        # Multi-scale feature extraction
        feature_3x3 = self.conv_3x3(x)
        feature_5x5 = self.conv_5x5(x)
        combined_features = torch.cat([feature_3x3, feature_5x5], dim=1)
        fused_features = self.feature_fusion(combined_features)

        # Embed message
        message_expanded = message.unsqueeze(1).repeat(1, fused_features.size(1), 1, 1)
        combined_input = torch.cat([fused_features, message_expanded], dim=1)
        stego_image = self.embed_message(combined_input)

        return stego_image
