import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import vit_b_16, ViT_B_16_Weights
import timm

# define fully connected NN
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(MLP, self).__init__()
        
        c, h, w = input_dim  # Unpack [3, 32, 32] or [1, 28, 28]
        input_feature =  c * h * w
        self.fc1 = nn.Linear(input_feature, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes);

    def forward(self, x):
        x = x.flatten(1) # [B x 784]
        x = F.relu(self.fc1(x)) # [B x 200]
        x = F.relu(self.fc2(x)) # [B x 200]
        x = self.out(x) # [B x 10]
        return x

# define Split MLP
class SplitMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        """
        input_dim: tuple like (C, H, W) or just a single integer if already flattened
        num_classes: number of output classes
        hidden_dim: size of hidden layers
        """
        super(SplitMLP, self).__init__()

        if isinstance(input_dim, (tuple, list)):
            c, h, w = input_dim
            input_feature = c * h * w
        else:
            input_feature = input_dim  # already flattened

        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_feature, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output



# Define CNN
class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()

        c, h, w = input_dim

        self.conv1 = nn.Conv2d(c, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy = F.max_pool2d(F.relu(self.bn1(self.conv1(dummy))), 2, 2)
            dummy = F.max_pool2d(F.relu(self.bn2(self.conv2(dummy))), 2, 2)
            flatten_dim = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(flatten_dim, 512)
        self.out = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2, 2)
        x = x.flatten(1)
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x


# Define Split CNN
class SplitCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SplitCNN, self).__init__()

        c, h, w = input_dim  # e.g., [3, 32, 32] or [1, 28, 28]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),   # Halves the spatial dimensions
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # Dynamically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            dummy_feat = self.feature_extractor(dummy)
            flattened_dim = dummy_feat.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, input):
        features = self.feature_extractor(input)
        output = self.classifier(features)
        return output


# ---------------- Full ResNet18 ----------------
class ResNet18(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        c, h, w = input_dim
        self.model = resnet18(pretrained=False)

        # Modify for small inputs like CIFAR-10
        self.model.conv1 = nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove initial maxpool

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ---------------- Split ResNet18 ----------------
class SplitResNet18(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        c, h, w = input_dim
        base_model = resnet18(pretrained=False)
        
        # Modify for input channels
        base_model.conv1 = nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()  # Remove maxpool for small inputs
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)

        # Assign layers directly (no sequential)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # You can skip or truncate layer3 and layer4 here if you want
        return x

    def classifier(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


# ---------------- Full ViT-B ----------------
class ViTB(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        c, h, w = input_dim
        base_model = vit_b_16(pretrained=False)

        # Replace the conv_proj if input channel != 3 (e.g., grayscale)
        if c != 3:
            base_model.conv_proj = nn.Conv2d(
                c,
                base_model.conv_proj.out_channels,
                kernel_size=base_model.conv_proj.kernel_size,
                stride=base_model.conv_proj.stride,
                padding=base_model.conv_proj.padding
            )

        # Check if heads is Sequential or Linear
        if isinstance(base_model.heads, nn.Sequential):
            in_features = base_model.heads[0].in_features
        else:
            in_features = base_model.heads.in_features

        base_model.heads = nn.Linear(in_features, num_classes)

        self.model = base_model
        self.target_size = (224, 224)  # ViT default input size

    def forward(self, x):
        # Resize input to 224x224 if not already
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return self.model(x)


# ---------------- Split ViT-B ----------------
class SplitViTB(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        c, h, w = input_dim
        base_model = vit_b_16(pretrained=False)
        self.target_size = (224, 224)

        # Adjust conv_proj for custom input channels
        if c != 3:
            base_model.conv_proj = nn.Conv2d(
                c,
                base_model.conv_proj.out_channels,
                kernel_size=base_model.conv_proj.kernel_size,
                stride=base_model.conv_proj.stride,
                padding=base_model.conv_proj.padding
            )

        # Embed-related modules
        self.conv_proj = base_model.conv_proj
        self.cls_token = base_model.class_token
        self.pos_embed = base_model.encoder.pos_embedding
        self.pos_dropout = base_model.encoder.dropout

        # Transformer blocks
        self._feature_blocks = nn.Sequential(*base_model.encoder.layers[:8])
        self.classifier_blocks = nn.Sequential(*base_model.encoder.layers[8:])
        self.encoder_norm = base_model.encoder.ln
        self.head = base_model.heads

    def embed_input(self, x):
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        B = x.size(0)
        x = self.conv_proj(x)  # (B, D, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, D)

        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)  # (B, 197, D)

        x = x + self.pos_embed
        x = self.pos_dropout(x)
        return x

    def feature_extractor(self, x):
        # Accepts raw image input, internally embeds it
        tokens = self.embed_input(x)
        return self._feature_blocks(tokens)

    def classifier(self, tokens):
        x = self.classifier_blocks(tokens)
        x = self.encoder_norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)