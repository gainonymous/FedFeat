from models.fedfeat import MLP, CNN, SplitMLP, SplitCNN, ResNet18, SplitResNet18, ViTB, SplitViTB

def get_model(model_name, fedfeat, args):
    name = model_name.lower()
    if name == "mlp":
        if fedfeat:
            return SplitMLP(input_dim=args["data_shape"], num_classes=args["num_classes"])
        else:
            return MLP(input_dim=args["data_shape"], num_classes=args["num_classes"])
    elif name == "cnn":
        if fedfeat:
            return SplitCNN(input_dim=args["data_shape"], num_classes=args["num_classes"])
        else:
            return CNN(input_dim=args["data_shape"], num_classes=args["num_classes"])
    elif name == "resnet":
        return SplitResNet18(input_dim=args["data_shape"], num_classes=args["num_classes"]) if fedfeat else ResNet18(input_dim=args["data_shape"], num_classes=args["num_classes"])
    
    elif name == "vit":
        return SplitViTB(input_dim=args["data_shape"], num_classes=args["num_classes"]) if fedfeat else ViTB(input_dim=args["data_shape"], num_classes=args["num_classes"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

