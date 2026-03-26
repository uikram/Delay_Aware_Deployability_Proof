from torchvision import transforms
from utils.helpers import pad_to_square

class TransformFactory:
    @staticmethod
    def get_transform(model):
        if hasattr(model, 'preprocess'): # Frozen
            print("Using Frozen Model Preprocessor")
            return model.preprocess
        if hasattr(model, 'processor'): # CLIP/LoRA
            print("Using Hugging Face Processor")
            return lambda x: model.processor(images=x if x.mode=="RGB" else x.convert("RGB"), return_tensors="pt")['pixel_values'].squeeze(0)
        
        print("---- Using default ImageNet transform (Warning, Check HG/Frozen Preprocessor) ----")
        return transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ])