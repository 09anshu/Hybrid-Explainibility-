import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from model import get_resnet50_model

def evaluate_pneumonia():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet50_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load("best_resnet50.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv("train.csv")
    
    # Get 5 positive cases of Pneumonia and 5 of Edema
    pneumonia_positives = df[df['Pneumonia'] == 1.0].head(5)['Path'].tolist()
    edema_positives = df[df['Edema'] == 1.0].head(5)['Path'].tolist()
    
    test_cases = pneumonia_positives + edema_positives
    
    print("Evaluating Model on Fully Trained Weights:")
    
    for path in test_cases:
        # Strip CheXpert-v1.0-small/
        actual_path = path.replace("CheXpert-v1.0-small/", "")
        if not os.path.exists(actual_path):
            continue
            
        image = Image.open(actual_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_logits = model(input_tensor)
            probs = torch.sigmoid(output_logits)[0].cpu().numpy()
        
        print(f"Path: {actual_path}")
        print(f"  Pneumonia Prob: {probs[0]:.2%}, Edema Prob: {probs[1]:.2%}")

if __name__ == "__main__":
    evaluate_pneumonia()
