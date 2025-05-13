import torch
import numpy as np
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def load_data(data_path):
    """Load the data containing adversarial texts and labels."""
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    
    # Extract adversarial texts and true labels
    adv_texts = data['adv_texts']
    labels = data['labels']
    
    print(f"Loaded {len(adv_texts)} adversarial examples")
    return adv_texts, labels

def quantize_and_load_model(model_path):
    """Load a model with 8-bit quantization."""
    print(f"Loading model from {model_path} with 8-bit quantization")
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    # Load the model with 8-bit quantization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=quantization_config,  # Enable 8-bit quantization
        device_map="auto"   # Automatically choose device
    )
    
    # Move the model to the appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def run_inference(model, tokenizer, texts, batch_size=16):
    """Run inference on texts using the quantized model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}")
    
    all_predictions = []
    
    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        
        # Move inputs to the appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)
    
    return np.array(all_predictions)

def calculate_metrics(true_labels, predictions):
    """Calculate accuracy and other metrics."""
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return accuracy, report

def main():
    # Parameters
    data_path = "/home/hice1/jjain47/scratch/lstrech/jai/adv_attack/text-adversarial-attack/adv_samples/gpt2_imdb_finetune_0-100_iters=100_cw_kappa=5_lambda_sim=1_lambda_perp=1_emblayer=-1_bertscore_idf.pth" 
    model_path = "/home/hice1/jjain47/scratch/lstrech/jai/adv_attack/text-adversarial-attack/result/gpt2_imdb_finetune" 
    
    # Load data
    adv_texts, true_labels = load_data(data_path)
    
    # Load model with 8-bit quantization
    model, tokenizer = quantize_and_load_model(model_path)
    
    # Run inference
    predictions = run_inference(model, tokenizer, adv_texts)
    
    # Calculate metrics
    accuracy, report = calculate_metrics(true_labels, predictions)
    
    print("Inference complete!")
    return accuracy

if __name__ == "__main__":
    main()