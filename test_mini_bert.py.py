import torch
from transformers import BertTokenizerFast
from BERT_from_scratch import MiniBertForPreTraining, MiniBertConfig

# --- Load trained model ---
checkpoint = torch.load("mini_bert.pt", map_location="cpu")
config = MiniBertConfig(**checkpoint["config"])

model = MiniBertForPreTraining(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(" Model loaded successfully on", device)

# --- Load tokenizer ---
tokenizer = BertTokenizerFast.from_pretrained(checkpoint["tokenizer"])

# --- 1 Test Masked Language Modeling ---
text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    prediction_scores, _ = model(
        inputs["input_ids"],
        inputs["token_type_ids"],
        inputs["attention_mask"]
    )

# Get index of [MASK]
mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = prediction_scores[0, mask_token_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"\n MLM Prediction:")
print(f"Input:  {text}")
print(f"Output: The capital of France is {predicted_token}.")

# --- 2️ Test Next Sentence Prediction ---
sentence_a = "The Eiffel Tower is in Paris."
sentence_b = "It was built in 1889."

encoding = tokenizer(sentence_a, sentence_b, return_tensors="pt").to(device)

with torch.no_grad():
    _, seq_relationship_logits = model(
        encoding["input_ids"],
        encoding["token_type_ids"],
        encoding["attention_mask"]
    )

pred = torch.argmax(seq_relationship_logits, dim=1).item()
label = "Is Next" if pred == 1 else "Not Next"

print(f"\n NSP Prediction:")
print(f"Sentence A: {sentence_a}")
print(f"Sentence B: {sentence_b}")
print(f"Prediction: {label}")
