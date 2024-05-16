from transformers import AutoModel, AutoTokenizer
import torch


# Prepare text
text = "Go straight and take a left at the intersection. "

def encode_text(text, target_dim=512, model_name='bert-base-uncased'):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Prepare text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=target_dim, return_attention_mask=True)

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1)  # Taking the mean of the output tensor across the sequence length

    return embeddings

embeddings = encode_text(text)
linear = torch.nn.Linear(embeddings.size(1), target_dim=512)
print("Embedding shape:", embeddings.shape)
