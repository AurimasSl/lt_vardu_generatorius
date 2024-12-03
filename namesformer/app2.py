# -*- coding: utf-8 -*-
import streamlit as st
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

file_path = 'myfile.txt'


# Dataset and Model Definitions (Unchanged)
class NameDataset(Dataset):
    def __init__(self, csv_file):
        self.names = pd.read_csv(csv_file)['name'].values
        self.chars = sorted(list(set(''.join(self.names) + ' '))) 
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx] + ' '
        encoded_name = [self.char_to_int[char] for char in name]
        return torch.tensor(encoded_name)

class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads):
        super(MinimalTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x

def sample_with_temperature(model, dataset, start_str='a', max_length=20, k=5, temperature=1.0):
    assert temperature > 0, "Temperatūra turi būti didesnė už 0"
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor([dataset.char_to_int[c] for c in start_str]).unsqueeze(0)
        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq)
            logits = output[0, -1] / temperature
            top_k_probs, top_k_indices = torch.topk(torch.softmax(logits, dim=0), k)
            next_char_idx = top_k_indices[torch.multinomial(top_k_probs, 1).item()].item()
            next_char = dataset.int_to_char[next_char_idx]
            if next_char == ' ':
                break
            output_name += next_char
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)
        return output_name

# 1. Load datasets

male_dataset = NameDataset('/namesformer/vyru_vardai_no_accents.txt')
female_dataset = NameDataset('/namesformer/moteru_vardai_no_accents.txt')

# 2. Load pre-trained models
male_model = MinimalTransformer(vocab_size=male_dataset.vocab_size, embed_size=128, num_heads=16)
male_model.load_state_dict(torch.load('vyru_model.pth', map_location=torch.device('cpu')))
male_model.eval()

female_model = MinimalTransformer(vocab_size=female_dataset.vocab_size, embed_size=128, num_heads=16)
female_model.load_state_dict(torch.load('moteru_model.pth', map_location=torch.device('cpu')))
female_model.eval()

# 3. Streamlit UI Enhancements
st.set_page_config(page_title="Vardų generatorius", layout="wide")
st.title("Generatorius")
st.subheader("Generuoja lietuviškus* vardus")

# 4. Gender Selection
with st.sidebar:
    st.markdown("Lytis")
    gender = st.radio("", ["Vyras", "Moteris"])
    model = male_model if gender == "Vyras" else female_model
    dataset = male_dataset if gender == "Vyras" else female_dataset

# Input Section
st.markdown("### Parametrai")
cols = st.columns([1, 2, 2])

with cols[0]:
    start_str = st.text_input(
        "Pradinės raidės:",
        value="",
        max_chars=100,
    )

with cols[1]:
    temperature = st.slider(
        "Temperatūros lygis:",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

with cols[2]:
    max_length = st.number_input(
        "Max vardo ilgis:",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Sugeneruos neilgesnį vardą nei nurodyta"
    )

if len(start_str) == 0:
    start_str = random.choice(dataset.chars)

# 5. Generate Names
if st.button("Generuoti"):
    st.markdown("### Rezultatai")
    try:
        for _ in range(5):
            name = sample_with_temperature(
                model,
                dataset,
                start_str=start_str,
                max_length=max_length,
                k=5,
                temperature=temperature,
            )
            st.success(f"{name}")
    except Exception as e:
        st.error(f"Error: {e}")
