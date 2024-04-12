from transformers import AlbertTokenizer
from tqdm import tqdm

tokenizer = AlbertTokenizer.from_pretrained("albert-base-uncased")

sentences = []
with open("scratch/taira.e/c4_10_dataset_distill.txt", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Tokenizing"):
        sentences.append(line.strip())  # Assuming you want to strip newline characters

tokenizer.build_vocab(sentences)
tokenizer.save_vocabulary("/scratch/taira.e/albert_vocab.txt")
