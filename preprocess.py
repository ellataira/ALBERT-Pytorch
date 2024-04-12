# from transformers import AlbertTokenizer
# from tqdm import tqdm
#
# tokenizer = AlbertTokenizer.from_pretrained("albert-base-uncased")
#
# sentences = []
# with open("scratch/taira.e/c4_10_dataset_distill.txt", "r", encoding="utf-8") as f:
#     for line in tqdm(f, desc="Tokenizing"):
#         sentences.append(line.strip())  # Assuming you want to strip newline characters
#
# tokenizer.build_vocab(sentences)
# tokenizer.save_vocabulary("/scratch/taira.e/albert_vocab.txt")
from transformers import AlbertTokenizer
from tqdm import tqdm
import os

# Set the directory to save the tokenizer vocabulary
vocab_dir = "/scratch/taira.e/"

# Specify the tokenizer model identifier
model_identifier = "albert-base-uncased"

# Initialize the tokenizer
try:
    tokenizer = AlbertTokenizer.from_pretrained(model_identifier)
except Exception as e:
    print(f"Error: Unable to initialize tokenizer from {model_identifier}.")
    print(f"Exception: {e}")
    exit()

# Read sentences from the input file
input_file_path = "/scratch/taira.e/c4_10_dataset_distill.txt"
if not os.path.exists(input_file_path):
    print(f"Error: Input file {input_file_path} not found.")
    exit()

sentences = []
try:
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading sentences"):
            sentences.append(line.strip())  # Assuming you want to strip newline characters
except Exception as e:
    print(f"Error: Unable to read sentences from {input_file_path}.")
    print(f"Exception: {e}")
    exit()

# Build vocabulary
try:
    tokenizer.build_vocab(sentences)
except Exception as e:
    print("Error: Failed to build vocabulary.")
    print(f"Exception: {e}")
    exit()

# Save vocabulary
output_vocab_file = os.path.join(vocab_dir, "albert_vocab.txt")
try:
    tokenizer.save_vocabulary(output_vocab_file)
    print(f"Vocabulary saved successfully to {output_vocab_file}.")
except Exception as e:
    print("Error: Failed to save vocabulary.")
    print(f"Exception: {e}")
