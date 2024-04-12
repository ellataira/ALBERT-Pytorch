from transformers import ALBERTTokenizer

tokenizer = ALBERTTokenizer.from_pretrained("albert-base-uncased")

sentences = []
with open("scratch/taira.e/c4_10_dataset_distill.txt", "r", encoding="utf-8") as f:
    for line in f:
        sentences.append(line)

tokenizer.build_vocab(sentences)
tokenizer.save_vocabulary("/scratch/taira.e/albert_vocab.txt")
