from datasets import load_dataset
from transformers import Trainer, TrainingArguments, BertForMaskedLM, BertTokenizerFast, BertConfig, DataCollatorForLanguageModeling
from tokenizers import BertWordPieceTokenizer
import os
import json

files = ["train1.txt"]
dataset = load_dataset("text", data_files=files, split="train")

def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)

d = dataset.train_test_split(test_size=0.1)
dataset_to_text(d["train"], "train.txt")
dataset_to_text(d["test"], "test.txt")

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
files = ["train.txt"]
vocab_size = 75
max_length = 200
truncate_longer_samples = False

# tokenizer = BertWordPieceTokenizer()
# tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
# tokenizer.enable_truncation(max_length=max_length)
# create a BERT tokenizer with trained vocab
vocab = 'vocab.txt'
tokenizer = BertWordPieceTokenizer(vocab)

model_path = "pretrained-bert"

if not os.path.isdir(model_path):
    os.mkdir(model_path)
# save the tokenizer
tokenizer.save_model(model_path)

with open(os.path.join(model_path, "config.json"), "w") as f:
    tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
    json.dump(tokenizer_cfg, f)

tokenizer = BertTokenizerFast.from_pretrained(model_path)
def encode_with_truncation(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                   max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
train_dataset = d["train"].map(encode, batched=True)
test_dataset = d["test"].map(encode, batched=True)

if truncate_longer_samples:
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

#---------------------------------------------------------------
from itertools import chain
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                  desc=f"Grouping texts in chunks of {max_length}")
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

print(len(train_dataset), len(test_dataset))

#------------------------------------------------------
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)
training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=10,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)
# initialize the trainer and pass everything to it
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# train the model
trainer.train()
model.eval()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)