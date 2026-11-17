from datasets import load_dataset
from transformers import (                   
    AutoImageProcessor, 
    AutoModelForImageClassification,
    TrainingArguments,                
    Trainer                          
)

import evaluate
import numpy as np
import pandas as pd
import torch

#------------------readme-------------------------#
#This script is for training a ViT (Vision Transformer) model on a binary image classification dataset.
#Check the checkdataset.py script for more information on the dataset format before exchanginging
#the dataset folder with your own dataset. The dataset folder should be in the same directory as this script.
#Ensure imports are correct if exchanging model used. The model will need to be a Vision Transformer.
#During training, a mismatch warning may occur if the pretrained model has a different classification head
#than what you are using. The model will reinitialize the classification head to match the number of classes 
#in the dataset. This warning can be ignored.
#During training, a missing keys warning may occur (with a list of the missing keys), caused by reloading 
#a model checkpoint for evaluation. As long as the script saves the model successfully (script runs till 
#the end and the "model saved" message appears), this warning can be ignored. 
#A csv will be generated showing the model's predictions on the test set. 
#Ensure that your pytorch version uses GPU if running locally, otherwise training will be very slow.

#set seed
seed = 42
torch.manual_seed(seed) #CPU
torch.cuda.manual_seed_all(seed) #GPU

#load dataset
dataset = load_dataset("imagefolder", data_dir="structuraldamage")

#splits training set into training and validation if no validation set is present
if "validation" not in dataset:
    train_val_split = dataset["train"].train_test_split(test_size=0.15, seed=42)
    dataset["train"] = train_val_split["train"]
    dataset["validation"] = train_val_split["test"]

#label
labels = dataset["train"].features["label"].names
label2id = {name: i for i, name in enumerate(labels)}
id2label = {i: name for i, name in enumerate(labels)}

#define pretrained model
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=len(labels),      #this is to replace the classification head 
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True, #required when replacing the classification head
    device_map="auto")

#tokenizer
def transforms(examples):
    examples["pixel_values"] = [
        processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0] 
        for img in examples["image"]
    ]
    del examples["image"]
    return examples

tokenized_dataset = dataset.with_transform(transforms)

#eval metrics
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_prediction):
    #predictions = raw 
    #labels = correct answers
    predictions, labels = eval_prediction

    #convert raw into predicted classes
    predicted_classes = np.argmax(predictions, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predicted_classes,references=labels)
    return {"accuracy": accuracy["accuracy"],}

#training settings
training_args = TrainingArguments(
    output_dir="vitbase_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

#evaluate model on test set
results = trainer.evaluate(tokenized_dataset["test"])
print(results)

#model predictions on test set
prediction_output = trainer.predict(tokenized_dataset["test"])

#------CSV------#
logits = prediction_output.predictions
true_labels = prediction_output.label_ids

#logits to probabilities and predicted classes
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

probabilities = softmax(logits)
predicted_classes = np.argmax(logits, axis=1)

#map labels to readable names
predicted_label_names = [id2label[idx] for idx in predicted_classes]
true_label_names = [id2label[idx] for idx in true_labels] if true_labels is not None else None

#uncomment to check shape of logits
#print("Logits shape:", logits.shape)

results_df = pd.DataFrame({
    "predicted_class_id": predicted_classes,
    "predicted_label": predicted_label_names,
    f"probability_{id2label[0]}": probabilities[:, 0],
    f"probability_{id2label[1]}": probabilities[:, 1],
})

if true_labels is not None:
    results_df["true_class_id"] = true_labels
    results_df["true_label"] = true_label_names
    results_df["correct"] = results_df["predicted_class_id"] == results_df["true_class_id"]

#save csv
results_df.to_csv("test_predictions.csv", index=False)
print("test predictions saved")

#--------CSV end---------#

#save model
trainer.save_model("vitbase_model_final")
print("model saved")

#uncomment the following lines to check the unique predicted class indices and the id2label mapping
#print("Unique predicted class indices:", np.unique(predicted_classes))
#print("id2label mapping:", id2label)