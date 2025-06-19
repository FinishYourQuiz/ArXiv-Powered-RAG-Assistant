"""
1. Get PreTrained Models
"""
import torch
torch.cuda.empty_cache()

## --- Load Model and Tokenizer --- 
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# AutoModelForCausalLM is a generic model class that will be instantiated 
# as one of the model classes of the library created from from_pretrained() or from_config()
model = AutoModelForCausalLM.from_pretrained(
    "./models/llama-2-7b-hf", 
    torch_dtype="auto",
    device_map="cuda"
).to(device)

# AutoTokenizer is a generic tokenizer class that will be instantiated 
# as one of the tokenizer classes of the library created with from_pretrained() 
# It converts text into an array of numbers (tensors), the inputs to a text model
tokenzier = AutoTokenizer.from_pretrained(
    "./models/llama-2-7b-hf"  
)

## --- Tokenize the inputs --- 
model_inputs = tokenzier(["The secret to baking a good cake is "], return_tensors="pt").to(device)

## --- Get the inference ---  
generated_ids = model.generate(**model_inputs, max_length=100)

result = tokenzier.batch_decode(generated_ids)[0] 
print(result) 
"""
<s> The secret to baking a good cake is 3 simple things:
1. The right flour
2. The right butter
3. The right eggs
For this recipe, I use the best flour and butter available. For the eggs, I use the freshest eggs I can find.
The eggs are the most important part of the cake. The eggs are what make the cake rise.
If you don’t use fresh eggs,
"""

"""
2. Runing Pipeline
- The most convenient way to inference with a pretrained model.
- It is a simple API for many tasks
- It supports many tasks such as text generation, image segmentation,
     automatic speech reconginition, document question answering, and more
"""
from transformers import pipeline

# Text generation
pipeline_ = pipeline("text-generation", model="./models/llama-2-7b-hf", device=0)
result = pipeline_("The descret to baking a good cake is ", max_length=50)
print(result)
"""
[{'generated_text': 'The descret to baking a good cake is 1/3 butter, 1/3 oil, and 1/3 sugar.\nThe descret to baking a good cake is 1/3 butter'}]
"""
 
# Image generation
pipeline_ = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
result = pipeline_("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
print(result)

# Speech recognition
pipeline_ = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
result = pipeline_("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(result)

"""
3. Trainer
- A Complete Traning and Evaluation loop for PyTorch models.
- You only need 
    a model, dataset, a preprocessor, and a data collator 
    to build batches of data from the dataset
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# --- Load model, tokenizer, dataset ---
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
tokenzier = AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
dataset = load_dataset("rotten_tomatoes")

# --- Convert dataset into Pytorch tensors for whole dataset ---
def tokenze_dataset(dataset):
    return tokenzier(dataset["text"])
dataset = dataset.map(tokenze_dataset, batched=True)

# --- Load a data collator for creating batches of data and pass the tokenizer to it  ---
from transformers import DataCollatorWithPadding
# Data collators are objects that will form a batch 
# by using a list of dataset elements as input
data_collator = DataCollatorWithPadding(tokenizer=tokenzier)

# --- Setup Traning Arguments with features and hyperparameters ---
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir = "distilbert-rotten-tomatoes",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8, # The batch size per device accelerator core/CPU for training.
    per_device_eval_batch_size = 8, #  The batch size per device accelerator core/CPU for evaluation.
    num_train_epochs = 2 
)

# --- Pass all finished components to Traning and call train() to start ---
from transformers import Trainer

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    tokenzier = tokenzier,
    data_collator = data_collator
)

trainer.train()