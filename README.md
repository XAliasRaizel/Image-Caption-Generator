# 📸 Image Caption Generator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg?logo=keras&logoColor=white)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-green.svg)](https://www.kaggle.com/datasets/adityajn105/flickr8k)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning multimodal system that automatically generates coherent, human-like natural language descriptions for arbitrary input images.

This project implements a **Merge-Model CNN-RNN architecture** combining **VGG16** (pre-trained on ImageNet for computer vision feature extraction) and an **LSTM network** (for sequence modeling and natural language generation), trained and evaluated on the **Flickr8k** dataset.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
- [Model Architecture & Hyperparameters](#-model-architecture--hyperparameters)
- [Evaluation & Results](#-evaluation--results)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Running via Jupyter / Google Colab](#1-running-via-jupyter--google-colab)
  - [2. Generating Captions for New Images](#2-generating-captions-for-new-images)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🔍 Overview

Image Captioning bridges the gap between **Computer Vision** and **Natural Language Processing (NLP)**. Given an image, the model:
1. Extracts high-level visual semantic features using a convolutional neural network (CNN).
2. Encodes partial caption sequences using an embedding layer and recurrent neural network (LSTM).
3. Merges the multimodal embeddings into a shared latent space.
4. Generates captions word-by-word iteratively using greedy search decoding until reaching an `<end>` sequence marker.

---

## 🏗 System Architecture

```
                    ┌─────────────────────────┐
                    │       Input Image       │
                    │      (224 x 224 x 3)    │
                    └────────────┬────────────┘
                                 │
                     [ VGG16 Backbone (fc2) ]
                                 │
                    ┌────────────▼────────────┐
                    │ Feature Vector (4096-D) │
                    └────────────┬────────────┘
                                 │
                            Dropout(0.4)
                                 │
                         Dense (256, ReLU)
                                 │
                     ┌───────────▼───────────┐
                     │ Image Embedding (256) │
                     └───────────┬───────────┘
                                 │
                                 ├───► [ Element-wise Add ] ──► Dense (256, ReLU) ──► Dense (Vocab, Softmax) ──► Next Word
                                 │
                     ┌───────────┴───────────┐
                     │  Text Embedding (256) │
                     └───────────▲───────────┘
                                 │
                             LSTM(256)
                                 │
                            Dropout(0.4)
                                 │
                       Embedding Layer (256)
                                 │
                    ┌────────────┴────────────┐
                    │ Input Token Sequence    │
                    │ (start, word_1, ... )   │
                    └─────────────────────────┘
```

---

## 📂 Repository Structure

```text
├── Final_Image_Caption_Generator_model.7z  # Compressed trained Keras model (.h5)
├── features.7z                             # Compressed pre-extracted VGG16 features (.pkl)
├── Image-Caption-Generator.ipynb           # Complete end-to-end pipeline notebook
├── max_length.pkl                          # Maximum caption sequence length (34)
├── tokenizer.pkl                           # Fitted Keras Tokenizer vocabulary (6,690 tokens)
├── .gitattributes                          # Git LFS tracking configuration
├── .gitignore                              # Ignored files and virtual environments
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

> **Note:** Large binary files (`.7z`, `.pkl`) are managed via **Git LFS** (Large File Storage).

---

## 📊 Dataset

The model is trained on the **Flickr8k** benchmark dataset:
- **8,091 images** covering diverse daily scenes, human actions, and animals.
- **5 reference captions per image** (`Flickr8k.lemma.token.txt`), totaling **40,455 captions**.
- **Split Ratio**: 90% Training (~7,281 images) and 10% Testing (810 images).

---

## ⚙️ Data Preprocessing Pipeline

### 1. Image Feature Extraction
- Images are resized to `(224, 224, 3)` and normalized using `vgg16.preprocess_input`.
- Passed through pre-trained **VGG16** with classification head removed (`fc2` layer output).
- Output is a fixed **4,096-dimensional representation** for each image, serialized into `features.pkl`.

### 2. Caption Text Cleaning
- Converted all characters to lowercase.
- Stripped special characters, numbers, and punctuation using regular expressions (`[^a-z ]`).
- Removed single-character artifacts (e.g., standalone letters).
- Added boundary tokens to each caption: `start <caption_text> end`.

### 3. Tokenization & Sequence Preparation
- Built a vocabulary using Keras `Tokenizer` with an out-of-vocabulary (`<unk>`) token.
- **Vocabulary Size**: 6,690 unique tokens.
- **Max Sequence Length**: 34 words (padded with `padding='post'`).
- Utilized a custom **data generator** (`data_generator`) to stream training batches of `(X_image, X_sequence), y_next_word`, ensuring memory-efficient training without RAM exhaustion.

---

## 🧠 Model Architecture & Hyperparameters

| Component | Layer / Parameter | Specifications |
| :--- | :--- | :--- |
| **Image Feature Extractor** | Backbone | VGG16 (`fc2` output, 4096-D) |
| | Regularization | Dropout (`0.4`) |
| | Projection Layer | Dense (`256` units, ReLU) |
| **Caption Sequence Model** | Input | Sequence length: 34 |
| | Embedding Layer | Vocab: 6,690, Dimension: 256 (`mask_zero=True`) |
| | Regularization | Dropout (`0.4`) |
| | Recurrent Layer | LSTM (`256` units) |
| **Multimodal Decoder** | Fusion | Element-wise Addition (`add([fe2, se3])`) |
| | Intermediate Layer | Dense (`256` units, ReLU) |
| | Classification Head | Dense (`6,690` units, Softmax) |
| **Training Setup** | Total Parameters | **5,071,906 trainable parameters** (~19.35 MB) |
| | Loss Function | Categorical Cross-Entropy |
| | Optimizer | Adam |
| | Batch Size / Epochs | 64 batch size / 15 epochs |

### Training Loss Progression
- **Epoch 1:** Loss = `4.7824`
- **Epoch 5:** Loss = `2.3492`
- **Epoch 10:** Loss = `1.9234`
- **Epoch 15:** Loss = `1.7144`

---

## 📈 Evaluation & Results

The model was quantitatively evaluated on the unseen **10% test split (810 images)** against all 5 ground truth captions using the **BLEU (Bilingual Evaluation Understudy)** metric:

| Metric | Score |
| :--- | :--- |
| **BLEU-1 (Unigram Precision)** | **0.3994** (~39.9%) |
| **BLEU-2 (Bigram Precision)** | **0.2256** (~22.6%) |

### Sample Predictions

#### Example 1: `1305564994_00513f9a5b.jpg`
> **Ground Truths:**
> - *two racer drive white bike down road*
> - *two motorist be ride along on their vehicle that be oddly design and color*
> - *two person be in small race car drive by green hill*
> - *two person in race uniform in street car*
>
> **Model Prediction:**
> `two person ride on race vehicle in matching position of car`

#### Example 2: `3430287726_94a1825bbf.jpg`
> **Ground Truths:**
> - *man in black outfit be ride blue snowmobile across the snow*
> - *man ride purple snowmobile*
> - *man wear snow clothes and helmet drive snow vehicle in the snow*
>
> **Model Prediction:**
> `child in red coat slide down snow tube`

---

## 🚀 Installation & Setup

### 1. Clone Repository & Initialize Git LFS
```bash
git clone https://github.com/XAliasRaizel/Image-Caption-Generator.git
cd Image-Caption-Generator

# Pull large model checkpoints and feature files via Git LFS
git lfs install
git lfs pull
```

### 2. Extract Compressed Artifacts
If you intend to use the pre-trained weights and pre-extracted features:
```bash
# Using 7-Zip or any archive manager:
7z x Final_Image_Caption_Generator_model.7z
7z x features.7z
```

### 3. Install Dependencies
Create an isolated Python virtual environment and install the required packages:
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install tensorflow numpy pillow matplotlib nltk tqdm ipykernel
```

---

## 💻 Usage Guide

### 1. Running via Jupyter / Google Colab
Launch Jupyter Notebook to view or run the full training and inference workflow:
```bash
jupyter notebook Image-Caption-Generator.ipynb
```
*(If running on Google Colab, enable GPU acceleration under `Runtime > Change runtime type > T4 GPU`.)*

### 2. Generating Captions for New Images
You can load the saved model, tokenizer, and feature extractor to generate a caption for any custom image:

```python
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load trained components
model = load_model("Final_Image_Caption_Generator_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# 2. Setup VGG16 feature extractor
vgg = VGG16()
feature_extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

def extract_image_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return feature_extractor.predict(img_array, verbose=0)

def generate_caption(image_path):
    feature = extract_image_features(image_path)
    in_text = "start"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")
        yhat = model.predict([feature, seq], verbose=0)
        word_idx = np.argmax(yhat)
        
        # Reverse lookup word by token index
        word = next((w for w, i in tokenizer.word_index.items() if i == word_idx), None)
        if word is None or word == "end":
            break
        in_text += " " + word

    return in_text.replace("start", "").strip()

# Test caption generation
print(generate_caption("path/to/your/image.jpg"))
```

---

## 🔮 Future Enhancements

- [ ] **Attention Mechanism**: Implement Bahdanau or Luong attention (Show, Attend and Tell) to allow the model to focus on relevant image regions during word generation.
- [ ] **Modern Backbones**: Benchmark newer vision feature extractors like **ResNet50**, **EfficientNet**, or **Vision Transformers (ViT)**.
- [ ] **Transformer Decoder**: Replace the LSTM recurrent decoder with a Multi-Head Self-Attention Transformer decoder or pre-trained LLM (e.g., GPT-2).
- [ ] **Beam Search**: Replace greedy search with beam search decoding to discover higher-probability global sequences and improve BLEU scores.
- [ ] **Interactive Web UI**: Build a Streamlit or Gradio interactive web app for drag-and-drop live image captioning.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
