# DTTQ - Delta Tagged Tokenized Quantization
Python framework for tokenizing weight matrices with delta tagging.

## Setup
pip install torch transformers numpy scikit-learn tqdm

## Usage
python main.py --model facebook/opt-125m --block_size 16 --vocab_size 4096
