##  Instructions to Run and Train the Model

###  **To Test the Model**
1. Download the pre-trained model file.  
2. Run the following command in your terminal:
   ```bash
   python test_mini_bert.py
   ```

---

###  **To Train the Model**
Run the following command:
```bash
python BERT_from_scratch.py --device cuda --epochs 2 --batch_size 16
```

> 💡 **Tip:**  
> If you don’t have a CUDA-compatible GPU, replace `--device cuda` with `--device cpu`.


# BERT from Scratch 

Model Implementation (from scratch):

Multi-layer Transformer encoder with self-attention and feed-forward networks (you can choose a smaller number of layers/hidden size than the full BERT_base to suit time and computing resources).
Input embeddings that sum token embeddings, position embeddings, and segment embeddings (as in BERT, segment embeddings distinguish sentence A vs B for the NSP task)
Special tokens [CLS] (classification token at start) and [SEP] (separator token at end of sentence or between sentence pairs).
Output heads:
MLM head – predict masked tokens.
NSP head – binary classifier on [CLS] for next-sentence prediction.
 Use standard layers (e.g. nn.Linear) but no prebuilt BERT.
 Tokenization via existing WordPiece libraries is allowed.

   Training Procedure:Train jointly on-
Masked Language Modelling: For each input sequence, randomly mask out a small percentage of tokens (e.g. 15%) and have the model predict the masked tokens’ identities. Use the strategy described in the paper: e.g. 80% of the time replace with [MASK] , 10% with a random word, 10% leave unchanged (to bias against the model just learning the mask token.
Next sentence prediction:  Create input sequence pairs for training. Some pairs are positive examples where the second sentence truly follows the first from the original text, and others are negative examples where the second sentence is chosen randomly from the corpus (not actually the next sentence).

Both losses (MLM and NSP) should be combined (e.g. summed) to train the model jointly, as in the paper . You will need to iterate through the corpus to generate these training examples. Tip: It’s often easiest to prepare the corpus as segmented sentences. Take consecutive sentence pairs as positive examples, and for each positive pair create a negative pair by pairing the first sentence with a random second sentence from the corpus.

Dataset: WikiText-2 from HuggingFace 

Resources that might help  

BERT Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
 
