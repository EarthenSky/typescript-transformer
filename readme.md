# typescript transformer

## setup
- node 24.13 w/ yarn -> https://nodejs.org/en/download
- 

## how to download weights
- go to meta website, the dropdown has llama2. You'll have to run their script download.sh, passing in the URL you were provided.

## todo
- where are llama2's weights?
  - provided by meta
- what is the structure of llama2?
  - info in the model card https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md
  - I guess the paper has some helpful details around section 2.2, but it's generally not clear where the weights come from unless you use llama2.c?
    - https://arxiv.org/pdf/2307.09288
  - https://github.com/meta-llama/llama/blob/main/llama/model.py
- how are tokens tokenized / de-tokenized?
  - https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py
  - 