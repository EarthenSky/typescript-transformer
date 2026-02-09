# typescript transformer

## setup
- node 24.13 w/ yarn -> https://nodejs.org/en/download
- download `tokenizer.bin` into `./data`
- download your model into `./data/models`. Feel free to try `wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin`
- `yarn install`
- `yarn build`
- `yarn start`

## how to download weights
- go to meta website, the dropdown has llama2. You'll have to run their script download.sh, passing in the URL you were provided.
- Oh, there are some weaker weights at https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

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
- How to get llama2 7B to work with our arch?
