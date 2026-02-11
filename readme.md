# typescript transformer

## setup
- install node 24.13 w/ yarn -> https://nodejs.org/en/download
- download `tokenizer.bin` into `./data`
- download your model into `./data/models`
  - A small model for stories `wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin`
  - A larger chat model ``
- `yarn install`
- `yarn build`
- `yarn start --checkpoint_path=data/models/stories42M.bin`

## notes
- This implementation is heavily derived from llama2.c and is only 20-30% slower.
- For the chat model, input your text like `[INST] <<SYS>>\nYou are a lovely human being.\n<</SYS>>\n\nWhat is the third best colour? [/INST]`
## Performance

|       | us        | llama2.c |
| ----- | --------- | -------- |
| 15M   | 50.1 tps  | 77.9 tps |
| 42M   | 19.3 tps  | 25.9 tps |
| 7000M | 

## how to download weights
- Go to meta website, the dropdown has llama2. You'll have to run their script download.sh, passing in the URL you were provided.

## misc
- llama2 resources
  - https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md
  - mostly unhelpful https://arxiv.org/pdf/2307.09288
  - very helpful! https://github.com/meta-llama/llama/blob/main/llama/model.py
  - for tokenization https://github.com/meta-llama/llama/blob/main/llama/tokenizer.py
- download llama2 weights
  - https://www.llama.com/llama-downloads/
  - follow steps
  - `python -m pip install llama-models`
  - `python -m pip install llama-stack`
  - `llama model download --source meta --model-id Llama-2-7b-chat`
    - will probably take 5 minutes or so
  - paste your link into the command
  - wait for the download to complete
  - `git clone https://github.com/karpathy/llama2.c`
  - `cd llama2.c`
  - `python -m pip install -r .\requirements.txt`
  - `python export.py llama2_7b_chat.bin --meta-llama ${your-path}/Llama-2-7b-chat`
    - will probably take 25+ minutes
