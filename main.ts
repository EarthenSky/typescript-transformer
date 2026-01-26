
class Tokenizer {
    constructor() {
        
    }
    encode(text: string): number[] {
        // TODO: return tokens using BPE
        return [];
    }
    decode(token: number): string {
        return null;
    }
}

interface Config {
    dim: number;
    hidden_dim: number;
    n_layers: number;
    n_heads: number;
    m_kv_heads: number;
    vocab_size: number;
    seq_len: number;
}
interface Weights {
    // stores the embedding for each token in the model's language
    token_embedding_table: Float32Array;
    // TODO: what're these for?
    rms_att_weight: Float32Array;
    rms_ffn_weight: Float32Array;

    // multi-head attention
    query: Float32Array;
    key: Float32Array;
    value: Float32Array;
    output: Float32Array;

    // TODO: update these names with what feed forward network they actually are
    w1: Float32Array;
    w2: Float32Array;
    w3: Float32Array;

    // final rmsnorm
    rms_final_weight: Float32Array;

    // ???
    wcls: Float32Array;
}
interface RuntimeBuffers {
    // current wave of activations
    x: Float32Array;

    // TODO: ?
    xb: Float32Array;
    xb2: Float32Array;
    hb: Float32Array;
    hb2: Float32Array;
    q: Float32Array;
    k: Float32Array;
    v: Float32Array;
    att: Float32Array;
    logits: Float32Array;

    // kv cache // TODO: ?
    key_cache: Float32Array;
    value_cache: Float32Array;
};

class Transformer {
    constructor(
        readonly config: Config,
        readonly weights: Weights,
        // TODO: initialize these
        readonly buffers: RuntimeBuffers
    ) {}

    static create_buffers(readonly config: Config): RuntimeBuffers {
        // TODO: this
        return RuntimeBuffers {

        };
    }

    forward(token: number, position: number) {
        let embedding = null;
        for (let layer_i: number = 0; layer_i < this.config.n_layers; layer_i++) {
            // TODO: implement this!
            this.forward_layer();
        }
    }

    forward_layer() {

    }
}

function read_checkpoint(checkpoint_path:string): [Config, Weights] {
    // TODO: this
    // read config
    // create an instance of TransformerWeights (or we can just allocate it into a single memory block & simulate pointers, since it's all f32)
    return [null, null];
}

function generate_response(prompt: string): string {
    const tokenizer = new Tokenizer();

    // TODO: tokenize prompt

    const transformer = new Transformer(...read_checkpoint("llama-2-7b-chat/consolidated.00.pth"));
    
    // transformer.forward();

    // TODO: read through how the temperature thing works & figure out how to generate a response

    return null;
}

const readline = require("readline");
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});
rl.question("Prompt:", (prompt: string) => {
    console.log(`Response: ${generate_response(prompt)}!`);
    rl.close();
});