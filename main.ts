
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

// ---- Utils ----

// Q: what weights are provided? Presumably many?
function rmsnorm(
    out: Float32Array,
    x: Float32Array,
    weight: Float32Array
) {
    let ss = 0.0f;
    for (const xi of x) { ss += xi*xi; }
    ss /= x.length;
    ss += 1e-5f; // avoid LARGE ss after 1/x

    ss = 1.0 / sqrt(ss);
    for (let i = 0; i < out.length; i++) {
        out[i] = weight[i] * ss * x[i];
    }
}

function softmax(x: Float32Array) {
    let max_val = x[0];
    for (let i = 1; i < x.length; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    let sum = 0.0f;
    for (let i = 0; i < x.length; i++) {
        x[i] = Math.fround(Math.exp(x[i] - max_val));
        sum += x[i];
    }

    for (let i = 0; i < x.length; i++) {
        x[i] /= sum;
    }
}

function vecmatmul(
    out: Float32Array, // (m,)
    M: Float32Array,   // (n, m)
    x: Float32Array,   // (n,)
) {
    const n = x.length;
    const m = out.length;
    for (let i = 0; i < m; i++) {
        let val = 0.0f;
        for (let j = 0; j < n; j++) {
            val += x[j] * M[i * n + j];
        }
        out[i] = val;
    }
}

// ---- Inference ----

interface Config {
    dim: number;
    hidden_dim: number;
    n_layers: number;
    n_heads: number;
    n_kv_heads: number;
    vocab_size: number;
    seq_len: number;
}
interface Weights {
    // stores the embedding for each token in the model's language
    token_embedding_table: Float32Array;

    // weights for rms norm
    rms_att: Float32Array; // (layer, dim)
    rms_ffn: Float32Array;

    // multi-head attention
    query: Float32Array; // (layer, dim, n_heads * head_size)
    key: Float32Array;   // (layer, dim, n_kv_heads * head_size)
    value: Float32Array;
    output: Float32Array;

    // TODO: update these names with what feed forward network they actually are
    w1: Float32Array;
    w2: Float32Array;
    w3: Float32Array;

    // final rmsnorm
    rms_final: Float32Array;

    // ???
    wcls: Float32Array;
}
interface RuntimeBuffers {
    // current wave of activations
    x: Float32Array; // (dim,)
    x_residual: Float32Array; // (dim,)

    // TODO: this
    xb2: Float32Array;
    hb: Float32Array;
    hb2: Float32Array;

    // for attention
    q: Float32Array; // (dim,)
    // TODO: these are just pointers and need not be here
    // k: Float32Array; // (not dim,)
    // v: Float32Array; // (not dim,)

    // TODO: this
    att: Float32Array;
    logits: Float32Array;

    // kv cache
    key_cache: Float32Array; // (
    value_cache: Float32Array;
};

class Transformer {
    constructor(
        readonly config: Config,
        readonly weights: Weights,
        // TODO: initialize these
        readonly b: RuntimeBuffers
    ) {}

    static load_weights(readonly config: Config): Weights {
        let head_size = config.dim / config.n_heads;
        let weights: Weights = {

             query: new Float32Array(),
             key: new Float32Array(),
             value: new Float32Array(config.dim * config.n_kv_heads * config.head_size),
             output: new Float32Array(),
        };  
  
        // TODO: load data from the checkpoint file here

        return weights; 
    }

    static create_buffers(readonly config: Config): RuntimeBuffers {
        // TODO: this
        const head_size = config.dim / config.n_heads;
        const kv_dim = config.n_kv_heads * head_size;
        return {
            x: new Float32Array(config.dim),
            x_residual: new Float32Arrat(config.dim),
             
            q: new Float32Array(config.dim),

            key_cache: new Float32Array(
                config.n_layers * config.seq_len * kv_dim),
            val_cache: new Float32Array(
                config.n_layers * config.seq_len * kv_dim),
        };
    }

    forward(token: number, position: number) {
        const dim = config.dim;       
        // TODO: finish this + understanding the kv cache logic. Read into MQA, briefly
        const head_size = dim / config.n_heads;
        const kv_dim =  config.n_kv_heads * head_size;
        // TODO: get x (the embedding) from the
        // token embedding table
        let x = null;

        for (let layer_i = 0; layer_i < this.config.n_layers; layer_i++) {
            // TODO: implement this!
            rmsnorm(
                b.x_residual,
                x,
                weights.rms_att + layer_i * dim
            );
           
            // key and value point to the kv cache
            const layer_off = layer_i * config.seq_len * kv_dim;
            const key   = b->key_cache.subarray(layer_off + position * kv_dim, kv_dim);
            const value = b->value_cache.subarray(layer_off + position * kv_dim, kv_dim);
             
            vecmatmul(
                b.q,
                weights.query + layer_i * dim * dim
                b.x_residual);
            vecmatmul(
                key,
                weights.key   + layer_i * dim*kv_dim
                b.x_residual);
            vecmatmul(
                value,
                weights.value + layer_i * dim * kv_dim
                b.x_residual);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (let i = 0; i < dim; i += 2) {
                let head_dim = i % head_size;
                let freq = 1.0f / Math.pow(10000.0f, head_dim / (float)head_size);
                let val = pos * freq;

                // how many vectors? 2 = q & k, 1 = q only
                let rotn = (i < kv_dim ? 2 : 1);
                for (let v = 0; v < rotn; v++) {
                    // the vector to rotate (query or key)
                    let vec = (v == 0 ? query : key);
                    let v0 = vec[i];
                    let v1 = vec[i+1];
                    // repr each two elems as the mag of a rotated vector
                    vec[i]   = v0 * Math.cos(val) - v1 * Math.sin(val);
                    vec[i+1] = v0 * Math.sin(val) + v1 * Math.cos(val);
                }
            }


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