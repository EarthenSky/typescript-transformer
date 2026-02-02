import * as fs from 'fs';

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

function rmsnorm(
    out: Float32Array,
    x: Float32Array,
    weight: Float32Array
) {
    let ss = 0.0;
    for (const xi of x)
        ss += xi * xi;
    ss /= x.length;
    ss += 1e-5f; // avoid LARGE ss after 1/x

    ss = 1.0 / Math.sqrt(ss);
    for (let i = 0; i < out.length; i++)
        out[i] = weight[i] * ss * x[i];
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

    for (let i = 0; i < x.length; i++)
        x[i] /= sum;
}

function vecmatmul(
    out: Float32Array, // (m,)
    M: Float32Array,   // (n, m)
    x: Float32Array,   // (n,)
) {
    const n = x.length;
    const m = out.length;
    for (let i = 0; i < m; i++) {
        let val = 0.0;
        for (let j = 0; j < n; j++)
            val += x[j] * M[i * n + j];
        out[i] = val;
    }
}

function view(
    x: Float32Array,
    start: number,
    size: number
): Float32Array {
    return x.subarray(start, start+size);
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
const CONFIG_NUM_ELEMENTS = 7;
const CONFIG_NUM_BYTES = CONFIG_NUM_ELEMENTS * 4;
interface Weights {
    // stores the embedding for each token
    token_embedding_table: Float32Array; // ()

    rms_att: Float32Array; // (n_layers, dim)
    rms_ffn: Float32Array;

    // multi-head attention
    query: Float32Array; // (n_layers, dim, n_heads * head_size)
    key: Float32Array;   // (n_layers, dim, n_kv_heads * head_size)
    value: Float32Array;
    output: Float32Array;

    ffn1: Float32Array;
    ffn2: Float32Array;
    ffn3: Float32Array;

    rms_final: Float32Array; // (dim,)
    classify: Float32Array; // (dim, vocab_size)
}
interface RuntimeBuffers {
    // current wave of activations
    x: Float32Array; // (dim,)
    xb: Float32Array;
    xb2: Float32Array;
    hb: Float32Array; // (hidden_dim,)
    hb2: Float32Array;

    // for attention
    q: Float32Array; // (dim,)

    // TODO: this
    att: Float32Array; // (n_heads, seq_len)
    logits: Float32Array;

    // kv cache
    key_cache: Float32Array; // (
    value_cache: Float32Array;
};

class FileLoader {
    readonly f: number;

    constructor (checkpoint_path: string) {
        this.f = fs.openSync(checkpoint_path, "rb");
        if (this.f == null)
            throw "ERROR: bad checkpoint path"
    }

    load_config(): Config {
        const data = new Uint8Array(CONFIG_NUM_BYTES);
        const bytesRead = fs.readSync(this.f, data, 0, CONFIG_NUM_BYTES, 0);
        if (bytesRead != CONFIG_NUM_BYTES)
            throw "ERROR: failed to read config"

        const view = new DataView(data.buffer);
        let config: Config = {
            dim: view.getInt32(0, true),
            hidden_dim: view.getInt32(1, true),
            n_layers: view.getInt32(2, true),
            n_heads: view.getInt32(3, true),
            n_kv_heads: view.getInt32(4, true),
            vocab_size: view.getInt32(5, true),
            seq_len: view.getInt32(6, true),
        };
        return config;
    }

    load_weights(config: Config): Weights {
        const dim = config.dim;
        const head_size = dim / config.n_heads;
        console.assert(config.n_heads * head_size == dim);

        let weights: Weights = {
            token_embedding_table: new Float32Array(
                config.vocab_size * config.dim 
            ),

            rms_att: new Float32Array(config.n_layers * dim),
            rms_ffn: new Float32Array(config.n_layers * dim),
    
            query: new Float32Array(config.n_layers * dim * dim),
            key: new Float32Array(config.n_layers * dim * config.n_kv_heads * head_size),
            value: new Float32Array(config.n_layers * dim * config.n_kv_heads * head_size),
            output: new Float32Array(config.n_layers * dim * dim),

            ffn1: new Float32Array(config.n_layers * config.hidden_dim * dim),
            ffn2: new Float32Array(config.n_layers * dim * config.hidden_dim),
            ffn3: new Float32Array(config.n_layers * config.hidden_dim * dim),

            rms_final: new Float32Array(dim),
            classify: new Float32Array(dim * config.vocab_size)
        };  
  
        let readIntoBuffer = (
            buff: Float32Array, position: number
        ) => {
            return fs.readSync(this.f, buff, 0, buff.length, position);
        }
        
        let position = 0;
        position += readIntoBuffer(weights.token_embedding_table, position);
        position += readIntoBuffer(weights.rms_att, position);
        position += readIntoBuffer(weights.rms_ffn, position);

        position += readIntoBuffer(weights.query, position);
        position += readIntoBuffer(weights.key, position);
        position += readIntoBuffer(weights.value, position);
        position += readIntoBuffer(weights.output, position);

        position += readIntoBuffer(weights.ffn1, position);
        position += readIntoBuffer(weights.ffn2, position);
        position += readIntoBuffer(weights.ffn3, position);
    
        position += readIntoBuffer(weights.rms_final, position);
        position += readIntoBuffer(weights.classify, position);

        return weights; 
    }
}

class Transformer {
    constructor(
        readonly config: Config,
        readonly weights: Weights,
        readonly b: RuntimeBuffers
    ) {}

    static create_buffers(config: Config): RuntimeBuffers {
        const head_size = config.dim / config.n_heads;
        const kv_dim = config.n_kv_heads * head_size;
        let buffers: RuntimeBuffers = {
            x: new Float32Array(config.dim),
            xb: new Float32Array(config.dim),
            xb2: new Float32Array(config.dim),
            hb: new Float32Array(config.hidden_dim),
            hb2: new Float32Array(config.hidden_dim),
             
            q: new Float32Array(config.dim),

            att: new Float32Array(config.n_heads * config.seq_len),
            logits: new Float32Array(config.dim * config.vocab_size),

            key_cache: new Float32Array(
                config.n_layers * config.seq_len * kv_dim
            ),
            value_cache: new Float32Array(
                config.n_layers * config.seq_len * kv_dim
            )
        };

        return buffers;
    }

    forward(token: number, position: number): Float32Array  {
        // size of query vector
        const dim = this.config.dim;       
        const hidden_dim = this.config.hidden_dim;
        const head_size = dim / this.config.n_heads;
        // size of key,value vectors
        const kv_dim = this.config.n_kv_heads * head_size;
        // ratio between num query heads and kv heads 
        const kv_mul = Math.trunc(this.config.n_heads / this.config.n_kv_heads);
        const seq_len = this.config.seq_len;
        
        let x = this.b.x;
        let xb = this.b.xb;
        let xb2 = this.b.xb2;
        let hb = this.b.hb;
        let hb2 = this.b.hb2;
            
	    x.set(view(
            this.weights.token_embedding_table,
            token * dim, 
            dim
        ));

        for (let layer_i = 0; layer_i < this.config.n_layers; layer_i++) {
            rmsnorm(
                xb,
                x,
                view(this.weights.rms_att, layer_i * dim, dim)
            );
           
            // key and value point to the kv cache
            const layer_off = layer_i * this.config.seq_len * kv_dim;
            const query = this.b.q;
            const key   = view(this.b.key_cache, layer_off + position * kv_dim, kv_dim);
            const value = view(this.b.value_cache, layer_off + position * kv_dim, kv_dim);
             
            vecmatmul(
                query,
                view(this.weights.query, layer_i * dim * dim, dim * dim),
                xb
            );
            vecmatmul(
                key,
                view(this.weights.key, layer_i * dim * kv_dim, dim * kv_dim),
                xb
            );
            vecmatmul(
                value,
                view(this.weights.value, layer_i * dim * kv_dim, dim * kv_dim),
                xb
            );

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (let i = 0; i < dim; i += 2) {
                let head_dim = i % head_size;
                let freq = 1.0 / Math.pow(10000.0, head_dim / head_size);
                let val = position * freq;

                // how many vectors? 2 = q & k, 1 = q only
                let rotn = (i < kv_dim ? 2 : 1);
                for (let v = 0; v < rotn; v++) {
                    let vec = (v == 0 ? query : key);
                    let v0 = vec[i];
                    let v1 = vec[i+1];
                    // repr each two elems as the mag of a rotated vector
                    vec[i]   = v0 * Math.cos(val) - v1 * Math.sin(val);
                    vec[i+1] = v0 * Math.sin(val) + v1 * Math.cos(val);
                }
            }

            // multihead attention
            for (let hi = 0; hi < this.config.n_heads; hi++) {
                // vecs for curr head
                let q   = view(this.b.q, hi * head_size, head_size);
                let att = view(this.b.att, hi * seq_len, seq_len);

                for (let ti = 0; ti <= position; ti++) {
                    let k = view(
                        this.b.key_cache,
                        layer_off + ti * kv_dim + Math.trunc(hi / kv_mul) * head_size,
                        head_size
                    );

                    // attention score is dot product
                    let score = 0.0;
                    for (let i = 0; i < head_size; i++) {
                        score += q[i] * k[i];
                        att[ti] = score / Math.sqrt(head_size);
                    }
                }

                // softmax scores to get attention weights
                softmax(view(att, 0, position + 1));
    
                // weighted sum of the values, store back into x
                let xb = view(this.b.xb, hi * head_size, head_size);
                xb.fill(0, 0, head_size);
                for (let t = 0; t <= position; t++) {
                    // get value vec for this head and timestep
                    let v = view(
                        this.b.value_cache,
                        layer_off + t * kv_dim + (hi / kv_mul) * head_size,
                        head_size
                    );

                    // accumulate the weighted values
                    // from each ts with its respective
                    // attention score.

                    // separate heads get concatenated
                    for (let i = 0; i < head_size; i++)
                        xb[i] += att[t] * v[i];
                }
            }

            // final matmul to get the output of the attention
            vecmatmul(
                xb2,
                view(this.weights.output, layer_i * dim * dim, dim * dim),
                xb
            );

            // residual connection
            for (let i = 0; i < dim; i++)
                x[i] += xb2[i];

            rmsnorm(
                xb,
                x,
                view(this.weights.rms_ffn, layer_i * dim, dim)
            );

            // w2 @ (silu(w1 @ x) * (w3 @ x))
            vecmatmul(
                hb,
                view(
                    this.weights.ffn1,
                    layer_i * dim * hidden_dim,
                    dim * hidden_dim
                ),
                xb
            );
            vecmatmul(
                xb2,
                view(
                    this.weights.ffn3,
                    layer_i * dim * hidden_dim,
                    dim * hidden_dim
                ),
                xb
            );

            for (let i = 0; i < hidden_dim; i++) {
                // silu(x) = x*σ(x) // SwiGLU
                let val = hb[i];
                val *= 1.0 / (1.0 + Math.exp(-val));

                // elementwise multiply with w3(x)
                val *= hb2[i];
                hb[i] = val;
            }

            // last matmul: w2 @ ...
            vecmatmul(
                xb,
                view(
                    this.weights.ffn2,
                    layer_i * dim * hidden_dim, 
                    hidden_dim * dim
                ),
                hb
            );

            // residual connection
            for (let i = 0; i < dim; i++)
                x[i] += xb[i];
        }

        rmsnorm(x, x, this.weights.rms_final);

        // classifier into logits
        vecmatmul(this.b.logits, this.weights.classify, x);
        return this.b.logits;
    }
}

function generate_response(prompt: string): string {
    // TODO: tokenize prompt
    const tokenizer = new Tokenizer();

    const fileLoader = new FileLoader("llama-2-7b-chat/consolidated.00.pth");
    const config = fileLoader.load_config();
    console.log(`config = ${config}`);

    const transformer = new Transformer(
        config,
        fileLoader.load_weights(config),
        Transformer.create_buffers(config)
    );
    
    // TODO: read through how the temperature thing works & figure out how to generate a response

    return "NOT YET IMPLEMENTED";
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
