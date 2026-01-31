
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
    // stores the embedding for each token
    token_embedding_table: Float32Array;

    rms_att: Float32Array; // (layer, dim)
    rms_ffn: Float32Array;

    // multi-head attention
    query: Float32Array; // (layer, dim, n_heads * head_size)
    key: Float32Array;   // (layer, dim, n_kv_heads * head_size)
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
    xb: Float32Array; // (dim,)
    xb2: Float32Array;
    hb: Float32Array; // (hidden_dim,)
    hb2: Float32Array;

    // for attention
    q: Float32Array; // (dim,)
    // TODO: these are just pointers and need not be here
    // k: Float32Array; // (not dim,)
    // v: Float32Array; // (not dim,)

    // TODO: this
    att: Float32Array; // (n_heads, seq_len)
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

             query: new Float32Array(config.dim * config.n_heads * head_size),
             key: new Float32Array(config.dim * config.n_kv_heads * head_size),
             value: new Float32Array(config.dim * config.n_kv_heads * head_size),
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
            xb: new Float32Arrat(config.dim),
            x_attn_out: new Float32Arrat(config.dim),
             
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
        
        let x = b.x;
        let content_row = weights
            .token_embedding_table
            .subarray(token * dim, dim);
	// copy the token embedding into x
	x.set(content_row);

        for (let layer_i = 0; layer_i < this.config.n_layers; layer_i++) {
            // TODO: implement this!
            rmsnorm(
                b.xb,
                x,
                weights.rms_att + layer_i * dim
            );
           
            // key and value point to the kv cache
            const layer_off = layer_i * config.seq_len * kv_dim;
            const key   = b.key_cache.subarray(layer_off + position * kv_dim, kv_dim);
            const value = b.value_cache.subarray(layer_off + position * kv_dim, kv_dim);
             
            vecmatmul(
                b.q,
                weights.query + layer_i * dim * dim
                b.xb);
            vecmatmul(
                key,
                weights.key + layer_i * dim * kv_dim
                b.xb);
            vecmatmul(
                value,
                weights.value + layer_i * dim * kv_dim
                b.xb);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (let i = 0; i < dim; i += 2) {
                let head_dim = i % head_size;
                let freq = 1.0f / Math.pow(10000.0f, head_dim / (float)head_size);
                let val = pos * freq;

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
            for (let hi = 0; hi < config.n_heads; hi++) {
	        // vecs for this head
		let q   = b.q.subarray(hi * head_size, head_size);
		let att = b.att.subarray(hi * config.seq_len, config.seq_len);

		// iter all timesteps, including current
		for (let t = 0; t <= pos; t++) {
		    let k = b.key_cache.subarray(
		         layer_off
			 + t * kv_dim
			 + Math.trunc(hi / kv_mul) * head_size,
			 head_size);

                    // attention score is dot product
		    let score = 0.0f;
		    for (let i = 0; i < head_size; i++)
                        score += q[i] * k[i];

	            att[t] = score / Math.sqrt(head_size);
	        }

                // softmax scores to get attention weights
                softmax(att.subarray(0, pos + 1));
   
                // weighted sum of the values, store back into x
	        let xb = b.xb.subarray(h * head_size, head_size);
		xb.fill(0, 0, head_size);
	       
	        for (let t = 0; t <= pos; t++) {
	            // get value vec for this head and timestep
                    let v = b.value_cache.subarray(
		        layer_off
		        + t * kv_dim
		        + (h / kv_mul) * head_size,
		        head_size
		    );

		    // accumulate the weighted values
		    // from each ts with its respective
		    // attention score.

                    // all heads get concat
		    for (let i = 0; i < head_size; i++)
		        xb[i] += att[t] * v[i];
	        }
            }

            // final matmul to get the output of the attention
            vecmatmul(
	        b.x_attn_out,
		weights.output.subarray(
		    layer_i * dim * dim, dim * dim
		),
		b.xb
            );

            // residual connection (x -> x_attn_out)
            for (let i = 0; i < dim; i++)
	        x[i] += b.x_attn_out[i];

	    rmsnorm(
	        b.xb,
		x,
		weights.rms_ffn.subarray(
		    layer_i * dim, dim
		)
	    );

	    // self.w2(F.silu(self.w1(x)) * self.w3(x))
	    vecmatmul(
	        b.hb,
		weights.ffn1.subarray(
		    layer_i * dim * hidden_dim,
		    dim * hidden_dim
		)
                b.xb
	    );
	    vecmatmul(
	        b.xb2, 
		weights.ffn3.subarray(
	            layer_i * dim * hidden_dim,
		    dim * hidden_dim
		),
		b.xb
	    );

            for (let i = 0; i < hidden_dim; i++) {
                // SwiGLU non-linearity
		// silu(x)=x*σ(x)
	        let val = b.hb[i];
		val *= 1.0f / (1.0f + Math.exp(-val));

		// elementwise multiply with w3(x)
		val *= b.hb2[i];
		b.hb[i] = val;
	    }

            // last matmulw2 * (silu(w1) * w3)
            vecmatmul(
	        b.xb,
		weights.ffn2.subarray(
		    layer_i * dim * hidden_dim, 
		    hidden_dim * dim
                ),
		b.hb
            );

	    // residual connection
            for (let i = 0; i < dim; i++)
	        x[i] += b.xb[i];

        }

	rmsnorm(x, x, weights.rms_final);

        // classifier into logits
	vecmatmul(b.logits, weights.classify, x);
	return b.logits;
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
