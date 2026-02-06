import fs from 'fs';


// U must be comparable
function bsearch<T, U>(xs: T[], t: U, f: (x:T) => U): T | null {
    let topi = xs.length - 1;
    let boti = 0;

    while ((topi - boti) <= 4) {
        let m = Math.trunc((topi + boti)/2);
        if (t > f(xs[m])) boti = m;
	else topi = m;
   
    // we do a small stride at the end for perf
    // (also to avoid worrying about rounding issues)
    for (let i = boti; i <= topi; i++)
        if (xs[i] == t)
            return i;

    return null;
}

interface TokenIndex {
   str: number[],
   id: number,
};

class Tokenizer {
    // TODO: how to do array of array?
    vocab: number[][];
    vocab_scores: Float32Array;
    max_token_length: number;

    sorted_vocab: TokenIndex[];
    byte_pieces: number[];

    constructor(
        tokenizer_path: string, readonly vocab_size: number
    ) {
        this.vocab = new Array(vocab_size);
        this.vocab_scores = new Float32Array(vocab_size);
        this.sorted_vocab = [];

        this.byte_pieces = [];
        for (let i = 0; i < 256; i++)
            this.byte_pieces.push(String.fromCharCode(i));
        
        let f = fs.openSync(tokenizer_path, "r");
        if (f == null)
            throw "ERROR: bad checkpoint path";

        const HDR_SIZE = 4;
        const data = new Uint8Array(HDR_SIZE);
        const bytes_read = fs.readSync(f, data, 0, HDR_SIZE, 0);
        if (bytes_read != HDR_SIZE)
            throw "ERROR: failed to read config";

        const view = new DataView(data.buffer);
        this.max_token_length = view.getInt32(0 * 4, true);

        let position = HDR_SIZE;
        for (let i = 0; i < vocab_size; i++) {
            const ENTRY_HDR_SIZE = 4 + 4;
            const data = new Uint8Array(ENTRY_HDR_SIZE);
            let bytes_read = fs.readSync(f, data, 0, ENTRY_HDR_SIZE, position);
            if (bytes_read != ENTRY_HDR_SIZE)
                throw "ERROR";
            position += bytes_read;
        
            const view = new DataView(data.buffer);
            this.vocab_scores[i] = view.getFloat32(0 * 4, true); 
            let len = view.getInt32(1 * 4, true);

            const vocab_data = new Uint8Array(len);
            bytes_read = fs.readSync(f, vocab_data, len, 1, position);
            if (bytes_read != len)
                throw "ERROR";

            for (let j = 0; j < len; j++)
	        this.vocab[i].push(vocab_data[j]);

            position += bytes_read;
        }
    }
    // TODO: implement this & fix the sorting of vocab
    static compare_raw_str(a:number[], b:number[]): number {
        if (a.str == b.str) return 0;
        else if (a.str < b.str) return -1;
        else return 1;
    }
    // find the perfect match for str in vocab, return its index or -1 if not found
    str_lookup(str:number[]): number {
        let t = bsearch(this.sorted_vocab, str, x => x.str, Tokenizer.compare_raw_str);
        return res == null ? -1 : t.id;
    }
    encode(
        // todo: ensure text is ascii, despite being u16
        text: string, bos: boolean, eos: boolean
    ): number[] {
        if (text == "") throw "text cannot be empty";

        if (this.sorted_vocab.length == 0) {
            // lazily alloc and sort the vocabulary
            for (let i = 0; i < this.vocab_size; i++) {
                this.sorted_vocab[i] = {
                    str: this.vocab[i],
                    id: i,
                };
            }

            this.sorted_vocab.sort(
                Tokenizer.compare_tokens);
        }

        // merge candidate buffer
        let buffer: number[] = [];
        let tokens: number[] = [];

        // optional BOS (=1) (<s>) token
        if (bos) tokens.push(1);

        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        // TODO(gabe): what is he talking about?
        if (text.length > 0) {
            let dummy_prefix = this.str_lookup(" ");
            tokens.push(dummy_prefix);
        }

        // process the raw (UTF-8) byte sequence of the input string
        for (let ci = 0; ci < text.length; ci++) {
            // reset buffer if the current byte is ASCII or a leading byte
            // in UTF-8, all continuation bytes start with "10" in first two bits
            if ((text.charCodeAt(ci) & 0xC0) != 0x80)
                // this must be a leading byte (11...) or an ASCII char (0x...)
                buffer = [];

            buffer.push(text.charCodeAt(ci));

            // while the next character is a continuation byte, continue appending
            // but if there are too many of them, just stop (> 4 is invalid anyways)
            if (
                (text.harCodeAt(ci+1)&0xC0) == 0x80
                && buffer.length < 4
            )
                continue;

            // ci+1 is not a continuation byte, so we've read in a full codepoint
            // TODO: update buffer to operate on byte sequences as "strings"
            // TODO: OR store buffer as one big string
            let id = this.str_lookup(buffer);
            if (id != -1) {
                tokens.push(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                for (let byte of buffer)
                    tokens.push(byte + 3);
            }

            buffer = [];
        }

        while (true) {
            let best_score = -1e10;
            let best_id = -1;
            let best_idx = -1;

            for (let i = 0; i < tokens.length-1; i++)  {
                const t: number = tokens[i];
                const tnext: number = tokens[i+1];

                // check if we can merge the pair (t, tnext)
                let merged = 
                    this.vocab[t] + this.vocab[tnext];

                let id = this.str_lookup(merged);
                if (id != -1 && this.vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = this.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1)
                break; // no more pairs to merge

            // merge consecutive pair into new token
            tokens[best_idx] = best_id;
            tokens.splice(best_idx+1, 1);
        }

        // optional EOS (=2) (</s>) token
        if (eos) tokens.push(2);

        return tokens;
    }
    decode(prev_token: number, token: number): string {
        let piece: string = this.vocab[token];

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if (prev_token == 1 && piece[0] == ' ')
            piece = piece.slice(1);

        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        let m = piece.match("<0x\d+>");
        if (m) {
            return this.byte_pieces[Number(m[0])];
        } else {
            return piece;
        }
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
    ss += 1e-5; // avoid LARGE ss after 1/x

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

    let sum = 0.0;
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

    shared_weights: boolean;
}
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
        this.f = fs.openSync(checkpoint_path, "r");
        if (this.f == null)
            throw "ERROR: bad checkpoint path"
    }

    load_config(): Config {
        const CONFIG_NUM_ELEMENTS = 7;
        const CONFIG_NUM_BYTES = CONFIG_NUM_ELEMENTS * 4;

        const data = new Uint8Array(CONFIG_NUM_BYTES);
        const bytes_read = fs.readSync(this.f, data, 0, CONFIG_NUM_BYTES, 0);
        if (bytes_read != CONFIG_NUM_BYTES)
            throw "ERROR: failed to read config"

        const view = new DataView(data.buffer);
        const vocab_size = view.getInt32(5 * 4, true);
        let config: Config = {
            dim:        view.getInt32(0 * 4, true),
            hidden_dim: view.getInt32(1 * 4, true),
            n_layers:   view.getInt32(2 * 4, true),
            n_heads:    view.getInt32(3 * 4, true),
            n_kv_heads: view.getInt32(4 * 4, true),
            vocab_size: Math.abs(vocab_size),
            seq_len:    view.getInt32(6 * 4, true),

            shared_weights: vocab_size > 0,
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

interface ProbIndex {
    prob: number;
    index: number;
}

class Sampler {
    constructor(
        readonly temperature: number,
        readonly topp: number
    ) {}

    sample_argmax(probs: Float32Array): number {
        let max_i = 0;
        let max_p = probs[0];

        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > max_p) {
                max_i = i;
                max_p = probs[i];
            }
        }

        return max_i;
    }

    sample_mult(
        probs: Float32Array,
	    coin: number
    ): number {
        // probabilities must sum to 1!
        let cdf = 0.0;
        for (let i = 0; i < (probs.length - 1); i++) {
            cdf += probs[i];
            if (coin < cdf)
                return i;
	    }
	    return probs.length - 1;
    }

    sample_topp(
        probs: Float32Array,
        coin: number
    ): number {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
	    // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        let prob_index: ProbIndex[] = [];

        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        const cutoff = (1.0-this.topp) / (probs.length - 1);
            for (let i = 0; i < probs.length; i++) {
            if (probs[i] >= cutoff) {
                prob_index.push({
                    index: i,
                    prob: probs[i],
                });
            }
	    }
	
        prob_index.sort(x => x.prob);

        let cumulative_prob = 0.0;
        let last_i = prob_index.length - 1; 
        for (let i = 0; i < prob_index.length; i++) {
            cumulative_prob += prob_index[i].prob;
            if (cumulative_prob > this.topp) {
                last_i = i;
                break;
            }
        }

        // sample from the truncated list
        let r = coin * cumulative_prob;
        let cdf = 0.0;
        for (let i = 0; i <= last_i; i++) {
            cdf += prob_index[i].prob;
            if (r < cdf)
                return prob_index[i].index;
        }

        return prob_index[last_i].index;
    }

    sample(logits: Float32Array): number {
        let next;
        if (this.temperature == 0.0) {
            // token with the highest probability
            next = this.sample_argmax(logits);
        } else {
            // softmax maintains differences, so small temperature amplifies differences and approximates argmax
            for (let qi = 0; qi < this.vocab_size; qi++)
                logits[qi] /= this.temperature;

            softmax(logits);

            let coin = Math.random();
            if (this.topp <= 0 || this.topp >= 1) {
                // simply sample from the predicted probability distribution
                next = this.sample_mult(logits, coin);
            } else {
                // top-p sampling clamps lowest to zero
                next = this.sample_topp(logits, coin);
            }
        }
        return next;
    }
}

function generate_response(prompt: string): string {
    const checkpoint_path = "models/stories15M.bin";
    const tokenizer_path = "tokenizer.bin";
    const temperature = 1.0;
    const topp = 0.9;
    const steps = 256;

    const sampler = Sampler(temperature, topp);

    const tokenizer = new Tokenizer(tokenizer_path);

    //const fileLoader = new FileLoader("llama2/llama-2-7b-chat/params.json");
    const fileLoader = new FileLoader("models/stories15M.bin");
    const config = fileLoader.load_config();
    console.log(`config = ${JSON.stringify(config, null, 2)}`);

    const transformer = new Transformer(
        config,
        fileLoader.load_weights(config),
        Transformer.create_buffers(config)
    );
    
    let prompt_tokens = tokenizer.encode(
        prompt, true, false);
    if (prompt_tokens.length < 1)
        throw "ERROR: too few tokens"

    let next_token: number; // will store the next token in the sequence
    let token = prompt_tokens[i];
    let i = 0;
    while (i < steps) {
        let logits = transformer.forward(token, i);

        // advance the state machine
        if (i < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt toke
            next_token = prompt_tokens[i + 1];
        } else {
            // otherwise sample the next token from the logits
            next_token = sample(sampler, logits);
        }

        i += 1;
        
	// terminating condition:BOS (=1) token delimits sequences
	if (next_token == 1) { break; }

	// print the token as string, decode it with the Tokenizer object
	string piece = tokenizer.decode(token, next_token);
	
	// TODO: impl this function
	// same as printf("%s", piece), but skips "unsafe" bytes
	safe_print(piece);
	
	token = next_token;
    }

    console.log();

    return "NOT YET IMPLEMENTED";
}

import readline from 'readline';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});
rl.question("Prompt: ", (prompt: string) => {
    console.log(`Response: ${generate_response(prompt)}!`);
    rl.close();
});
