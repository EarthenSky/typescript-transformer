import fs from 'fs';

// U must be comparable
function bsearch<T, U>(
    xs: T[],
    t: U,
    f: (x:T) => U,
    compare: (x: U, y: U) => number
): number | null {
    let topi = xs.length - 1;
    let boti = 0;

    while ((topi - boti) > 4) {
        let mi = Math.trunc((topi + boti)/2);
        let fm = f(xs[mi]);
        if (compare(t, fm) > 0) boti = mi;
        else topi = mi;
    }

    // we do a small stride at the end for perf
    // (also to avoid worrying about rounding issues)
    for (let i = boti; i <= topi; i++) {
        if (compare(f(xs[i]), t) == 0)
            return i;
    }

    return null;
}

interface TokenIndex {
   bytes: number[],
   id: number,
};

const UNK = 0;
const BOS = 1;
const EOS = 2;

class Tokenizer {
    vocab: string[]; // token to utf-8 string mapping
    sorted_vocab: TokenIndex[]; // searchable format
    vocab_scores: Float32Array;

    byte_pieces: string[];

    constructor(
        tokenizer_path: string,
        readonly vocab_size: number
    ) {    
        this.vocab = [];
        this.vocab_scores = new Float32Array(this.vocab_size);
        this.sorted_vocab = [];

        this.byte_pieces = [];
        for (let i = 0; i < 256; i++)
            this.byte_pieces.push(String.fromCharCode(i));
        
        let f = fs.openSync(tokenizer_path, "r");
        if (f == null)
            throw "ERROR: bad checkpoint path";

        const HDR_SIZE = 4;
        // const data = new Uint8Array(HDR_SIZE);
        // const bytes_read = fs.readSync(f, data, 0, HDR_SIZE, 0);
        // if (bytes_read != HDR_SIZE)
        //     throw "ERROR: failed to read config";
        //
        // const view = new DataView(data.buffer);
        // const max_token_length = view.getInt32(0 * 4, true);

        let position = HDR_SIZE;
        for (let i = 0; i < this.vocab_size; i++) {
            const ENTRY_HDR_SIZE = 4 + 4;
            const data = new Uint8Array(ENTRY_HDR_SIZE);
            let bytes_read = fs.readSync(f, data, 0, ENTRY_HDR_SIZE, position);
            if (bytes_read != ENTRY_HDR_SIZE)
                throw "ERROR";
            position += bytes_read;
        
            const view = new DataView(data.buffer);
            this.vocab_scores[i] = view.getFloat32(0 * 4, true); 
            const len = view.getInt32(1 * 4, true);

            const vocab_data = new Uint8Array(len);
            bytes_read = fs.readSync(f, vocab_data, 0, len, position);
            if (bytes_read != len)
                throw "ERROR";
            position += bytes_read;

	        this.vocab[i] = (new TextDecoder("utf-8")).decode(vocab_data);
        }
    }
    static compare_bytes(a:number[], b:number[]): number {
        for (let i = 0; i < Math.min(a.length, b.length); i++) {
            if (a[i] < b[i])
                return -1;
            else if (a[i] > b[i])
                return 1;
        }

        if (a.length < b.length) return -1;
        else if (a.length > b.length) return 1;
        else return 0;
    }
    static bytes_from_str(s:string): number[] {
        return Array.from((new TextEncoder()).encode(s));
    }
    str_lookup(bytes:number[]): number {
        // find the perfect match for str in vocab, return its index or -1 if not found
        let ti = bsearch(this.sorted_vocab, bytes, x => x.bytes, Tokenizer.compare_bytes);
        return ti == null ? -1 : this.sorted_vocab[ti].id;
    }
    encode(
        text: string, bos: boolean, eos: boolean
    ): number[] {
        // if (text == "") throw "text cannot be empty";

        if (this.sorted_vocab.length == 0) {
            // lazily alloc and sort the vocabulary
            for (let i = 0; i < this.vocab_size; i++) {
                this.sorted_vocab[i] = {
                    bytes: Tokenizer.bytes_from_str(this.vocab[i]),
                    id: i,
                };
            }

            this.sorted_vocab.sort((x,y) => Tokenizer.compare_bytes(x.bytes, y.bytes));
        }

        // merge candidate buffer
        let buffer: number[] = [];
        let tokens: number[] = [];

        // optional BOS (<s>) token
        if (bos) tokens.push(BOS);

        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        // TODO: (gabe) what is he talking about?
        if (text.length > 0) {
            let dummy_prefix = this.str_lookup(
                [" ".charCodeAt(0)]
            );
            tokens.push(dummy_prefix);
        }

        let text_bytes = new TextEncoder().encode(text);

        // process the raw (UTF-8) byte sequence of the input string
        for (let ci = 0; ci < text_bytes.length; ci++) {
            // reset buffer if the current byte is ASCII or a leading byte
            // in UTF-8, all continuation bytes start with "10" in first two bits
            if ((text_bytes[ci] & 0xC0) != 0x80)
                // this must be a leading byte (11...) or an ASCII char (0x...)
                buffer = [];

            buffer.push(text_bytes[ci]);

            // continue while the next character is a continuation byte
            if (
                (ci+1) < text_bytes.length
                && (text_bytes[ci+1] & 0xC0) == 0x80
                && buffer.length < 4
            )
                continue;

            // ci+1 is not a continuation byte, so we've read in a full codepoint
            let id = this.str_lookup(buffer);
            if (id != -1) {
                tokens.push(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here b/c the first 3 tokens are <unk>, <s>, </s>
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

                const t_bytes = Tokenizer.bytes_from_str(this.vocab[t]);
                const tnext_bytes = Tokenizer.bytes_from_str(this.vocab[tnext]);

                // check if we can merge the pair (t, tnext)
                let merged: number[] = t_bytes.concat(tnext_bytes);

                let id = this.str_lookup(merged);
                if (id != -1 && this.vocab_scores[id] > best_score) {
                    // merge pair exists in vocab! record its score and position
                    best_score = this.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            // no more pairs to merge
            if (best_idx == -1)
                break;

            // merge consecutive pair into new token
            tokens[best_idx] = best_id;
            tokens.splice(best_idx+1, 1);
        }

        // optional EOS (</s>) token
        if (eos) tokens.push(EOS);

        return tokens;
    }
    decode(prev_token: number, token: number): string {
        let piece: string = this.vocab[token];

        // following BOS, sentencepiece decoder strips any leading whitespace
        if (prev_token == BOS && piece[0] == " ")
            piece = piece.slice(1);

        // some tokens designate raw bytes, and look like <0x01>
        // parse this & return the byte
        let is_raw_bytes =
            (piece.length > 4)
            && piece[0] == "<"
            && piece[1] == "0"
            && piece[2] == "x"
            && piece[piece.length-1] == ">";

        if (is_raw_bytes) {
            return this.byte_pieces[
                parseInt(piece.slice(3, -1), 16)
            ];
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
    if (out.length != x.length || weight.length != x.length) {
        console.log(`out=${out.length} weight=${weight.length} x=${x.length}`);
        throw new Error("BAD rmsnorm");
    }

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
    if (M.length != out.length * x.length) {
        console.log(`n=${n} M=${M.length} m=${m}`);
        throw new Error("BAD matmul");
    }

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
    size: number,
): Float32Array {
    if (!Number.isInteger(start) || !Number.isInteger(size)) {
        console.log(`view(start=${start}, size=${size})`);
        throw "ERROR: not integer passed to view";
    }
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
    token_embedding_table: Float32Array; // (dim, vocab_size)
    rms_att: Float32Array; // (n_layers, dim)

    // multi-head attention
    query: Float32Array; // (n_layers, dim, n_heads * head_size)
    key: Float32Array;   // (n_layers, dim, n_kv_heads * head_size)
    value: Float32Array;
    output: Float32Array;

    rms_ffn: Float32Array;

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

    att: Float32Array; // (n_heads, seq_len)
    logits: Float32Array;

    // kv cache
    key_cache: Float32Array; // (n_layers, seq_len, n_kv_heads * head_size) 
    value_cache: Float32Array;
};

class FileLoader {
    static CONFIG_NUM_ELEMENTS = 7;
    static CONFIG_NUM_BYTES = FileLoader.CONFIG_NUM_ELEMENTS * 4;

    readonly f: number;

    constructor (checkpoint_path: string) {
        this.f = fs.openSync(checkpoint_path, "r");
        if (this.f == null)
            throw "ERROR: bad checkpoint path"
    }

    load_config(): Config {
        const data = new Uint8Array(FileLoader.CONFIG_NUM_BYTES);
        const bytes_read = fs.readSync(this.f, data, 0, FileLoader.CONFIG_NUM_BYTES, 0);
        if (bytes_read != FileLoader.CONFIG_NUM_BYTES)
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
        if (config.n_heads * head_size != dim)
            throw "ERROR in load_weights"

        let weights: Weights = {
            token_embedding_table: new Float32Array(dim * config.vocab_size),
            rms_att: new Float32Array(config.n_layers * dim),
    
            query: new Float32Array(config.n_layers * dim * dim),
            key: new Float32Array(config.n_layers * dim * config.n_kv_heads * head_size),
            value: new Float32Array(config.n_layers * dim * config.n_kv_heads * head_size),
            output: new Float32Array(config.n_layers * dim * dim),

            rms_ffn: new Float32Array(config.n_layers * dim),

            ffn1: new Float32Array(config.n_layers * config.hidden_dim * dim),
            ffn2: new Float32Array(config.n_layers * dim * config.hidden_dim),
            ffn3: new Float32Array(config.n_layers * config.hidden_dim * dim),

            rms_final: new Float32Array(dim),
            classify: new Float32Array(dim * config.vocab_size),
        };  

        let readIntoBuffer = (
            buff: Float32Array, position: number
        ) => {
            if ((buff.length*4) >= 2**30)
                console.log(`Reading weights of size = ${((4 * buff.length) / 1000 / 1000).toFixed(1)} MB`);

            // read in chunks smaller than 2^31, because it gets casted into an i32
            let amountRead = 0;
            let amountLeft = 4 * buff.length;
            const LOTS = 2**31 - 1;
            while (amountLeft != 0) {
                let bytesRead = fs.readSync(
                    this.f, buff, amountRead, Math.min(LOTS, amountLeft), position + amountRead);
                amountRead += bytesRead;
                amountLeft -= bytesRead;
            }
            return amountRead;
        }
    
        let position = FileLoader.CONFIG_NUM_BYTES;
        position += readIntoBuffer(weights.token_embedding_table, position);
        position += readIntoBuffer(weights.rms_att, position);

        position += readIntoBuffer(weights.query, position);
        position += readIntoBuffer(weights.key, position);
        position += readIntoBuffer(weights.value, position);
        position += readIntoBuffer(weights.output, position);

        position += readIntoBuffer(weights.rms_ffn, position);

        position += readIntoBuffer(weights.ffn1, position);
        position += readIntoBuffer(weights.ffn2, position);
        position += readIntoBuffer(weights.ffn3, position);
    
        position += readIntoBuffer(weights.rms_final, position);
        position += config.seq_len * (head_size / 2); // skip freq_cis_real
        position += config.seq_len * (head_size / 2); // skip freq_cis_imag

        // TODO: what is the purpose of shared weights?
        if (config.shared_weights) {
            weights.classify.set(weights.token_embedding_table);
        } else {
            position += readIntoBuffer(weights.classify, position);
        }

        return weights; 
    }
}

function does_not_contain_nan(x:any) {
    for (let i = 0; i < x.length; i++) {
        if (Number.isNaN(x[i])) {
            console.log(`(x) FOUND NAN at ${i} (pre-rmsnorm)`)
            throw "FOUND NAN"
        }
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
            logits: new Float32Array(config.vocab_size),

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
                let rotn = (i < kv_dim) ? 2 : 1;
                for (let v = 0; v < rotn; v++) {
                    let vec = (v == 0) ? query : key;
                    let v0 = vec[i];
                    let v1 = vec[i+1];
                    // repr each two elems as the mag of a rotated vector
                    vec[i]   = v0 * Math.cos(val) - v1 * Math.sin(val);
                    vec[i+1] = v0 * Math.sin(val) + v1 * Math.cos(val);
                }
            }

            // multihead attention
            for (let hi = 0; hi < this.config.n_heads; hi++) {
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
                    }
                    att[ti] = score / Math.sqrt(head_size); // TODO: why this?
                }

                // softmax scores to get attention weights
                softmax(view(att, 0, position + 1));
    
                // weighted sum of the values, store back into x
                let xb = view(this.b.xb, hi * head_size, head_size);
                xb.fill(0);
                for (let ti = 0; ti <= position; ti++) {
                    // get value vec for this head and timestep
                    let v = view(
                        this.b.value_cache,
                        layer_off + ti * kv_dim + Math.trunc(hi / kv_mul) * head_size,
                        head_size
                    );

                    // accumulate the weighted values
                    // from each ts with its respective
                    // attention score.

                    // separate heads get concatenated
                    for (let i = 0; i < head_size; i++)
                        xb[i] += att[ti] * v[i];
                }
            }

            // final matmul to get the output of the attention
            vecmatmul(
                xb2,
                view(this.weights.output, layer_i * dim * dim, dim * dim),
                xb
            );

            // residual connection
            for (let i = 0; i < x.length; i++)
                x[i] += xb2[i];

            rmsnorm(
                xb, x,
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
                hb2,
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
        does_not_contain_nan(this.b.logits);

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
        readonly topp: number,
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
	    coin: number,
    ): number {
        // probabilities must sum to 1 !
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
        coin: number,
    ): number {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
	    // have very low probabilities and are less likely to go "off the rails".

        let prob_index: ProbIndex[] = [];

        // sort indices in descending order of probabilities, then remove all
        // values smaller than (1 - topp) / (n - 1)
        const cutoff = (1.0 - this.topp) / (probs.length - 1);
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] >= cutoff) {
                prob_index.push({
                    index: i,
                    prob: probs[i],
                });
            }
	    }
	
        // max sort, so we select largest prob events first
        prob_index.sort((x,y) => y.prob - x.prob);

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
            for (let qi = 0; qi < logits.length; qi++)
                logits[qi] /= this.temperature;

            // does_not_contain_nan(logits);
            softmax(logits);

            let coin = Math.random();
            if (this.topp <= 0 || this.topp >= 1) {
                // sample from the predicted probability distribution
                next = this.sample_mult(logits, coin);
            } else {
                next = this.sample_topp(logits, coin);
            }
        }
        return next;
    }
}

interface InputParameters {
    checkpoint_path: string;
    tokenizer_path: string;
    temperature: number;
    topp: number;
    max_steps: number;
};

function generate_response(params: InputParameters, prompt: string): string {
    const fileLoader = new FileLoader(params.checkpoint_path);
    const config = fileLoader.load_config();
    console.log(`\nconfig = ${JSON.stringify(config, null, 2)}\n`);

    console.log("loading Transformer...");
    const transformer = new Transformer(
        config,
        fileLoader.load_weights(config),
        Transformer.create_buffers(config)
    );

    console.log("loading Tokenizer...");
    const tokenizer = new Tokenizer(params.tokenizer_path, config.vocab_size);
    
    console.log("loading Sampler...");
    const sampler = new Sampler(params.temperature, params.topp);
    
    let prompt_tokens = tokenizer.encode(
        prompt, true, false);
    if (prompt_tokens.length < 1)
        throw "ERROR: too few tokens"

    // encode & decode seem to be working properly
    console.log("\nPrompt Tokens")
    console.log(prompt_tokens.toString())
    for (let i = 0; i < prompt_tokens.length; i++) {
        console.log(`[${i}]\t${tokenizer.decode(BOS, prompt_tokens[i])}`);
    }
    console.log()

    let start_time_ms = Date.now();
    let response: string = "";

    let i = 0;
    let next_token: number; // will store the next token in the sequence
    let token = prompt_tokens[i];
    while (i < params.max_steps) {
        let logits = transformer.forward(token, i);

        if (i < prompt_tokens.length - 1) {
            next_token = prompt_tokens[i + 1];
        } else {
            next_token = sampler.sample(logits);
        }

        i += 1;

        // terminating condition: BOS (=1) token delimits sequences
        if (next_token == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        let piece = tokenizer.decode(token, next_token);
        response += piece;
        process.stdout.write(piece);
        
        token = next_token;
    }
    
    console.log("\n\nStats:");
    console.log(`${(1000 * i / (Date.now() - start_time_ms)).toFixed(1)}\ttps`);
    console.log(`${i}\ttokens`);
    console.log(`${params.temperature}\ttemperature`);
    console.log(`${params.topp}\ttopp`);

    return response;
}

let params: InputParameters = {
    checkpoint_path: "data/models/stories15M.bin", // "llama2/llama-2-7b-chat/params.json"
    tokenizer_path: "data/tokenizer.bin",
    temperature: 0.8,
    topp: 0.9,
    max_steps: 512,
};

for (const arg of process.argv.slice(2)) { 
    if (arg.startsWith("--checkpoint_path")) {
        params.checkpoint_path = arg.slice("--checkpoint_path=".length);
    } else if (arg.startsWith("--tokenizer_path")) {
        params.tokenizer_path = arg.slice("--tokenizer_path=".length);
    } else if (arg.startsWith("--temperature")) {
        params.temperature = parseFloat(arg.slice("--temperature=".length));
    } else if (arg.startsWith("--topp")) {
        params.topp = parseFloat(arg.slice("--topp=".length));
    } else if (arg.startsWith("--max_steps")) {
        params.max_steps = parseInt(arg.slice("--max_steps=".length));
    }
}

console.log(`params = ${JSON.stringify(params, null, 2)}`);

import readline from 'readline';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});
rl.question("Prompt: ", (prompt: string) => {
    console.log(`\nResponse:\n\n${generate_response(params, prompt)}\n`);
    rl.close();
});
