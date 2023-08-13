fn accum(a: &mut [f64], b: &[f64]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += b;
    }
}

fn softmax(a: &mut [f64]) {
    let max = a.iter().fold(f64::NAN, |a, b| a.max(*b));
    let mut sum = 0.0;
    for a in a.iter_mut() {
        *a = (*a - max).exp();
        sum += *a;
    }
    for a in a.iter_mut() {
        *a /= sum;
    }
}

fn rmsnorm(o: &mut [f64], x: &[f64], w: &[f64]) {
    let ss = x.iter().fold(0.0, |s, n| s + n * n);
    let rms = ((ss / x.len() as f64) + 1e-5).sqrt();
    // normalize and scale
    for i in 0..o.len() {
        o[i] = x[i] * w[i] / rms;
    }
}

fn matmul(xout: &mut [f64], x: &[f64], w: &[Vec<f64>]) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..w.len() {
        xout[i] = 0.0;
        for j in 0..x.len() {
            xout[i] += w[i][j] * x[j];
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct TransformerConfig {
    dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    vocab_size: usize,
    n_layers: usize,
    seq_len: usize,
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<Vec<f64>>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<Vec<f64>>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<Vec<f64>>, // (layer, dim)
    // weights for matmuls
    wq: Vec<Vec<Vec<f64>>>, // (layer, dim, dim)
    wk: Vec<Vec<Vec<f64>>>, // (layer, dim, dim)
    wv: Vec<Vec<Vec<f64>>>, // (layer, dim, dim)
    wo: Vec<Vec<Vec<f64>>>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<Vec<Vec<f64>>>, // (layer, hidden_dim, dim)
    w2: Vec<Vec<Vec<f64>>>, // (layer, dim, hidden_dim)
    w3: Vec<Vec<Vec<f64>>>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f64>, // (dim, )
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<Vec<f64>>, // (seq_len, head_size/2)
    freq_cis_imag: Vec<Vec<f64>>, // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Vec<Vec<f64>>,
}

struct TransformerState {
    x: Vec<f64>,        // activation at current time stamp (dim,)
    xb: Vec<f64>,       // same, but inside a residual branch (dim,)
    xb2: Vec<f64>,      // an additional buffer just for convenience (dim,)
    hb: Vec<f64>,       // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f64>,      // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f64>,        // query (dim, )
    k: Vec<f64>,        // key (dim, )
    v: Vec<f64>,        // value (dim, )
    att: Vec<Vec<f64>>, // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f64>,   // output logits (vocab_size, )
    // ProbIndex *probindex; // buffer used in top-p sampling
    key_cache: Vec<Vec<Vec<f64>>>,   // (layer, seq_len, dim)
    value_cache: Vec<Vec<Vec<f64>>>, // (layer, seq_len, dim)
}

struct TransformerRunner {
    conf: TransformerConfig,
    weights: TransformerWeights,
    state: TransformerState,
}

impl TransformerRunner {
    pub fn new(conf: &TransformerConfig, weights: TransformerWeights) -> Self {
        let state = TransformerState {
            x: vec![0.0; conf.dim],
            xb: vec![0.0; conf.dim],
            xb2: vec![0.0; conf.dim],
            hb: vec![0.0; conf.hidden_dim],
            hb2: vec![0.0; conf.hidden_dim],
            q: vec![0.0; conf.dim],
            k: vec![0.0; conf.dim],
            v: vec![0.0; conf.dim],
            att: (0..conf.n_heads).map(|_| vec![0.0; conf.dim]).collect(),
            logits: vec![0.0; conf.vocab_size],
            key_cache: (0..conf.n_layers)
                .map(|_| (0..conf.seq_len).map(|_| vec![0.0; conf.dim]).collect())
                .collect(),
            value_cache: (0..conf.n_layers)
                .map(|_| (0..conf.seq_len).map(|_| vec![0.0; conf.dim]).collect())
                .collect(),
        };

        Self {
            conf: *conf,
            weights,
            state,
        }
    }

    pub fn run(&mut self, token: usize, pos: usize) {
        // a few convenience variables
        let s = &mut self.state;
        let w = &self.weights;
        let dim = self.conf.dim;
        let hidden_dim = self.conf.hidden_dim;
        let head_size = self.conf.dim / self.conf.n_heads;

        // copy the token embedding into x
        let content_row = &self.weights.token_embedding_table[token];
        s.x.copy_from_slice(content_row);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        let freq_cis_real_row = &self.weights.freq_cis_real[pos];
        let freq_cis_imag_row = &self.weights.freq_cis_imag[pos];

        // forward all the layers
        for l in 0..self.conf.n_layers {
            // attention rmsnorm
            rmsnorm(&mut s.xb, &s.x, &self.weights.rms_att_weight[l]);
            matmul(&mut s.q, &s.xb, &w.wq[l]);
            matmul(&mut s.k, &s.xb, &w.wk[l]);
            matmul(&mut s.v, &s.xb, &w.wv[l]);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            for i in 0..self.conf.dim {
                let q0 = s.q[i];
                let q1 = s.q[i + 1];
                let k0 = s.k[i];
                let k1 = s.k[i + 1];
                let fcr = freq_cis_real_row[(i % head_size) / 2];
                let fci = freq_cis_imag_row[(i % head_size) / 2];
                s.q[i] = q0 * fcr - q1 * fci;
                s.q[i + 1] = q0 * fci + q1 * fcr;
                s.k[i] = k0 * fcr - k1 * fci;
                s.k[i + 1] = k0 * fci + k1 * fcr;
            }

            // save key,value at this time step (pos) to our kv cache
            let key_cache_row = &s.key_cache[l][pos];
            let value_cache_row = &s.value_cache[l][pos];
            s.k.copy_from_slice(key_cache_row);
            s.v.copy_from_slice(value_cache_row);

            // multihead attention. iterate over all heads
            for h in 0..self.conf.n_heads {
                // get the query vector for this head
                let q = &s.q[h * head_size..h * head_size + head_size];
                //  attention scores for this head
                let att = &mut s.att[h];
                // iterate over all timesteps, including the current one
                for t in 0..(pos + 1) {
                    let k = &s.key_cache[l][t][h * head_size..h * head_size + head_size];
                    // calculate the attention score as the dot product of q and k
                    let mut score = (0..head_size).map(|i| q[i] * k[i]).sum::<f64>();
                    score /= (head_size as f64).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(&mut att[0..pos + 1]);

                // weighted sum of the values, store back into xb
                let xb = &mut s.xb[h * head_size..h * head_size + head_size];
                xb.fill(0.0);
                for t in 0..pos+1 {
                    let v = &s.value_cache[l][t][h * head_size..h * head_size + head_size];
                    // get the attention weight for this timestep
                    let a = att[t];
                    // accumulate the weighted value into xb
                    for i in 0..head_size {
                        xb[i] += a * v[i]
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(&mut s.xb2, &s.xb, &w.wo[l]);

            // residual connection back into x
            accum(&mut s.x, &s.xb2);

            // ffn rmsnorm
            rmsnorm(&mut s.xb, &s.x, &w.rms_ffn_weight[l]);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(&mut s.hb, &s.xb, &w.w1[l]);
            matmul(&mut s.hb2, &s.xb, &w.w3[l]);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for i in 0..hidden_dim {
                s.hb[i] = s.hb[i] * (1.0 / (1.0 + (-s.hb[i]).exp()));
            }

            // elementwise multiply with w3(x)
            for i in 0..hidden_dim {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            matmul(&mut s.xb, &s.hb, &w.w2[l]);

            // residual connection
            accum(&mut s.x, &s.xb);
        }

        // final rmsnorm
        rmsnorm(&mut s.x, &s.x, &w.rms_final_weight);

        // classifier into logits
        matmul(&mut s.logits, &s.x, &w.wcls);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accum() {
        let mut a = [1.0, 2.0];
        let b = [1.2, 3.0];
        accum(&mut a, &b);
        assert_eq!(a[0], 2.2);
        assert_eq!(a[1], 5.0);
    }

    #[test]
    fn test_matmul() {
        let w = [vec![1.0, 2.0, 3.0], vec![1.0, 5.0, 1.0]]; // (2,3)
        let x = [2.0, 4.0, 8.0]; // (3,)
        let out: &mut [f64; 2] = &mut [0.0, 0.0]; // (2, )
        matmul(out, &x, &w);
        assert_eq!(out[0], 34.0);
        assert_eq!(out[1], 30.0);
    }
}