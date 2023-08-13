use memmap::MmapOptions;
use memmap::Mmap;
use std::fs::File;
use std::ops::Index;
use std::result::Result;

fn accum(a: &mut [f32], b: &[f32]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += b;
    }
}

fn softmax(a: &mut [f32]) {
    let max = a.iter().fold(f32::NAN, |a, b| a.max(*b));
    let mut sum = 0.0;
    for a in a.iter_mut() {
        *a = (*a - max).exp();
        sum += *a;
    }
    for a in a.iter_mut() {
        *a /= sum;
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], w: &[f32]) {
    let ss = x.iter().fold(0.0, |s, n| s + n * n);
    let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
    // normalize and scale
    for i in 0..o.len() {
        o[i] = x[i] * w[i] / rms;
    }
}

fn rmsnorm2(x: &mut [f32], w: &[f32]) {
    let ss = x.iter().fold(0.0, |s, n| s + n * n);
    let rms = ((ss / x.len() as f32) + 1e-5).sqrt();
    // normalize and scale
    for i in 0..x.len() {
        x[i] = x[i] * w[i] / rms;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: impl Index<usize, Output=[f32]>) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..xout.len() {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TransformerErrorKind {
    InvalidConfig,
    InvalidWeights,
}

#[derive(Debug)]
struct TransformerError {
    kind: TransformerErrorKind,
    message: String,
    source: Option<Box<dyn std::error::Error>>,
}

impl std::fmt::Display for TransformerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message);
        if let Some(source) = &self.source {
            write!(f, ": {}", source);
        }
        Ok(())
    }
}

impl std::error::Error for TransformerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| &**e)
    }
}

#[derive(Debug, Default)]
struct Tensor2D<'a> {
    data: &'a [f32],
    shape: (usize, usize),
}

impl<'a> Tensor2D<'a> {
    pub fn new(data: &'a [f32], shape: (usize, usize)) -> Self {
        Self { data: &data[0..shape.0 * shape.1], shape }
    }

    pub fn row_slice_at(&self, idx: usize) -> &'a [f32] {
        let start = idx * self.shape.1;
        &self.data[start..start + self.shape.1]
    }
}

impl std::ops::Index<usize> for &Tensor2D<'_> {
    type Output = [f32];

    fn index(&self, idx: usize) -> &Self::Output {
        self.row_slice_at(idx)
    }
}

#[derive(Debug, Default)]
struct Tensor3D<'a> {
    data: &'a [f32],
    shape: (usize, usize, usize),
}

impl<'a> Tensor3D<'a> {
    pub fn new(data: &'a [f32], shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self { data: &data[0..size], shape }
    }

    pub fn tensor2d_at(&self, idx: usize) -> Tensor2D<'a> {
        let start = idx * self.shape.1 * self.shape.2;
        Tensor2D::new(&self.data[start..start + self.shape.1 * self.shape.2], (self.shape.1, self.shape.2))
    }
}

#[derive(Default)]
struct TransformerWeights<'a> {
    // token embedding table
    token_embedding_table: Tensor2D<'a>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Tensor2D<'a>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Tensor2D<'a>, // (layer, dim)
    // weights for matmuls
    wq: Tensor3D<'a>, // (layer, dim, dim)
    wk: Tensor3D<'a>, // (layer, dim, dim)
    wv: Tensor3D<'a>, // (layer, dim, dim)
    wo: Tensor3D<'a>, // (layer, dim, dim)
    // weights for ffn
    w1: Tensor3D<'a>, // (layer, hidden_dim, dim)
    w2: Tensor3D<'a>, // (layer, dim, hidden_dim)
    w3: Tensor3D<'a>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: &'a [f32], // (dim, )
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Tensor2D<'a>, // (seq_len, head_size/2)
    freq_cis_imag: Tensor2D<'a>, // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Tensor2D<'a>, // (vocab_size, dim)
}

impl<'a> TransformerWeights<'a> {
    fn init_from_checkpoint(&mut self, data: &[u8], conf: &TransformerConfig) -> Self {
        let weights = TransformerWeights::default();
        weights
    }
}

struct TransformerState {
    x: Vec<f32>,        // activation at current time stamp (dim,)
    xb: Vec<f32>,       // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,      // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,       // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,      // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,        // query (dim, )
    k: Vec<f32>,        // key (dim, )
    v: Vec<f32>,        // value (dim, )
    att: Vec<Vec<f32>>, // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>,   // output logits (vocab_size, )
    // ProbIndex *probindex; // buffer used in top-p sampling
    key_cache: Vec<Vec<Vec<f32>>>,   // (layer, seq_len, dim)
    value_cache: Vec<Vec<Vec<f32>>>, // (layer, seq_len, dim)
}

struct TransformerRunner {
    conf: TransformerConfig,
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
            state,
        }
    }

    pub fn run(&mut self, w: &TransformerWeights, token: usize, pos: usize) {
        // a few convenience variables
        let s = &mut self.state;
        let hidden_dim = self.conf.hidden_dim;
        let head_size = self.conf.dim / self.conf.n_heads;

        // copy the token embedding into x
        let content_row = w.token_embedding_table.row_slice_at(token);
        s.x.copy_from_slice(content_row);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        let freq_cis_real_row = w.freq_cis_real.row_slice_at(pos);
        let freq_cis_imag_row = w.freq_cis_imag.row_slice_at(pos);

        // forward all the layers
        for l in 0..self.conf.n_layers {
            // attention rmsnorm
            rmsnorm(&mut s.xb, &s.x, w.rms_att_weight.row_slice_at(l));
            matmul(&mut s.q, &s.xb, &w.wq.tensor2d_at(l));
            matmul(&mut s.k, &s.xb, &w.wk.tensor2d_at(l));
            matmul(&mut s.v, &s.xb, &w.wv.tensor2d_at(l));

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
            let key_cache_row = &mut s.key_cache[l][pos];
            let value_cache_row = &mut s.value_cache[l][pos];
            key_cache_row.copy_from_slice(&s.k);
            value_cache_row.copy_from_slice(&s.v);

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
                    let mut score = (0..head_size).map(|i| q[i] * k[i]).sum::<f32>();
                    score /= (head_size as f32).sqrt();
                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(&mut att[0..pos + 1]);

                // weighted sum of the values, store back into xb
                let xb = &mut s.xb[h * head_size..h * head_size + head_size];
                xb.fill(0.0);
                for t in 0..pos + 1 {
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
            matmul(&mut s.xb2, &s.xb, &w.wo.tensor2d_at(l));

            // residual connection back into x
            accum(&mut s.x, &s.xb2);

            // ffn rmsnorm
            rmsnorm(&mut s.xb, &s.x, w.rms_ffn_weight.row_slice_at(l));

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(&mut s.hb, &s.xb, &w.w1.tensor2d_at(l));
            matmul(&mut s.hb2, &s.xb, &w.w3.tensor2d_at(l));

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for i in 0..hidden_dim {
                s.hb[i] = s.hb[i] * (1.0 / (1.0 + (-s.hb[i]).exp()));
            }

            // elementwise multiply with w3(x)
            for i in 0..hidden_dim {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            matmul(&mut s.xb, &s.hb, &w.w2.tensor2d_at(l));

            // residual connection
            accum(&mut s.x, &s.xb);
        }

        // final rmsnorm
        rmsnorm2(&mut s.x, &w.rms_final_weight);

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
        let wvec = vec![1.0, 2.0, 3.0, 1.0, 5.0, 1.0]; 
        let w = Tensor2D::new(&wvec, (2, 3)); // (2,3)
        let x = [2.0, 4.0, 8.0]; // (3,)
        let out: &mut [f32; 2] = &mut [0.0, 0.0]; // (2, )
        matmul(out, &x, &w);
        assert_eq!(out[0], 34.0);
        assert_eq!(out[1], 30.0);
    }
}
