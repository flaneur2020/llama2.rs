use memmap::MmapOptions;
use memmap::Mmap;
use std::mem;
use std::fs::File;
use std::ops::Index;
use std::ops::Range;
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

fn matmul(xout: &mut [f32], x: &[f32], w: impl Index<(usize, usize), Output=f32>) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..xout.len() {
        xout[i] = 0.0;
        for j in 0..x.len() {
            xout[i] += w[(i,j)] * x[j];
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Llama2Config {
    dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    vocab_size: usize,
    n_layers: usize,
    seq_len: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Llama2ErrorKind {
    InvalidConfig,
    InvalidWeights,
    InvalidData,
    InvalidIndex,
}

#[derive(Debug)]
struct Llama2Error {
    kind: Llama2ErrorKind,
    message: String,
    source: Option<Box<dyn std::error::Error>>,
}

impl std::fmt::Display for Llama2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message);
        if let Some(source) = &self.source {
            write!(f, ": {}", source);
        }
        Ok(())
    }
}

impl std::error::Error for Llama2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| &**e)
    }
}

#[derive(Debug, Default)]
struct Tensor<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> Result<Self, Llama2Error> {
        if data.len() != shape.iter().product() {
            return Err(Llama2Error {
                kind: Llama2ErrorKind::InvalidData,
                message: format!("invalid shape {:?} for data of length {}", shape, data.len()),
                source: None,
            });
        }

        let tensor = Self { data, shape };
        Ok(tensor)
    }

    pub fn flat(&self) -> &[f32] {
        self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn at(&self, idx: usize) -> Result<Self, Llama2Error> {
        if idx >= self.shape[0] {
            return Err(Llama2Error {
                kind: Llama2ErrorKind::InvalidIndex,
                message: format!("index {} out of bounds for shape {:?}", idx, self.shape),
                source: None,
            });
        }
        if self.shape.len() == 1 {
            let data = &self.data[idx..idx + 1];
            return Self::new(data, vec![1]);
        }
        let chunk_size: usize = self.shape[1..].iter().product();
        let start = idx * chunk_size;
        Self::new(&self.data[start..start + chunk_size], self.shape[1..].to_vec())
    }
}

impl Index<(usize, usize)> for &Tensor<'_> {
    type Output = f32;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[idx.0 * self.shape[1] + idx.1]
    }
}

struct Llama2WeightsReader<'a> {
    buf: &'a [u8],
}

impl<'a> Llama2WeightsReader<'a> {
    fn read_tensor(&mut self, shape: &Vec<usize>) -> Result<Tensor<'a>, Llama2Error> {
        let elems = shape.iter().product::<usize>();
        let data = &self.buf[..elems*4];
        let size_f32 = mem::size_of::<f32>();
        let size_u8 = mem::size_of::<u8>();
        let data_f32: &[f32] = unsafe {
            assert!(data.len() % size_f32 == 0);
            let ptr = data.as_ptr();
            mem::transmute(std::slice::from_raw_parts(ptr, data.len() / (size_f32 / size_u8)))
        };
        self.buf = &self.buf[elems*4..];
        return Tensor::new(data_f32, shape.to_vec());
    }
}

#[derive(Default)]
struct Llama2Weights<'a> {
    // token embedding table
    token_embedding_table: Tensor<'a>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Tensor<'a>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Tensor<'a>, // (layer, dim)
    // weights for matmuls
    wq: Tensor<'a>, // (layer, dim, dim)
    wk: Tensor<'a>, // (layer, dim, dim)
    wv: Tensor<'a>, // (layer, dim, dim)
    wo: Tensor<'a>, // (layer, dim, dim)
    // weights for ffn
    w1: Tensor<'a>, // (layer, hidden_dim, dim)
    w2: Tensor<'a>, // (layer, dim, hidden_dim)
    w3: Tensor<'a>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Tensor<'a>, // (dim, )
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Tensor<'a>, // (seq_len, head_size/2)
    freq_cis_imag: Tensor<'a>, // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Tensor<'a>, // (vocab_size, dim)
}

impl<'a> Llama2Weights<'a> {
    fn init_from_checkpoint(&mut self, data: &[u8], conf: &Llama2Config) -> Result<Self, Llama2Error> {
        let mut weights = Llama2Weights::default();
        let head_size = conf.dim / conf.n_heads;
        let token_embedding_table = Tensor::new(data, vec![conf.vocab_size, conf.dim])?;
        weights
    }
}

struct Llama2State {
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

struct Llama2Runner {
    conf: Llama2Config,
    state: Llama2State,
}

impl Llama2Runner {
    pub fn new(conf: &Llama2Config, weights: Llama2Weights) -> Self {
        let state = Llama2State {
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

    pub fn run(&mut self, w: &Llama2Weights, token: usize, pos: usize) -> Result<(), Llama2Error>{
        // a few convenience variables
        let s = &mut self.state;
        let hidden_dim = self.conf.hidden_dim;
        let head_size = self.conf.dim / self.conf.n_heads;

        // copy the token embedding into x
        let content_row = w.token_embedding_table.at(token)?;
        s.x.copy_from_slice(content_row.flat());

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        let freq_cis_real_row = w.freq_cis_real.at(pos)?;
        let freq_cis_imag_row = w.freq_cis_imag.at(pos)?;

        // forward all the layers
        for l in 0..self.conf.n_layers {
            // attention rmsnorm
            rmsnorm(&mut s.xb, &s.x, w.rms_att_weight.at(l)?.flat());
            matmul(&mut s.q, &s.xb, &w.wq.at(l)?);
            matmul(&mut s.k, &s.xb, &w.wk.at(l)?);
            matmul(&mut s.v, &s.xb, &w.wv.at(l)?);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            for i in 0..self.conf.dim {
                let q0 = s.q[i];
                let q1 = s.q[i + 1];
                let k0 = s.k[i];
                let k1 = s.k[i + 1];
                let fcr = freq_cis_real_row.flat()[(i % head_size) / 2];
                let fci = freq_cis_imag_row.flat()[(i % head_size) / 2];
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
            matmul(&mut s.xb2, &s.xb, &w.wo.at(l)?);

            // residual connection back into x
            accum(&mut s.x, &s.xb2);

            // ffn rmsnorm
            rmsnorm(&mut s.xb, &s.x, w.rms_ffn_weight.at(l)?.flat());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(&mut s.hb, &s.xb, &w.w1.at(l)?);
            matmul(&mut s.hb2, &s.xb, &w.w3.at(l)?);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for i in 0..hidden_dim {
                s.hb[i] = s.hb[i] * (1.0 / (1.0 + (-s.hb[i]).exp()));
            }

            // elementwise multiply with w3(x)
            for i in 0..hidden_dim {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            matmul(&mut s.xb, &s.hb, &w.w2.at(l)?);

            // residual connection
            accum(&mut s.x, &s.xb);
        }

        // final rmsnorm
        rmsnorm2(&mut s.x, &w.rms_final_weight.flat());

        // classifier into logits
        matmul(&mut s.logits, &s.x, &w.wcls);

        Ok(())
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
        let w = Tensor::new(&wvec, vec![2, 3]).unwrap(); // (2,3)
        let x = [2.0, 4.0, 8.0]; // (3,)
        let out: &mut [f32; 2] = &mut [0.0, 0.0]; // (2, )
        matmul(out, &x, &w);
        assert_eq!(out[0], 34.0);
        assert_eq!(out[1], 30.0);
    }

    #[test]
    fn test_tensor() -> Result<(), Llama2Error> {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3]).unwrap();
        assert_eq!(t.at(0)?.flat().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(t.at(1)?.flat().to_vec(), vec![4.0, 5.0, 6.0]);

        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::new(&v, vec![2, 3, 1]).unwrap();
        assert_eq!(t.at(0)?.flat().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(t.at(1)?.flat().to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(t.at(0)?.at(0)?.flat().to_vec(), vec![1.0]);
        assert_eq!(t.at(0)?.at(1)?.flat().to_vec(), vec![2.0]);
        assert_eq!(t.at(0)?.at(2)?.flat().to_vec(), vec![3.0]);
        assert_eq!(t.at(1)?.at(0)?.flat().to_vec(), vec![4.0]);
        assert_eq!(t.at(1)?.shape().to_vec(), vec![3, 1]);

        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let t = Tensor::new(&v, vec![2, 3, 2, 1]).unwrap();
        assert_eq!(t.at(0)?.flat().to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.at(1)?.flat().to_vec(), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        Ok(())
    }


    fn test_stories() -> Result<(), Llama2Error> {
        Ok(())
    }
}
