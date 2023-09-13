use memmap::Mmap;
use memmap::MmapOptions;
use rand::Rng;
use std::collections::HashMap;
use std::fs::File;
use std::mem;
use std::ops::AddAssign;
use std::ops::Index;
use std::time::Duration;
use std::time::Instant;
use std::vec;

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

fn matmul(xout: &mut [f32], x: &[f32], w: impl Index<(usize, usize), Output = f32>) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..xout.len() {
        xout[i] = 0.0;
        for j in 0..x.len() {
            xout[i] += w[(i, j)] * x[j];
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Llama2ErrorKind {
    IOError,
    BadInput,
    Unexpected,
    TensorError,
}

#[derive(Debug)]
pub struct Llama2Error {
    kind: Llama2ErrorKind,
    message: String,
    source: Option<Box<dyn std::error::Error>>,
}

impl std::fmt::Display for Llama2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.kind)?;
        write!(f, "{}", self.message)?;
        if let Some(source) = &self.source {
            write!(f, ": {}", source)?;
        }
        Ok(())
    }
}

impl std::error::Error for Llama2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref()
    }
}

pub type Result<T> = std::result::Result<T, Llama2Error>;

#[derive(Debug, Default, Clone)]
struct Tensor<'a> {
    data: &'a [f32],
    shape: Vec<usize>,
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> Result<Self> {
        if data.len() != shape.iter().product() {
            return Err(Llama2Error {
                kind: Llama2ErrorKind::TensorError,
                message: format!(
                    "invalid shape {:?} for data of length {}",
                    shape,
                    data.len()
                ),
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

    pub fn at(&self, idx: usize) -> Result<Self> {
        if idx >= self.shape[0] {
            return Err(Llama2Error {
                kind: Llama2ErrorKind::TensorError,
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
        Self::new(
            &self.data[start..start + chunk_size],
            self.shape[1..].to_vec(),
        )
    }
}

impl Index<(usize, usize)> for &Tensor<'_> {
    type Output = f32;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[idx.0 * self.shape[1] + idx.1]
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Llama2Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

#[derive(Default)]
pub struct Llama2Weights<'a> {
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

pub struct Llama2CheckpointReader<'a> {
    buf: &'a [u8],
    total_bytes: usize,
}

impl<'a> Llama2CheckpointReader<'a> {
    #[allow(dead_code)]
    fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    fn read_i32(&mut self) -> Result<i32> {
        if self.buf.len() < 4 {
            return Err(Llama2Error {
                kind: Llama2ErrorKind::IOError,
                message: format!("expected 4 bytes, found {}", self.buf.len()),
                source: None,
            });
        }
        let (int_bytes, rest) = self.buf.split_at(4);
        self.total_bytes += 4;
        self.buf = rest;
        Ok(i32::from_le_bytes([
            int_bytes[0],
            int_bytes[1],
            int_bytes[2],
            int_bytes[3],
        ]))
    }

    fn read_tensor(&mut self, shape: Vec<usize>) -> Result<Tensor<'a>> {
        let elems = shape.iter().product::<usize>();
        let size_f32 = mem::size_of::<f32>();
        let data = &self.buf[..elems * size_f32];
        let data_f32: &[f32] = unsafe {
            assert!(data.len() % size_f32 == 0);
            let ptr = data.as_ptr();
            mem::transmute(std::slice::from_raw_parts(ptr, data.len() / size_f32))
        };
        self.total_bytes += elems * size_f32;
        self.buf = &self.buf[elems * size_f32..];
        return Tensor::new(data_f32, shape);
    }
}

pub struct Llama2CheckpointLoader {
    mmap: Mmap,
}

impl Llama2CheckpointLoader {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path).map_err(|e| Llama2Error {
            kind: Llama2ErrorKind::IOError,
            message: format!("failed to open file {}: {}", path, e),
            source: Some(Box::new(e)),
        })?;
        let mmap = unsafe {
            MmapOptions::new().map(&file).map_err(|e| Llama2Error {
                kind: Llama2ErrorKind::IOError,
                message: format!("failed to mmap file {}: {}", path, e),
                source: Some(Box::new(e)),
            })?
        };
        Ok(Self { mmap })
    }

    pub fn load(&self) -> Result<(Llama2Config, Llama2Weights)> {
        let mut r = self.reader();
        let conf = Self::load_config(&mut r)?;
        let weights = Self::load_weights(&mut r, &conf)?;
        Ok((conf, weights))
    }

    pub(crate) fn reader(&self) -> Llama2CheckpointReader {
        Llama2CheckpointReader {
            buf: &self.mmap[..],
            total_bytes: 0,
        }
    }

    pub(crate) fn load_config(r: &mut Llama2CheckpointReader<'_>) -> Result<Llama2Config> {
        let dim = r.read_i32()? as usize;
        let hidden_dim = r.read_i32()? as usize;
        let n_layers = r.read_i32()? as usize;
        let n_heads = r.read_i32()? as usize;
        let n_kv_heads = r.read_i32()? as usize;
        let vocab_size = r.read_i32()? as usize;
        let seq_len = r.read_i32()? as usize;
        Ok(Llama2Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }

    pub(crate) fn load_weights<'a>(
        r: &mut Llama2CheckpointReader<'a>,
        conf: &Llama2Config,
    ) -> Result<Llama2Weights<'a>> {
        let shared_weights = conf.vocab_size > 0;
        let mut weights = Llama2Weights::default();
        let head_size = conf.dim / conf.n_heads;
        weights.token_embedding_table = r.read_tensor(vec![conf.vocab_size, conf.dim])?;
        weights.rms_att_weight = r.read_tensor(vec![conf.n_layers, conf.dim])?;
        weights.wq = r.read_tensor(vec![conf.n_layers, conf.dim, conf.n_heads * head_size])?;
        weights.wk = r.read_tensor(vec![conf.n_layers, conf.dim, conf.n_kv_heads * head_size])?;
        weights.wv = r.read_tensor(vec![conf.n_layers, conf.dim, conf.n_kv_heads * head_size])?;
        weights.wo = r.read_tensor(vec![conf.n_layers, conf.n_heads * head_size, conf.dim])?;
        weights.rms_ffn_weight = r.read_tensor(vec![conf.n_layers, conf.dim])?;
        weights.w1 = r.read_tensor(vec![conf.n_layers, conf.hidden_dim, conf.dim])?;
        weights.w2 = r.read_tensor(vec![conf.n_layers, conf.dim, conf.hidden_dim])?;
        weights.w3 = r.read_tensor(vec![conf.n_layers, conf.hidden_dim, conf.dim])?;
        weights.rms_final_weight = r.read_tensor(vec![conf.dim])?;
        weights.freq_cis_real = r.read_tensor(vec![conf.seq_len * head_size / 2])?; // skip what used to be freq_cis_real (for RoPE)
        weights.freq_cis_imag = r.read_tensor(vec![conf.seq_len * head_size / 2])?; // skip what used to be freq_cis_imag (for RoPE)
        weights.wcls = if shared_weights {
            weights.token_embedding_table.clone()
        } else {
            r.read_tensor(vec![conf.vocab_size, conf.dim])?
        };
        Ok(weights)
    }
}

pub struct Llama2Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    max_token_length: usize,
    byte_pieces: [u8; 256],
    vocab_index: HashMap<String, usize>,
}

impl Llama2Tokenizer {
    pub fn decode(&self, prev_token: usize, token: usize) -> Result<String> {
        let mut piece: &[u8] = self.vocab[token].as_bytes();
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token == 1 && piece[0] == b' ' {
            piece = &piece[1..];
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        if piece.starts_with(b"<0x") && piece[piece.len() - 1] == b'>' {
            let s = String::from_utf8_lossy(&piece[1..piece.len() - 1]);
            let s = s.trim_start_matches("0x");
            if let Ok(byte) = u8::from_str_radix(s, 16) {
                piece = &self.byte_pieces[(byte as usize)..(byte as usize) + 1]
            }
        }
        Ok(String::from_utf8(piece.to_vec()).unwrap())
    }

    #[allow(dead_code)]
    pub fn decode_string(&self, tokens: &[usize]) -> Result<String> {
        let mut result = String::new();
        let mut prev_token = 0;
        for token in tokens {
            let piece = self.decode(prev_token, *token)?;
            result.push_str(&piece);
            prev_token = *token;
        }
        Ok(result)
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<usize>> {
        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        let mut str_buf = String::with_capacity(self.max_token_length * 2 + 1 + 2);
        let mut tokens: Vec<usize> = vec![];

        if bos {
            tokens.push(1);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if !text.starts_with('\u{0}') {
            let dummy_prefix = self.vocab_index.get(" ").unwrap();
            tokens.push(*dummy_prefix);
        }

        let chars = text.chars();
        for ch in chars {
            str_buf.clear();
            str_buf.push(ch);
            if let Some(tok) = self.vocab_index.get(&str_buf) {
                // we found this codepoint in vocab, add it as a token
                tokens.push(*tok);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for byte in str_buf.bytes() {
                    tokens.push(byte as usize + 3);
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx: Option<usize> = None;
            let mut best_token: Option<usize> = None;
            let mut i = 0;

            while i < (tokens.len() - 1) {
                str_buf.clear();
                str_buf.push_str(&self.vocab[tokens[i]]);
                str_buf.push_str(&self.vocab[tokens[i + 1]]);
                if let Some(tok) = self.vocab_index.get(&str_buf) {
                    let new_score = self.vocab_scores[*tok];
                    if new_score > best_score {
                        best_score = new_score;
                        best_idx = Some(i);
                        best_token = Some(*tok);
                    }
                }
                i += 1;
            }

            if let Some(idx) = best_idx {
                tokens[idx] = best_token.unwrap();
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(2);
        }

        Ok(tokens)
    }
}

pub struct Llama2TokenizerLoader {
    r: Box<dyn std::io::Read>,
}

impl Llama2TokenizerLoader {
    pub fn new(path: &str) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|e| Llama2Error {
            kind: Llama2ErrorKind::IOError,
            message: format!("failed to open file {}: {}", path, e),
            source: Some(Box::new(e)),
        })?;
        let f = std::io::BufReader::new(f);
        Ok(Self { r: Box::new(f) })
    }

    pub fn load(&mut self, vocab_size: usize) -> Result<Llama2Tokenizer> {
        let mut vocabs = vec![String::new(); vocab_size];
        let mut vocab_scores = vec![0.0; vocab_size];
        let mut byte_pieces = [0u8; 256];

        for i in 0..256 {
            byte_pieces[i] = i as u8;
        }

        let max_token_length = self.read_i32()? as usize;
        for i in 0..vocab_size {
            vocab_scores[i] = self.read_f32()?;
            let len = self.read_i32()?;
            vocabs[i] = self.read_string(len as usize)?;
        }

        let vocab_index = vocabs
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        Ok(Llama2Tokenizer {
            vocab: vocabs,
            vocab_scores,
            vocab_index,
            max_token_length,
            byte_pieces,
        })
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.r.read_exact(&mut buf).map_err(|e| Llama2Error {
            kind: Llama2ErrorKind::IOError,
            message: format!("failed to read i32: {}", e),
            source: Some(Box::new(e)),
        })?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.r.read_exact(&mut buf).map_err(|e| Llama2Error {
            kind: Llama2ErrorKind::IOError,
            message: format!("failed to read f32: {}", e),
            source: Some(Box::new(e)),
        })?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_bytes(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.r.read_exact(&mut buf).map_err(|e| Llama2Error {
            kind: Llama2ErrorKind::IOError,
            message: format!("failed to read bytes: {}", e),
            source: Some(Box::new(e)),
        })?;
        Ok(buf)
    }

    fn read_string(&mut self, len: usize) -> Result<String> {
        let buf = self.read_bytes(len)?;
        String::from_utf8(buf).map_err(|e| Llama2Error {
            kind: Llama2ErrorKind::IOError,
            message: format!("failed to read string: {}", e),
            source: Some(Box::new(e)),
        })
    }
}

pub struct Llama2Sampler {
    prob_index: Vec<(f32, usize)>,
    temperature: f32,
    topp: f32,
}

impl Llama2Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32) -> Self {
        Self {
            prob_index: vec![(0.0, 0); vocab_size],
            temperature,
            topp,
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> Result<usize> {
        if self.temperature == 0.0 {
            return Self::sample_argmax(logits);
        }

        // apply the temperature to the logits. the lower the temperature,
        // the more deterministic the sampling.
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits);

        // flip a (float) coin (this is our source of entropy for sampling)
        let mut rng = rand::thread_rng();
        let coin: f32 = rng.gen_range(0.0..1.0);

        // we sample from this distribution to get the next token
        if self.topp <= 0_f32 || self.topp >= 1.0_f32 {
            // simply sample from the predicted probability distribution
            Self::sample_multi(logits, coin);
        }

        Self::sample_topp(logits, self.topp, &mut self.prob_index, coin)
    }

    pub fn sample_multi(probs: &[f32], coin: f32) -> usize {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        let mut cdf = 0_f32;
        for (i, p) in probs.iter().enumerate() {
            cdf += p;
            if cdf > coin {
                return i;
            }
        }
        probs.len() - 1 // in case of rounding errors
    }

    pub fn sample_topp(
        probs: &[f32],
        topp: f32,
        prob_index: &mut [(f32, usize)],
        coin: f32,
    ) -> Result<usize> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        let cutoff = (1.0_f32 - topp) / (probs.len() - 1) as f32;
        let mut n0 = 0;
        for i in 0..probs.len() {
            if probs[i] >= cutoff {
                prob_index[n0] = (probs[i], i);
                n0 += 1;
            }
        }
        prob_index[..n0].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // truncate the list where cumulative probability exceeds topp
        let mut cumulative_prob = 0_f32;
        let mut last_idx = n0 - 1; // in case of rounding errors consider all elements
        for i in 0..n0 {
            cumulative_prob += prob_index[i].0;
            if cumulative_prob > topp {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0_f32;
        for i in 0..=last_idx {
            cdf += prob_index[i].0;
            if cdf > r {
                return Ok(prob_index[i].1);
            }
        }
        Ok(prob_index[last_idx].1) // in case of rounding errors
    }

    pub fn sample_argmax(probs: &[f32]) -> Result<usize> {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .ok_or_else(|| Llama2Error {
                kind: Llama2ErrorKind::Unexpected,
                message: format!("failed to sample from logits"),
                source: None,
            })
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

pub struct Llama2Runner<'a> {
    conf: Llama2Config,
    state: Llama2State,
    weights: Llama2Weights<'a>,
    tokenizer: Llama2Tokenizer,
}

impl<'a> Llama2Runner<'a> {
    pub fn new(
        conf: &Llama2Config,
        weights: Llama2Weights<'a>,
        tokenizer: Llama2Tokenizer,
    ) -> Self {
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
            weights,
            tokenizer,
        }
    }

    pub fn generate(
        &'a mut self,
        prompt: &str,
        steps: usize,
        sampler: &'a mut Llama2Sampler,
    ) -> Result<Llama2RunnerOutputGenerator<'a>> {
        Llama2RunnerOutputGenerator::new(self, sampler, prompt, steps, self.conf.seq_len)
    }

    fn head_size(&self) -> usize {
        self.conf.dim / self.conf.n_heads
    }

    fn kv_dim(&self) -> usize {
        (self.conf.dim * self.conf.n_kv_heads) / self.conf.n_heads
    }

    fn rope(&mut self, l: usize, pos: usize) {
        for i in (0..self.conf.dim).step_by(2) {
            let head_dim = i % self.head_size();
            let freq = 1.0 / 10000_f32.powf(head_dim as f32 / self.head_size() as f32);
            let val = pos as f32 * freq;
            let fcr = val.cos();
            let fci = val.sin();
            let rotn = if i < self.kv_dim() { 2 } else { 1 }; // how many vectors? 2 = q & k, 1 = q only
            for v in 0..rotn {
                let vec = if v == 0 {
                    &mut self.state.q
                } else {
                    &mut self.state.k
                };
                let v0 = vec[i];
                let v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }

    fn attention_head(&mut self, l: usize, h: usize, pos: usize) {
        let head_size = self.head_size();

        // get the query vector for this head
        let q = &self.state.q[h * head_size..h * head_size + head_size];
        //  attention scores for this head
        let att = &mut self.state.att[h];
        // iterate over all timesteps, including the current one
        for t in 0..(pos + 1) {
            let k = &self.state.key_cache[l][t][h * head_size..h * head_size + head_size];
            // calculate the attention score as the dot product of q and k
            let mut score = (0..head_size).map(|i| q[i] * k[i]).sum::<f32>();
            score /= (head_size as f32).sqrt();
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(&mut att[0..pos + 1]);

        // weighted sum of the values, store back into xb
        let xb = &mut self.state.xb[h * head_size..h * head_size + head_size];
        xb.fill(0.0);
        for t in 0..pos + 1 {
            let v = &self.state.value_cache[l][t][h * head_size..h * head_size + head_size];
            // get the attention weight for this timestep
            let a = att[t];
            // accumulate the weighted value into xb
            for i in 0..head_size {
                xb[i] += a * v[i]
            }
        }
    }

    fn ffn(&mut self, l: usize) -> Result<()> {
        let hidden_dim = self.conf.hidden_dim;
        // final matmul to get the output of the attention
        matmul(&mut self.state.xb2, &self.state.xb, &self.weights.wo.at(l)?);

        // residual connection back into x
        accum(&mut self.state.x, &self.state.xb2);

        // ffn rmsnorm
        rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            self.weights.rms_ffn_weight.at(l)?.flat(),
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(&mut self.state.hb, &self.state.xb, &self.weights.w1.at(l)?);
        matmul(&mut self.state.hb2, &self.state.xb, &self.weights.w3.at(l)?);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            self.state.hb[i] = self.state.hb[i] * (1.0 / (1.0 + (-self.state.hb[i]).exp()));
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            self.state.hb[i] *= self.state.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(&mut self.state.xb, &self.state.hb, &self.weights.w2.at(l)?);

        // residual connection
        accum(&mut self.state.x, &self.state.xb);

        Ok(())
    }

    fn matmul_qkv(&mut self, l: usize) -> Result<()> {
        // attention rmsnorm
        rmsnorm(
            &mut self.state.xb,
            &self.state.x,
            self.weights.rms_att_weight.at(l)?.flat(),
        );
        matmul(&mut self.state.q, &self.state.xb, &self.weights.wq.at(l)?);
        matmul(&mut self.state.k, &self.state.xb, &self.weights.wk.at(l)?);
        matmul(&mut self.state.v, &self.state.xb, &self.weights.wv.at(l)?);
        Ok(())
    }

    fn kv_cache(&mut self, l: usize, pos: usize) {
        let kv_dim = self.kv_dim();
        let key_cache_row = &mut self.state.key_cache[l][pos];
        let value_cache_row = &mut self.state.value_cache[l][pos];
        key_cache_row.copy_from_slice(&self.state.k[0..kv_dim]);
        value_cache_row.copy_from_slice(&self.state.v[0..kv_dim]);
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> Result<&mut [f32]> {
        // copy the token embedding into x
        let content_row = self.weights.token_embedding_table.at(token)?;
        self.state.x.copy_from_slice(content_row.flat());

        // forward all the layers
        for l in 0..self.conf.n_layers {
            self.matmul_qkv(l)?;

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            self.rope(l, pos);

            // save key,value at this time step (pos) to our kv cache
            self.kv_cache(l, pos);

            // multihead attention. iterate over all heads
            for h in 0..self.conf.n_heads {
                self.attention_head(l, h, pos);
            }
            self.ffn(l)?;
        }

        // final rmsnorm
        rmsnorm2(&mut self.state.x, self.weights.rms_final_weight.flat());

        // classifier into logits
        matmul(&mut self.state.logits, &self.state.x, &self.weights.wcls);

        Ok(&mut self.state.logits)
    }
}

pub struct Llama2RunnerOutputGenerator<'a> {
    pos: usize,
    steps: usize,
    seq_len: usize,
    prompt_tokens: Vec<usize>,
    token: usize,
    sampler: &'a mut Llama2Sampler,
    runner: &'a mut Llama2Runner<'a>,
    total_time: Duration,
}

impl<'a> Llama2RunnerOutputGenerator<'a> {
    fn new(
        runner: &'a mut Llama2Runner<'a>,
        sampler: &'a mut Llama2Sampler,
        prompt: &str,
        steps: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let prompt_tokens = runner.tokenizer.encode(prompt, true, false)?;
        if prompt_tokens.is_empty() {
            return Err(Llama2Error {
                kind: Llama2ErrorKind::BadInput,
                message: "something is wrong, expected at least 1 prompt token".to_string(),
                source: None,
            });
        }

        let token = prompt_tokens[0];
        Ok(Self {
            pos: 0,
            steps,
            token,
            prompt_tokens,
            sampler,
            runner,
            seq_len,
            total_time: Duration::new(0, 0),
        })
    }

    pub fn average_tokens_per_seconds(&self) -> f32 {
        let total_time = self.total_time.as_secs_f32();
        self.pos as f32 / total_time
    }

    fn forward_next(&mut self) -> Result<Option<String>> {
        if self.pos >= self.steps + self.prompt_tokens.len() {
            return Ok(None);
        }
        if self.pos >= self.seq_len {
            return Ok(None);
        }

        // forward the transformer to get logits for the next token
        let start_time = Instant::now();
        let logits = self.runner.forward(self.token, self.pos)?;

        // advance the state state machine
        let (next_token, is_prompt) = if self.pos < self.prompt_tokens.len() - 1 {
            // if we are still processing the input prompt, force the next prompt token
            (self.prompt_tokens[self.pos + 1], true)
        } else {
            // otherwise sample the next token from the logits
            let token = self.sampler.sample(logits)?;
            (token, false)
        };
        self.total_time.add_assign(start_time.elapsed());

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if next_token == 1 {
            return Ok(None);
        }

        let prev_token = self.token;
        self.pos += 1;
        self.token = next_token;

        if is_prompt {
            return Ok(Some("".to_string()));
        }

        Ok(Some(self.runner.tokenizer.decode(prev_token, self.token)?))
    }
}

impl<'a> Iterator for Llama2RunnerOutputGenerator<'a> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let r = self.forward_next().transpose();
            if let Some(Ok(s)) = &r {
                if s.is_empty() {
                    continue;
                }
            }
            return r;
        }
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
    fn test_tensor() -> Result<()> {
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

        let v = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let t = Tensor::new(&v, vec![2, 3, 2, 1]).unwrap();
        assert_eq!(t.at(0)?.flat().to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            t.at(1)?.flat().to_vec(),
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        Ok(())
    }

    #[test]
    fn test_checkpoint_loader() -> Result<()> {
        let loader = Llama2CheckpointLoader::new("testdata/stories15M.bin")?;
        let mut r = loader.reader();
        let conf = Llama2CheckpointLoader::load_config(&mut r)?;
        assert_eq!(conf.dim, 288);
        assert_eq!(conf.hidden_dim, 768);
        assert_eq!(conf.n_heads, 6);
        assert_eq!(conf.n_kv_heads, 6);
        assert_eq!(conf.vocab_size, 32000);
        assert_eq!(conf.n_layers, 6);
        assert_eq!(conf.seq_len, 256);
        assert_eq!(r.total_bytes(), 7 * 4);
        let weights = Llama2CheckpointLoader::load_weights(&mut r, &conf)?;
        assert_eq!(weights.token_embedding_table.shape(), &[32000, 288]);
        assert_eq!(weights.rms_att_weight.shape(), &[6, 288]);
        assert_eq!(weights.rms_ffn_weight.shape(), &[6, 288]);
        assert_eq!(weights.wq.shape(), &[6, 288, 288]);
        assert_eq!(weights.wk.shape(), &[6, 288, 288]);
        assert_eq!(weights.wv.shape(), &[6, 288, 288]);
        assert_eq!(weights.wo.shape(), &[6, 288, 288]);
        assert_eq!(weights.w1.shape(), &[6, 768, 288]);
        assert_eq!(weights.w2.shape(), &[6, 288, 768]);
        assert_eq!(weights.w3.shape(), &[6, 768, 288]);
        assert_eq!(weights.rms_final_weight.shape(), &[288]);
        assert_eq!(weights.freq_cis_real.shape(), &[6144]);
        assert_eq!(weights.freq_cis_imag.shape(), &[6144]);
        assert_eq!(weights.wcls.shape(), &[32000, 288]);
        assert_eq!(r.total_bytes(), 60816028);
        Ok(())
    }

    #[test]
    fn test_tokenizer_decode() -> Result<()> {
        // all the tokens are in utf-8
        let mut loader = Llama2TokenizerLoader::new("testdata/tokenizer.bin")?;
        let tk = loader.load(32000)?;
        assert_eq!(tk.vocab.len(), 32000);
        assert_eq!(tk.vocab_scores[0], 0.0);
        assert_eq!(tk.decode(2, 3)?, "\u{0}");
        assert_eq!(tk.decode(2, 5)?, "\u{2}");
        assert_eq!(tk.decode(2, 6)?, "\u{3}");
        assert_eq!(tk.decode(2, 1000)?, "ied");
        assert_eq!(tk.decode(2, 1001)?, "ER");
        let max_token_len = tk.vocab.iter().map(|v| v.len()).max().unwrap();
        assert_eq!(max_token_len, 27);
        assert_eq!(tk.max_token_length, 27);
        Ok(())
    }

    #[test]
    fn test_tokenizer_encode() -> Result<()> {
        // all the tokens are in utf-8
        let mut loader = Llama2TokenizerLoader::new("testdata/tokenizer.bin")?;
        let tk = loader.load(32000)?;
        let tests = vec![
            ("hello, world", "\n<s>\n -  hello - , -  world - \n</s>\n"),
            (
                "i'm 4 years old",
                "\n<s>\n -  i - ' - m -   - 4 -  years -  old - \n</s>\n",
            ),
            ("tiktok", "\n<s>\n -  t - ik - tok - \n</s>\n"),
            (
                "wake up september",
                "\n<s>\n -  w - ake -  up -  september - \n</s>\n",
            ),
            ("boy", "\n<s>\n -  boy - \n</s>\n"),
            ("fan girl", "\n<s>\n -  fan -  girl - \n</s>\n"),
        ];
        for tt in tests {
            let tokens = tk.encode(tt.0, true, true)?;
            let tokens_in_string = tokens
                .iter()
                .map(|t| tk.vocab[*t].clone())
                .collect::<Vec<String>>()
                .join(" - ");
            assert_eq!(tokens_in_string, tt.1, "failed to encode {}", tt.0);
        }

        Ok(())
    }

    #[test]
    fn test_generate() -> Result<()> {
        let checkpoint_loader = Llama2CheckpointLoader::new("testdata/stories15M.bin")?;
        let mut tokenizer_loader = Llama2TokenizerLoader::new("testdata/tokenizer.bin")?;

        let (conf, weights) = checkpoint_loader.load()?;
        let tokenizer = tokenizer_loader.load(conf.vocab_size)?;
        let mut sampler = Llama2Sampler::new(conf.vocab_size, 0.0, 0.0);
        let mut runner = Llama2Runner::new(&conf, weights, tokenizer);
        let output = runner.generate("hello, world", 15, &mut sampler)?;
        let s = output.collect::<Result<Vec<String>>>()?.join("");
        assert_eq!(
            s,
            "ers. They were very friendly and always had a smile on their faces. One"
        );
        Ok(())
    }
}
