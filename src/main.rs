mod llama2;
use clap::Parser;
use llama2::{Llama2CheckpointLoader, Llama2Runner, Llama2Sampler, Llama2TokenizerLoader, Result};
use std::io::Write;

#[derive(Parser, Debug)]
struct CommandArgs {
    /// The checkpoint file to load
    #[arg(short, long, default_value_t = format!("./testdata/stories15M.bin"))]
    checkpoint: String,

    // The tokenizer file to load
    #[arg(short, long, default_value_t = format!("./testdata/tokenizer.bin"))]
    tokenizer: String,

    // The number of tokens to generate
    #[arg(short, long, default_value_t = 300)]
    steps: usize,

    // The probability of sampling from the top-p.
    #[arg(short, long, default_value_t = 0.9)]
    probability: f32,

    #[arg(short, long, default_value_t = 1.0)]
    temperature: f32,

    /// The prompt
    prompt: String,
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();

    // configure rayon
    let threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let mut checkpoint_loader = Llama2CheckpointLoader::new(&args.checkpoint, &args.tokenizer)?;

    let (conf, weights, tokenizer) = checkpoint_loader.load()?;
    let mut sampler = Llama2Sampler::new(conf.vocab_size, args.temperature, args.probability);
    let mut runner = Llama2Runner::new(&conf, weights, tokenizer);
    let mut output = runner.generate(&args.prompt, args.steps, &mut sampler)?;
    for token in output.by_ref() {
        print!("{}", token?);
        std::io::stdout().flush().unwrap();
    }
    println!();
    println!("{} tokens/s, {} threads", output.average_tokens_per_seconds(), threads);

    Ok(())
}
