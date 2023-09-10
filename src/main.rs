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

    /// The prompt
    prompt: String,
}

fn main() -> Result<()> {
    let args = CommandArgs::parse();

    let checkpoint_loader = Llama2CheckpointLoader::new(&args.checkpoint)?;
    let mut tokenizer_loader = Llama2TokenizerLoader::new(&args.tokenizer)?;

    let (conf, weights) = checkpoint_loader.load()?;
    let tokenizer = tokenizer_loader.load(conf.vocab_size)?;
    let mut sampler = Llama2Sampler::new(conf.vocab_size, 0.0, 0.0);
    let mut runner = Llama2Runner::new(&conf, weights, tokenizer);
    let mut output = runner.generate(&args.prompt, args.steps, &mut sampler)?;
    for token in output.by_ref() {
        print!("{}", token?);
        std::io::stdout().flush().unwrap();
    }
    println!();
    println!("{} tokens/s", output.average_tokens_per_seconds());

    Ok(())
}
