mod llama2;
use clap::Parser;
use crate::llama2::{Llama2CheckpointLoader, Llama2TokenizerLoader, Llama2Sampler, Llama2Runner, Result};

#[derive(Parser, Debug)]
struct CommandArgs {
    /// The checkpoint file to load
    #[arg(short, long, default_value_t = format!("./testdata/stories15M.bin"))]
    checkpoint: String,

    // The tokenizer file to load
    #[arg(short, long, default_value_t = format!("./testdata/tokenizer.bin"))]
    tokenizer: String,

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
    let output = runner.generate(&args.prompt, 15, &mut sampler)?;
    for token in output {
        print!("{}", token?);
    }

    Ok(())
}
