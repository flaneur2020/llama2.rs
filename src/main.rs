mod llama2;
use clap::Parser;

#[derive(Parser)]
struct CommandArgs {
    /// The checkpoint file to load
    checkpoint: String,
    /// The prompt
    prompt: String,
}

fn main() {
    let args = CommandArgs::parse();
}
