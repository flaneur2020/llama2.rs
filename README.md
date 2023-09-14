# 🦙🦙.🦀

An rust reimplementatin of <https://github.com/karpathy/llama2.c>.

## Run it

This repo contains a tinystories15M in the `testdata/` folder, you can run it with the following command:

```bash
cargo run --release
./target/release/llama2-rs "rust is a crab"
```

https://github.com/flaneur2020/llama2.rs/assets/129800/8754c25e-c016-47b0-bb97-99930624a097

## Performance

| model | tokens/s |
| ----- | -------- |
| tinystories15M | 120~197 |  


