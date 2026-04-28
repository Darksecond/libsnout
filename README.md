# Libsnout

This is a rust implementation of Project Babble's baballonia face tracking sofware.
It's designed to be a library; easy to integrate in a variety of frontend projects.

## Building libsnout

Make sure you have `llvm-devel` installed.
A working face tracking model is supplied. It's the same as in the baballonia repository, but ran through `onnxsim`.

On fedora it's:
```sh
dnf install llvm llvm-devel onnxruntime onnxruntime-devel
```

### Generating `snout.h`

```sh
cargo install --force cbindgen

export PATH=$PATH:/home/proto/.cargo/bin
cbindgen --config cbindgen.toml --output include/snout.h
```

## Building and running the CLI

Configure the `config.toml` to your liking and run. It will show the different commands.

```sh
cargo run --release -p snout-cli -- -c config.toml help
```

## License

Right now it's licensed under the same license as Baballonia from Project Babble is, considering this is a derivative work.
