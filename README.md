# Libsnout

This is a rust implementation of Project Babble's baballonia face tracking sofware.
It's designed to be a library; easy to integrate in a variety of frontend projects.

## Building libsnout

Make sure you have `onnxruntime` and `opencv` installed.

On fedora it's:
```sh
dnf install onnxruntime onnxruntime-devel opencv opencv-devel
```

### Compiling

```bash
export ORT_PREFER_DYNAMIC_LINK=1
cargo build
```

## Building and running frontend

```sh
export ORT_LIB_PATH=/usr/lib64/
export ORT_PREFER_DYNAMIC_LINK=1
cargo build -p snout-frontend
```

## License

Right now it's licensed under the same license as Baballonia from Project Babble is, considering this is a derivative work.
