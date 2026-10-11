# Compute-node environment installation

Do this only after entering an allocated GPU compute node through the company
jump host. Never compile CUDA packages on the jump host.

```bash
conda create -n multiaxis32b python=3.10 -y
conda activate multiaxis32b

# Install the PyTorch 2.5.1 build matching the actual NVIDIA driver/CUDA policy.
# Use the company's approved package mirror/index when required.
pip install -r requirements-multi-axis-32b.txt
pip install --no-build-isolation -r requirements-multi-axis-32b-flash.txt

python -m pip check
python -m pytest source/checkout/tools/multi_axis/test_multiaxis.py -q
```

The package intentionally does not include a guessed CUDA wheelhouse: PyTorch,
FlashAttention, and NCCL ABI selection depends on the GPU/driver information
that will only be known at the company. The exact Python dependency pins are
included; record the installed wheel filenames and `pip freeze` in the run
directory after hardware discovery.
