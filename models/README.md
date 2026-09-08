# Model checkpoints

Model weights are local runtime inputs, not source artifacts.

The symbolic CLaMP 3 C2 checkpoint is already stored in `clamp3-c2/`. The
vendored `vendor/clamp3/code/config.py` selects `weights_clamp3_c2`, and the
existing symlink in that directory points back to this checkpoint. Do not move,
replace, or re-download it.

Audio is optional. Install the SAAS checkpoint only with an explicit command:

```bash
bash models/download_checkpoint.sh saas
```

The script refuses to overwrite an existing checkpoint. SAAS weights are
ignored by Git. C2 and SAAS share a dimensionality but are distinct trained
checkpoint spaces; vectors from one must not be compared directly with vectors
from the other.

