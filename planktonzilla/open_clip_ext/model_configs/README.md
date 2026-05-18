# planktonzilla open_clip_ext model_configs

This directory is **empty by design**. It holds project-local CLIP model
JSON configs that would be registered with upstream `open_clip` via
`open_clip.add_model_config(__path__[0])` from
`planktonzilla/open_clip_ext/factory.py` at import time.

## Why empty today

The Phase 1 audit (`docs/open_clip_audit.md` Q2, row 9) found that **no
project-local CLIP model JSONs exist** — all currently-used architectures
(`ViT-B-16/openai`, `EVA02-L-14/merged2b_s4b_b131k`) are stock upstream
model definitions registered by `open-clip-torch`'s own
`model_configs/` directory.

## How to add a project-local config

If you need to add a CLIP architecture not in upstream's registry:

1. Add a JSON config file here, e.g., `MyCustomCLIP.json`, in the same
   schema as upstream's `open_clip/src/open_clip/model_configs/*.json`
   files. See the upstream files for examples; the schema is
   `{"embed_dim": int, "vision_cfg": {...}, "text_cfg": {...}}`.

2. Wire registration in `planktonzilla/open_clip_ext/factory.py` by
   adding a single line at module level, guarded by a presence check:

   ```python
   # Add to the top of factory.py, after imports:
   from pathlib import Path as _Path
   _CONFIGS_DIR = _Path(__file__).parent / "model_configs"
   if any(_CONFIGS_DIR.glob("*.json")):
       open_clip.add_model_config(str(_CONFIGS_DIR))
   ```

3. Update `factory.py`'s module docstring `Overrides:` section to note
   the registration: "Also calls `open_clip.add_model_config()` to
   register project-local JSON configs from `model_configs/`."

4. Reference the new model in your Hydra config:
   `configs/model/my_custom_clip.yaml` with `_args_: [MyCustomCLIP, null]`.

## Why this README exists when there are no JSONs

EXT-02 mandates a mirrored upstream layout. Upstream has
`model_configs/`; the override layer has `model_configs/`. The README
documents the affordance for the next contributor who needs the
mechanism — without forcing today's planner to scaffold registration
code that the audit explicitly said is unneeded.
