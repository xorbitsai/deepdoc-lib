# Deepdoc

### Installations

CPU-only (default):

```bash
pip install deepdoc-lib
```

GPU (Linux x86_64 only):

```bash
pip install deepdoc-lib[gpu]
```

Note: `onnxruntime` (CPU) and `onnxruntime-gpu` should not be installed together. If you're switching an existing environment to GPU, uninstall CPU ORT first:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.19.2
```

### Parser Usage

```python
from deepdoc import (
    DocxParser,
    ExcelParser,
    HtmlParser,
    PdfModelConfig,
    PdfParser,
    TokenizerConfig,
)

# Build configs
# Method 1: Explicit configuration (offline mode)
tokenizer_cfg = TokenizerConfig(
    offline=True,
    nltk_data_dir="/path/to/nltk_data",
)
pdf_model_cfg = PdfModelConfig(
    vision_model_dir="/path/to/models/vision",
    xgb_model_dir="/path/to/models/xgb",
    model_provider="local",
)

# Method 2: Empty configuration (auto-download models and nltk_data)
# tokenizer_cfg = TokenizerConfig()
# pdf_model_cfg = PdfModelConfig()


# Parse PDF
pdf_parser = PdfParser(model_cfg=pdf_model_cfg, tokenizer_cfg=tokenizer_cfg)
result = pdf_parser("document.pdf")

# Parse DOCX / HTML (tokenizer only)
docx_parser = DocxParser(tokenizer_cfg=tokenizer_cfg)
html_parser = HtmlParser(tokenizer_cfg=tokenizer_cfg)

# Parse Excel (no model/tokenizer dependency)
excel_parser = ExcelParser()
with open("data.xlsx", "rb") as f:
    result = excel_parser(f.read())
```

Or use explicit env factories:

```python
tokenizer_cfg = TokenizerConfig.from_env()
pdf_model_cfg = PdfModelConfig.from_env()
pdf_parser = PdfParser(model_cfg=pdf_model_cfg, tokenizer_cfg=tokenizer_cfg)
```

Or rely on defaults (env + cache). Deepdoc will look for cached bundles under
`$DEEPDOC_MODEL_HOME` (or `~/.cache/deepdoc`) and only download missing files
when the provider allows remote access:

```python
pdf_parser = PdfParser()
```

env definitions:

```bash
# provider: auto | local | modelscope
export DEEPDOC_MODEL_PROVIDER=auto

# shared model cache root (default: ~/.cache/deepdoc)
export DEEPDOC_MODEL_HOME=/path/to/deepdoc-models

# optional bundle-specific local directories
export DEEPDOC_VISION_MODEL_DIR=/path/to/vision
export DEEPDOC_XGB_MODEL_DIR=/path/to/xgb

# single combined ModelScope repo (all bundles in one repo)
# (default: Xorbits/deepdoc)
export DEEPDOC_MODELSCOPE_REPO=Xorbits/deepdoc
# optional shared revision (default: master)
export DEEPDOC_MODELSCOPE_REVISION=master

# offline mode for tokenizer NLTK auto-download
export DEEPDOC_OFFLINE=0

# optional NLTK data controls for tokenizer
export DEEPDOC_NLTK_DATA_DIR=/path/to/nltk_data
```

### Download model artifacts

To pre-download all model bundles (vision/xgb/tokenizer) into the default cache directory (`~/.cache/deepdoc`), run:

```bash
deepdoc-download-models
# or (from source checkout)
python -m deepdoc.download_models
```

If you want to override the cache location, set `DEEPDOC_MODEL_HOME`:

```bash
export DEEPDOC_MODEL_HOME=./models
deepdoc-download-models
```

By default this also downloads the required NLTK resources into `~/.cache/deepdoc/nltk_data` (or `$DEEPDOC_MODEL_HOME/nltk_data`).


### Vision Model Usage

``` python
from deepdoc import create_vision_model
```

- Use Environment Variable

```bash
# Vision model configs
export DEEPDOC_VISION_PROVIDER="qwen"
export DEEPDOC_VISION_API_KEY="your-api-key"
export DEEPDOC_VISION_MODEL="qwen-vl-max"
export DEEPDOC_VISION_LANG="Chinese"
export DEEPDOC_VISION_BASE_URL="http://your_base_url"

# Other configs
export DEEPDOC_LIGHTEN=0  # Whether to use lighten mode
```

``` python
vision_model = create_vision_model()
```

- Use Default Provider

``` bash
export DEEPDOC_VISION_API_KEY="your-api-key"
```

``` python
vision_model = create_vision_model("qwen")
```

Supported providers: ["openai", "qwen", "zhipu", "ollama", "gemini", "anthropic"]

- Use Configuration File

Create `deepdoc_config.yaml`:

```yaml
vision_model:
  provider: "qwen"
  model_name: "qwen-vl-max"
  api_key: "your-api-key"
  lang: "Chinese"
  base_url : "http://your-base-url"
```

``` python
vision_model = create_vision_model("/path/to/deepdoc_config.yaml")
```

#### Run
``` python
with open("image.jpg", "rb") as f:
    result = vision_model.describe_with_prompt(f.read())
```
