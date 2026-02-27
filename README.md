# Deepdoc

### Installations

CPU-only (default):

```bash
pip install git+https://github.com/xorbitsai/deepdoc-lib
```

GPU (Linux x86_64 only):

```bash
pip install "deepdoc-lib[gpu] @ git+https://github.com/xorbitsai/deepdoc-lib"
```

Note: `onnxruntime` (CPU) and `onnxruntime-gpu` should not be installed together. If you're switching an existing environment to GPU, uninstall CPU ORT first:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.19.2
```

### Parser Usage

```python
from deepdoc import PdfParser, DocxParser, ExcelParser

# 解析 PDF
pdf_parser = PdfParser()
result = pdf_parser("document.pdf")

# 解析 Word
docx_parser = DocxParser()
result = docx_parser("document.docx")

# 解析 Excel
excel_parser = ExcelParser()
with open("data.xlsx", "rb") as f:
    result = excel_parser(f.read())
```


### Vision Model Usage

``` python
from deepdoc import create_vision_model
```

- Use Environment Variable

```bash
# 视觉模型配置
export DEEPDOC_VISION_PROVIDER="qwen"
export DEEPDOC_VISION_API_KEY="your-api-key"
export DEEPDOC_VISION_MODEL="qwen-vl-max"
export DEEPDOC_VISION_LANG="Chinese"
export DEEPDOC_VISION_BASE_URL="http://your_base_url"

# 其他配置
export DEEPDOC_LIGHTEN=0  # 是否使用轻量模式
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

