# PyPI Trusted Publisher 配置说明

## 问题描述

如果遇到以下错误：
```
Error: Trusted publishing exchange failure: 
Token request failed: the server refused the request for the following reasons:
* `invalid-publisher`: valid token, but no corresponding publisher
```

这表示 PyPI 上还没有配置 trusted publisher。

## 解决方案

### 1. 在 GitHub 上创建 Environment

1. 进入仓库：`https://github.com/HuangPuStar/deepdoc-lib`
2. 点击 **Settings** → **Environments**
3. 创建两个 environment：
   - `pypi`（用于生产环境）
   - `testpypi`（用于测试环境）

### 2. 在 PyPI 上配置 Trusted Publisher

#### 对于 TestPyPI (test.pypi.org)

1. 登录 TestPyPI：https://test.pypi.org/manage/account/publishing/
2. 点击 **Add a new pending publisher**
3. 填写以下信息：
   - **PyPI project name**: `deepdoc-lib`
   - **Owner**: `HuangPuStar`
   - **Repository name**: `deepdoc-lib`
   - **Workflow filename**: `.github/workflows/publish.yml`
   - **Environment name**: `testpypi`（必须与 workflow 中的 environment 名称一致）
   - **Publisher type**: `GitHub Actions`
4. 点击 **Add**

#### 对于 PyPI (pypi.org)

1. 登录 PyPI：https://pypi.org/manage/account/publishing/
2. 点击 **Add a new pending publisher**
3. 填写以下信息：
   - **PyPI project name**: `deepdoc-lib`
   - **Owner**: `HuangPuStar`
   - **Repository name**: `deepdoc-lib`
   - **Workflow filename**: `.github/workflows/publish.yml`
   - **Environment name**: `pypi`（必须与 workflow 中的 environment 名称一致）
   - **Publisher type**: `GitHub Actions`
4. 点击 **Add**

### 3. 验证配置

配置完成后，从错误信息中获取的 `sub` 值应该匹配：
- TestPyPI: `repo:HuangPuStar/deepdoc-lib:environment:testpypi`
- PyPI: `repo:HuangPuStar/deepdoc-lib:environment:pypi`

### 4. 测试发布

配置完成后，可以：
1. 推送到 `feature/*` 分支 → 自动发布到 TestPyPI
2. 创建 tag（如 `v1.0.0`）→ 自动发布到 PyPI

## 参考文档

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)

