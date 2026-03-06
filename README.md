# PineDocs - 文档向量管理系统

PineDocs 是一个基于 Gradio 的文档管理和向量搜索工具，支持将各种格式的文档上传到 Pinecone 向量数据库，并进行语义搜索。

## 功能特性

- **多格式文档支持**：支持 TXT、MD、DOC、DOCX、XLSX、PDF、图片等多种格式
- **向量数据库存储**：使用 Pinecone 作为向量数据库，支持高效的语义搜索
- **OCR 文字识别**：集成 Ollama OCR 模型，支持图片和 PDF 的文字识别
- **PDF 转 Markdown**：使用 OCR 技术将 PDF 文档转换为 Markdown 格式
- **命名空间管理**：支持多命名空间管理，便于分类存储不同文档
- **语义搜索**：基于向量相似度的智能文档搜索
- **灵活的嵌入模型**：支持自定义 OpenAI 兼容格式的嵌入模型

## 系统要求

- Windows 10/11
- Python 3.8+
- Microsoft Word（用于处理 .doc 文件，可选）

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/deki18/PineDocs.git
cd PineDocs
```

### 2. 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 文件为 `.env`，并填写你的 API 密钥：

```bash
copy .env.example .env
```

编辑 `.env` 文件：

```env
PINECONE_API_KEY=your_pinecone_api_key
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.vectorengine.ai/v1
EMBEDDING_MODEL=text-embedding-3-small
```

### 5. 启动应用

```bash
python app_gradio.py
```

应用将在 http://localhost:7860 启动。

## 嵌入模型配置

支持任何 OpenAI API 格式的嵌入模型服务，配置以下环境变量：

```env
EMBEDDING_API_KEY=your_api_key
EMBEDDING_BASE_URL=https://api.example.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

**常用服务示例：**
- **VectorEngine**: `https://api.vectorengine.ai/v1`
- **OpenAI**: `https://api.openai.com/v1`
- **Ollama**: `http://localhost:11434/v1`

**注意**：不同模型输出维度不同（1536/3072），需与 Pinecone 索引维度一致。

## OCR 模型配置

### 安装 Ollama

1. 下载并安装 [Ollama](https://ollama.com/)
2. 拉取 OCR 模型：

```bash
# GLM-OCR 模型（通用）
ollama pull my-glm-ocr:latest

# PaddleOCR-VL 模型（文档识别效果更好）
ollama pull my-PaddleOCR-VL:0.9b
```

## 常见问题

### 1. 命名空间不显示

Pinecone 只返回包含向量的命名空间。如果命名空间中没有数据，不会在下拉框中显示。可以直接在输入框中输入命名空间名称。

### 2. .doc 文件处理失败

处理 .doc 文件需要：
- Windows 系统
- 安装 Microsoft Word
- 安装 pywin32：`pip install pywin32`

建议将 .doc 文件另存为 .docx 格式后上传。

### 3. OCR 识别失败

- 确保 Ollama 服务已启动
- 确保已安装所需的 OCR 模型
- 检查 Ollama URL 配置是否正确

### 4. 嵌入模型维度不匹配

如果更换嵌入模型后出现维度错误，需要：
1. 删除现有索引，或
2. 创建新的索引（在界面中选择或输入新索引名称）

不同模型的输出维度：
- text-embedding-3-small: 1536 维
- text-embedding-3-large: 3072 维
- text-embedding-ada-002: 1536 维

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 作者

- GitHub: [@deki18](https://github.com/deki18)

## 致谢

- [Gradio](https://gradio.app/) - Web 界面框架
- [Pinecone](https://www.pinecone.io/) - 向量数据库
- [VectorEngine](https://vectorengine.ai/) - 嵌入模型服务
- [Ollama](https://ollama.com/) - 本地大模型运行框架
