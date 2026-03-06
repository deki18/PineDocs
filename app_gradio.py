import os
import base64
import io
import hashlib
from typing import List, Tuple, Optional
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
import requests
import fitz  # PyMuPDF

from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from openai import OpenAI

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("VECTORENGINE_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.vectorengine.ai/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
os.makedirs(DOCS_DIR, exist_ok=True)


# ============ 核心功能函数 ============

def generate_safe_id(filename: str, batch_id: int, chunk_id: int) -> str:
    """生成 ASCII 安全的 Vector ID"""
    ext = Path(filename).suffix
    name_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
    safe_id = f"doc_{name_hash}_b{batch_id}_c{chunk_id}{ext}"
    return safe_id


def get_embed_client(api_key=None, base_url=None):
    """获取嵌入客户端"""
    key = api_key or EMBEDDING_API_KEY
    url = base_url or EMBEDDING_BASE_URL
    
    if not key:
        return None
    return OpenAI(
        base_url=url,
        api_key=key,
    )


def get_pinecone_index(index_name: str, dimension=None):
    """获取或创建 Pinecone 索引"""
    if not PINECONE_API_KEY:
        return None, "请在.env文件中设置PINECONE_API_KEY"

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        
        dim = dimension or EMBEDDING_DIMENSION
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1
                ),
            )
        
        return pc.Index(index_name), None
    except Exception as e:
        return None, f"获取索引失败: {str(e)}"


def embed_texts(texts: List[str], model=None, api_key=None, base_url=None) -> Tuple[List[List[float]], str]:
    """将文本转换为向量"""
    client = get_embed_client(api_key=api_key, base_url=base_url)
    if not client:
        return [], "未配置嵌入模型API密钥"

    try:
        model_name = model or EMBEDDING_MODEL
        response = client.embeddings.create(
            model=model_name,
            input=texts,
        )
        return [item.embedding for item in response.data], None
    except Exception as e:
        return [], f"嵌入失败: {str(e)}"


def split_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """将文本分割成块"""
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        
        if end < length:
            last_newline = text.rfind('\n', start, end)
            last_period = text.rfind('。', start, end)
            last_exclamation = text.rfind('！', start, end)
            last_question = text.rfind('？', start, end)
            last_semicolon = text.rfind('；', start, end)
            
            best_break = max(last_newline, last_period, last_exclamation, last_question, last_semicolon)
            
            if best_break > start + max_chars // 2:
                end = best_break + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        if end == length:
            break
        
        start = max(end - overlap, 0)

    return chunks


def extract_text_from_file(file_obj, ollama_url: str = "http://localhost:11434", ocr_model: str = "my-glm-ocr:latest") -> Tuple[str, str]:
    """从文件中提取文本"""
    try:
        # Gradio 的 File 组件返回的是文件路径（字符串）或文件对象
        if isinstance(file_obj, str):
            # 如果是字符串路径
            filename = file_obj
            with open(file_obj, 'rb') as f:
                content = f.read()
        elif hasattr(file_obj, 'name'):
            # 如果是文件对象
            filename = file_obj.name
            if hasattr(file_obj, 'read'):
                content = file_obj.read()
            else:
                # 可能是路径对象
                with open(file_obj.name, 'rb') as f:
                    content = f.read()
        else:
            return "", f"无效的文件对象类型: {type(file_obj)}"
        
        ext = Path(filename).suffix.lower()
        
        # 图片文件使用OCR
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return ocr_image(content, ollama_url, ocr_model)
        
        # PDF文件
        elif ext == '.pdf':
            return pdf_to_text(content, ollama_url, ocr_model, filename)
        
        # Word文档
        elif ext in ['.doc', '.docx']:
            return extract_word_text(content, ext)
        
        # Excel文件
        elif ext == '.xlsx':
            return extract_excel_text(content)
        
        # 文本文件
        elif ext in ['.txt', '.md']:
            try:
                return content.decode('utf-8'), None
            except:
                try:
                    return content.decode('gbk'), None
                except:
                    return content.decode('latin-1'), None
        
        else:
            return "", f"不支持的文件格式: {ext}"
            
    except Exception as e:
        return "", f"提取文本失败: {str(e)}"


def ocr_image(image_bytes: bytes, ollama_url: str, model_name: str) -> Tuple[str, str]:
    """使用Ollama进行OCR识别"""
    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "请识别图片中的文字内容，直接输出识别到的文字，不要添加任何解释。",
                "images": [image_base64],
                "stream": False
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", ""), None
        else:
            return "", f"OCR请求失败: {response.status_code}"
    except Exception as e:
        return "", f"OCR识别失败: {str(e)}"


def pdf_to_text(pdf_bytes: bytes, ollama_url: str, model_name: str, filename: str) -> Tuple[str, str]:
    """将PDF转换为文本"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                all_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}")
        
        doc.close()
        return "\n\n".join(all_text), None
    except Exception as e:
        return "", f"PDF处理失败: {str(e)}"


def extract_word_text(file_bytes: bytes, ext: str) -> Tuple[str, str]:
    """从Word文档提取文本"""
    try:
        if ext == '.docx':
            # 处理 .docx 文件 (ZIP格式的XML)
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs), None
        elif ext == '.doc':
            # 处理 .doc 文件 (旧版Word二进制格式)
            # 在Windows上使用Word COM接口
            try:
                import win32com.client
                import pythoncom
                import tempfile
                import os
                
                # 初始化COM
                pythoncom.CoInitialize()
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_doc:
                    temp_doc.write(file_bytes)
                    temp_doc_path = temp_doc.name
                
                temp_docx_path = temp_doc_path + 'x'
                
                try:
                    # 使用Word将.doc转换为.docx
                    word = win32com.client.Dispatch("Word.Application")
                    word.Visible = False
                    doc_obj = word.Documents.Open(temp_doc_path)
                    doc_obj.SaveAs(temp_docx_path, FileFormat=16)  # 16 = wdFormatXMLDocument
                    doc_obj.Close()
                    word.Quit()
                    
                    # 读取转换后的.docx文件
                    from docx import Document
                    doc = Document(temp_docx_path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    return "\n".join(paragraphs), None
                    
                finally:
                    # 清理临时文件
                    pythoncom.CoUninitialize()
                    if os.path.exists(temp_doc_path):
                        os.remove(temp_doc_path)
                    if os.path.exists(temp_docx_path):
                        os.remove(temp_docx_path)
                        
            except ImportError:
                return "", "无法处理 .doc 文件。请安装 pywin32: pip install pywin32"
            except Exception as e:
                return "", f".doc文件处理失败: {str(e)}"
        else:
            return "", f"不支持的Word格式: {ext}"
    except Exception as e:
        return "", f"Word文档处理失败: {str(e)}"


def extract_excel_text(file_bytes: bytes) -> Tuple[str, str]:
    """从Excel文件提取文本"""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(file_bytes))
        all_text = []
        
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            sheet_text = [f"--- 工作表: {sheet_name} ---"]
            
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join(str(cell) for cell in row if cell is not None)
                if row_text.strip():
                    sheet_text.append(row_text)
            
            all_text.append("\n".join(sheet_text))
        
        return "\n\n".join(all_text), None
    except Exception as e:
        return "", f"Excel处理失败: {str(e)}"


def batched(items, batch_size=32):
    """将列表分批"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def pdf_to_markdown_with_ollama(pdf_bytes: bytes, ollama_url: str = "http://localhost:11434", model_name: str = "my-glm-ocr:latest", filename: str = "document.pdf") -> Tuple[str, str]:
    """使用Ollama的OCR模型将PDF转换为Markdown
    
    支持的模型:
    - my-PaddleOCR-VL:0.9b: PaddleOCR-VL-1.5 (推荐，效果更好)
    - my-glm-ocr:latest: GLM-OCR模型
    
    Args:
        pdf_bytes: PDF文件的字节数据
        ollama_url: Ollama服务地址
        model_name: 使用的OCR模型名称
        filename: 原始文件名，用于保存Markdown文件
    
    Returns:
        (markdown_content, error_message)
    """
    try:
        # 使用PyMuPDF将PDF转换为图片列表
        try:
            # 打开PDF
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            # 设置缩放比例 (zoom=2.0 相当于 144 DPI, zoom=3.0 相当于 216 DPI)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # 将页面渲染为图片
                pix = page.get_pixmap(matrix=mat)
                # 转换为PNG数据
                img_data = pix.tobytes("png")
                images.append(img_data)
            
            pdf_document.close()
            
        except Exception as e:
            return "", f"PDF转图片失败: {str(e)}"
        
        if not images:
            return "", "PDF中没有找到页面"
        
        all_content = []
        
        # 逐页处理
        for page_num, img_data in enumerate(images, start=1):
            # 将图片转换为base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # 根据模型选择不同的调用方式
            if model_name == "my-PaddleOCR-VL:0.9b":
                # PaddleOCR-VL-1.5 使用chat API
                response = requests.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": "my-PaddleOCR-VL:0.9b",
                        "messages": [
                            {
                                "role": "user",
                                "content": "请识别这张图片中的文字内容，并以Markdown格式输出。保留原有的段落结构和格式。",
                                "images": [img_base64]
                            }
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 4096
                        }
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("message", {}).get("content", "")
                    if not content:
                        content = result.get("response", "")
                    if content:
                        all_content.append(f"## 第 {page_num} 页\n\n{content}")
            else:
                # GLM-OCR 使用Ollama原生 /api/generate 端点
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": "my-glm-ocr:latest",
                        "prompt": "Text Recognition:",
                        "images": [img_base64],
                        "stream": False
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("response", "")
                    if not content:
                        content = result.get("message", {}).get("content", "")
                    if content:
                        all_content.append(f"## 第 {page_num} 页\n\n{content}")
        
        if all_content:
            markdown_content = "\n\n---\n\n".join(all_content)
            
            # 自动保存Markdown文件到docs文件夹
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(filename)[0]
                md_filename = f"{base_name}_{timestamp}.md"
                md_filepath = os.path.join(DOCS_DIR, md_filename)
                
                with open(md_filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
            except Exception as save_error:
                print(f"保存Markdown文件失败: {str(save_error)}")
            
            return markdown_content, None
        else:
            return "", "未能识别出任何内容"
            
    except Exception as e:
        return "", f"PDF转Markdown失败: {str(e)}"


# ============ Gradio 界面函数 ============

def get_index_list():
    """获取索引列表"""
    if not PINECONE_API_KEY:
        return [], "未配置PINECONE_API_KEY"
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        indexes = [idx["name"] for idx in pc.list_indexes()]
        return indexes, None
    except Exception as e:
        return [], f"获取索引列表失败: {str(e)}"


def get_namespace_list(index_name: str):
    """获取命名空间列表"""
    if not index_name or index_name == "请先创建索引":
        return ["default"]
    
    try:
        index, error = get_pinecone_index(index_name)
        if error:
            print(f"获取索引失败: {error}")
            return ["default"]
        
        stats = index.describe_index_stats()
        namespaces = []
        
        # 调试信息
        print(f"Index stats type: {type(stats)}")
        print(f"Index stats: {stats}")
        
        # 尝试不同的方式获取命名空间
        ns_data = None
        if stats:
            # 方式1: 直接访问 namespaces 属性
            if hasattr(stats, 'namespaces'):
                ns_data = stats.namespaces
                print(f"方式1 - stats.namespaces: {ns_data}")
            # 方式2: 转换为字典
            elif hasattr(stats, '__dict__'):
                stats_dict = stats.__dict__
                print(f"方式2 - stats.__dict__: {stats_dict}")
                ns_data = stats_dict.get('namespaces')
            # 方式3: 如果是字典类型
            elif isinstance(stats, dict):
                ns_data = stats.get('namespaces')
                print(f"方式3 - dict get: {ns_data}")
        
        print(f"ns_data type: {type(ns_data)}")
        print(f"ns_data: {ns_data}")
        
        if ns_data:
            # 如果是字典类型
            if isinstance(ns_data, dict):
                for ns in ns_data.keys():
                    if ns and str(ns).strip():
                        namespaces.append(str(ns))
            # 如果是其他可迭代类型
            elif hasattr(ns_data, '__iter__') and not isinstance(ns_data, str):
                for ns in ns_data:
                    if ns and str(ns).strip():
                        namespaces.append(str(ns))
        
        # 添加default选项
        if "default" not in namespaces:
            namespaces.insert(0, "default")
        
        result = sorted(list(set(namespaces)))
        print(f"最终返回命名空间列表: {result}")
        return result
    except Exception as e:
        print(f"获取命名空间列表失败: {str(e)}")
        return ["default"]


def refresh_namespaces(index_name: str):
    """刷新命名空间列表"""
    namespaces = get_namespace_list(index_name)
    return gr.update(choices=namespaces, value=namespaces[0] if namespaces else "default")


def upload_files(files, index_name, namespace, max_chars, overlap, ollama_url, ocr_model, 
                 embedding_api_key, embedding_base_url, embedding_model, embedding_dimension):
    """上传文件到Pinecone"""
    if not files:
        return "请至少选择一个文件"
    
    if not index_name:
        return "请先选择一个索引"
    
    # 使用自定义命名空间或选择的命名空间
    final_namespace = namespace.strip() if namespace and namespace.strip() else "default"
    
    # 使用自定义嵌入模型配置
    api_key = embedding_api_key.strip() if embedding_api_key and embedding_api_key.strip() else None
    base_url = embedding_base_url.strip() if embedding_base_url and embedding_base_url.strip() else None
    model = embedding_model.strip() if embedding_model and embedding_model.strip() else None
    dimension = int(embedding_dimension) if embedding_dimension else None
    
    index, error = get_pinecone_index(index_name, dimension=dimension)
    if error:
        return error
    
    results = []
    total_docs = 0
    total_vectors = 0
    
    for file_obj in files:
        try:
            filename = os.path.basename(file_obj.name)
            results.append(f"📄 处理文件: {filename}")
            
            # 提取文本
            content, error = extract_text_from_file(file_obj, ollama_url, ocr_model)
            if error:
                results.append(f"  ❌ {error}")
                continue
            
            if not content.strip():
                results.append(f"  ⚠️ 文件内容为空，跳过")
                continue
            
            total_docs += 1
            
            # 分割文本
            chunks = split_text(content, max_chars=max_chars, overlap=overlap)
            results.append(f"  ✓ 分割成 {len(chunks)} 个文本块")
            
            # 生成向量并上传
            vectors_to_upsert = []
            for batch_id, batch_chunks in enumerate(batched(chunks, batch_size=32), start=1):
                embeddings, error = embed_texts(
                    batch_chunks, 
                    model=model, 
                    api_key=api_key, 
                    base_url=base_url
                )
                if error:
                    results.append(f"  ❌ 嵌入失败: {error}")
                    continue
                
                for i, (chunk_text, embedding) in enumerate(zip(batch_chunks, embeddings), start=1):
                    vec_id = generate_safe_id(filename, batch_id, i)
                    vectors_to_upsert.append({
                        "id": vec_id,
                        "values": embedding,
                        "metadata": {
                            "source": filename,
                            "chunk_index": i,
                            "text": chunk_text,
                        },
                    })
            
            if vectors_to_upsert:
                index.upsert(vectors=vectors_to_upsert, namespace=final_namespace)
                total_vectors += len(vectors_to_upsert)
                results.append(f"  ✓ 成功上传 {len(vectors_to_upsert)} 个向量到命名空间 '{final_namespace}'")
            
        except Exception as e:
            results.append(f"  ❌ 处理失败: {str(e)}")
    
    results.append(f"\n{'='*50}")
    results.append(f"📊 上传完成!")
    results.append(f"  - 处理文档: {total_docs}")
    results.append(f"  - 创建向量: {total_vectors}")
    results.append(f"  - 命名空间: {final_namespace}")
    
    return "\n".join(results)


def search_documents(query, index_name, namespace, top_k=5, 
                     embedding_api_key=None, embedding_base_url=None, embedding_model=None):
    """搜索文档"""
    if not query.strip():
        return "请输入搜索内容"
    
    if not index_name:
        return "请先选择一个索引"
    
    final_namespace = namespace.strip() if namespace and namespace.strip() else "default"
    
    # 使用自定义嵌入模型配置
    api_key = embedding_api_key.strip() if embedding_api_key and embedding_api_key.strip() else None
    base_url = embedding_base_url.strip() if embedding_base_url and embedding_base_url.strip() else None
    model = embedding_model.strip() if embedding_model and embedding_model.strip() else None
    
    index, error = get_pinecone_index(index_name)
    if error:
        return error
    
    # 获取查询的向量
    embeddings, error = embed_texts([query], model=model, api_key=api_key, base_url=base_url)
    if error:
        return error
    
    if not embeddings:
        return "无法生成查询向量"
    
    try:
        results = index.query(
            vector=embeddings[0],
            top_k=top_k,
            namespace=final_namespace,
            include_metadata=True
        )
        
        if not results.matches:
            return f"在命名空间 '{final_namespace}' 中未找到相关文档"
        
        output = [f"🔍 搜索结果 (命名空间: {final_namespace})\n"]
        for i, match in enumerate(results.matches, 1):
            source = match.metadata.get("source", "未知")
            text = match.metadata.get("text", "")[:300]
            score = match.score
            output.append(f"{i}. 📄 {source} (相似度: {score:.3f})")
            output.append(f"   {text}...\n")
        
        return "\n".join(output)
    except Exception as e:
        return f"搜索失败: {str(e)}"


def clear_namespace(index_name, namespace):
    """清空命名空间"""
    if not index_name:
        return "请先选择一个索引"
    
    if not namespace or namespace == "default":
        return "不能清空默认命名空间"
    
    index, error = get_pinecone_index(index_name)
    if error:
        return error
    
    try:
        index.delete(delete_all=True, namespace=namespace)
        return f"✅ 命名空间 '{namespace}' 已清空"
    except Exception as e:
        return f"清空失败: {str(e)}"


def convert_pdf_to_markdown(pdf_file, ollama_url, ocr_model):
    """PDF转Markdown的Gradio界面函数"""
    if pdf_file is None:
        return "请先上传PDF文件"
    
    try:
        # 读取PDF文件
        if hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            filename = pdf_file.name if hasattr(pdf_file, 'name') else "document.pdf"
        else:
            return "无效的文件对象"
        
        # 调用转换函数
        markdown_content, error = pdf_to_markdown_with_ollama(pdf_bytes, ollama_url, ocr_model, filename)
        
        if error:
            return f"转换失败: {error}"
        
        return markdown_content
    except Exception as e:
        return f"转换过程出错: {str(e)}"


# ============ 创建 Gradio 界面 ============

def create_ui():
    indexes, error = get_index_list()
    
    # 获取初始命名空间列表
    initial_namespaces = ["default"]
    if indexes and indexes[0] and indexes[0] != "请先创建索引":
        initial_namespaces = get_namespace_list(indexes[0])
    
    with gr.Blocks(title="PineDocs - 文档向量管理系统", css="""
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .header h1 { color: #1f77b4; }
        .section { margin: 1rem 0; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 8px; }
    """) as demo:
        
        gr.Markdown("""
        # 🌲 PineDocs
        ## 文档向量管理系统
        """)
        
        if error:
            gr.Warning(error)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 配置")
                
                index_dropdown = gr.Dropdown(
                    choices=indexes if indexes else ["请先创建索引"],
                    value=indexes[0] if indexes else "请先创建索引",
                    label="选择索引",
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 刷新命名空间", size="sm")
                
                namespace_dropdown = gr.Dropdown(
                    choices=initial_namespaces,
                    value=initial_namespaces[0] if initial_namespaces else "default",
                    label="选择已有命名空间",
                    interactive=True,
                    info="只显示有数据的命名空间"
                )
                
                custom_namespace = gr.Textbox(
                    label="输入命名空间（新建或已有）",
                    placeholder="输入命名空间名称",
                    value="default",
                    info="上传文档时会自动创建，搜索时会自动查找"
                )
                
                ollama_url = gr.Textbox(
                    label="Ollama URL",
                    value="http://localhost:11434",
                    info="用于OCR识别的本地服务地址"
                )
                
                ocr_model = gr.Dropdown(
                    choices=["my-glm-ocr:latest", "my-PaddleOCR-VL:0.9b"],
                    value="my-glm-ocr:latest",
                    label="OCR模型"
                )
                
                gr.Markdown("---")
                gr.Markdown("**嵌入模型设置**")
                
                embedding_api_key = gr.Textbox(
                    label="嵌入模型 API Key",
                    value=EMBEDDING_API_KEY if EMBEDDING_API_KEY else "",
                    placeholder="输入API密钥",
                    info="支持 OpenAI 兼容格式",
                    type="password"
                )
                
                embedding_base_url = gr.Textbox(
                    label="嵌入模型 Base URL",
                    value=EMBEDDING_BASE_URL,
                    placeholder="https://api.example.com/v1",
                    info="OpenAI 兼容的 API 基础URL"
                )
                
                embedding_model = gr.Textbox(
                    label="嵌入模型名称",
                    value=EMBEDDING_MODEL,
                    placeholder="text-embedding-3-small",
                    info="模型名称，如 text-embedding-3-small"
                )
                
                embedding_dimension = gr.Number(
                    label="向量维度",
                    value=EMBEDDING_DIMENSION,
                    precision=0,
                    info="模型的输出维度（如 1536, 3072）"
                )
            
            with gr.Column(scale=2):
                with gr.Tab("📤 上传文档"):
                    gr.Markdown("### 上传文件到向量数据库")
                    
                    file_input = gr.File(
                        label="选择文件",
                        file_count="multiple",
                        file_types=[".txt", ".md", ".doc", ".docx", ".xlsx", ".pdf", ".jpg", ".jpeg", ".png", ".bmp"]
                    )
                    
                    with gr.Row():
                        max_chars = gr.Slider(
                            minimum=100,
                            maximum=5000,
                            value=1000,
                            step=100,
                            label="每块最大字符数"
                        )
                        overlap = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=100,
                            step=50,
                            label="重叠字符数"
                        )
                    
                    upload_btn = gr.Button("🚀 开始上传", variant="primary", size="lg")
                    upload_output = gr.Textbox(
                        label="上传结果",
                        lines=15,
                        max_lines=30,
                        interactive=False
                    )
                
                with gr.Tab("🔍 搜索文档"):
                    gr.Markdown("### 语义搜索")
                    
                    query_input = gr.Textbox(
                        label="搜索内容",
                        placeholder="输入您想搜索的内容...",
                        lines=3
                    )
                    
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="返回结果数量"
                    )
                    
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                    search_output = gr.Textbox(
                        label="搜索结果",
                        lines=20,
                        max_lines=40,
                        interactive=False
                    )
                
                with gr.Tab("🗑️ 管理"):
                    gr.Markdown("### 命名空间管理")
                    gr.Markdown("""
                    **说明:** 下拉框只显示包含向量的命名空间。如果看不到某个命名空间，可能是因为：
                    1. 该命名空间中没有上传过文档
                    2. 该命名空间中的向量已被删除
                    """)
                    
                    clear_namespace_dropdown = gr.Dropdown(
                        choices=initial_namespaces,
                        value=initial_namespaces[0] if initial_namespaces else "default",
                        label="要清空的命名空间",
                        interactive=True
                    )
                    
                    clear_btn = gr.Button("🗑️ 清空命名空间", variant="stop")
                    clear_output = gr.Textbox(label="操作结果", interactive=False)
                
                with gr.Tab("📄 PDF转Markdown"):
                    gr.Markdown("### PDF转Markdown（使用OCR）")
                    
                    pdf_input = gr.File(
                        label="选择PDF文件",
                        file_types=[".pdf"]
                    )
                    
                    gr.Markdown("""
                    **使用说明:**
                    1. 确保Ollama已安装并运行
                    2. 确保已安装所选模型: `ollama pull {model}`
                    3. 上传PDF文件
                    4. 点击转换按钮
                    5. 转换后的Markdown会自动保存到docs文件夹
                    
                    **模型说明:**
                    - **my-PaddleOCR-VL:0.9b**: PaddleOCR-VL-1.5，文档识别效果更好的模型
                    - **my-glm-ocr:latest**: GLM-OCR，通用OCR模型
                    """.format(model=ocr_model.value if hasattr(ocr_model, 'value') else "my-glm-ocr:latest"))
                    
                    convert_btn = gr.Button("🔄 开始转换", variant="primary", size="lg")
                    
                    markdown_output = gr.Textbox(
                        label="转换结果（Markdown格式）",
                        lines=25,
                        max_lines=50,
                        interactive=False
                    )
        
        # 事件绑定
        refresh_btn.click(
            fn=refresh_namespaces,
            inputs=[index_dropdown],
            outputs=[namespace_dropdown]
        )
        
        refresh_btn.click(
            fn=refresh_namespaces,
            inputs=[index_dropdown],
            outputs=[clear_namespace_dropdown]
        )
        
        index_dropdown.change(
            fn=refresh_namespaces,
            inputs=[index_dropdown],
            outputs=[namespace_dropdown]
        )
        
        index_dropdown.change(
            fn=refresh_namespaces,
            inputs=[index_dropdown],
            outputs=[clear_namespace_dropdown]
        )
        
        upload_btn.click(
            fn=upload_files,
            inputs=[file_input, index_dropdown, custom_namespace, max_chars, overlap, ollama_url, ocr_model,
                   embedding_api_key, embedding_base_url, embedding_model, embedding_dimension],
            outputs=[upload_output]
        )
        
        search_btn.click(
            fn=search_documents,
            inputs=[query_input, index_dropdown, custom_namespace, top_k,
                   embedding_api_key, embedding_base_url, embedding_model],
            outputs=[search_output]
        )
        
        clear_btn.click(
            fn=clear_namespace,
            inputs=[index_dropdown, clear_namespace_dropdown],
            outputs=[clear_output]
        )
        
        convert_btn.click(
            fn=convert_pdf_to_markdown,
            inputs=[pdf_input, ollama_url, ocr_model],
            outputs=[markdown_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
