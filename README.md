# arxiv-daily

一个基于本地论文库或 WebDAV 目录的 arXiv 每日推荐工具。它会从你的已有论文收藏中提取兴趣信号，拉取近期 arXiv 候选论文，完成检索、重排，并通过一个简单的 Web 页面展示每日推荐结果。

## 适合什么场景

如果你已经有自己的论文库，希望每天自动看到“更可能值得读”的新论文，而不是手动刷 arXiv，这个项目就是为这个场景准备的。

它支持：

- 本地目录或 WebDAV 作为 corpus 来源
- 基于已有论文内容提取兴趣关键词
- 结合 BM25 和本地 reranker 做排序
- 可选使用 LLM 生成 TLDR
- FastAPI Web 界面和定时任务

## 快速开始

安装：

```bash
pip install -e .
```

复制配置模板：

```bash
cp config/custom.example.yaml config/custom.yaml
```

至少配置一种论文来源：

- `webdav.local_path`
- 或 `webdav.url` / `webdav.username` / `webdav.password` / `webdav.path`

启动服务：

```bash
arxiv-daily
```

然后访问：

```text
http://127.0.0.1:5555
```

如果没有配置 `llm.api_key`，项目仍然可以正常运行，只是不会生成 TLDR。

## 配置

项目会把 `config/default.yaml` 和本地 `config/custom.yaml` 合并使用。通常你只需要在 `custom.yaml` 里覆盖少数字段。

最常用的配置项包括：

- `webdav.local_path`
- `webdav.url/username/password/path`
- `executor.schedule_hour/schedule_minute/timezone`
- `executor.max_paper_num`
- `source.arxiv.recent_days`
- `source.arxiv.max_recent_days`
- `llm.api_key`
- `llm.base_url` / `llm.model`
- `server.host/port`

更完整的参数说明可以直接看 `config/default.yaml`。

## 项目结构

```text
config/                配置模板和默认配置
src/arxiv_daily/       核心实现
tests/                 测试
```

其中几个核心模块：

- `main.py`：Web 服务入口和 API
- `executor.py`：推荐流程编排
- `webdav.py`：本地/WebDAV corpus 扫描
- `retriever/`：候选检索
- `reranker/`：候选重排

## 开发

安装开发依赖：

```bash
pip install -e .[dev]
```

运行测试：

```bash
pytest
```

运行检查：

```bash
ruff check .
```
