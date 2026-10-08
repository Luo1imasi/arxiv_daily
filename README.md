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

安装（CPU 版 PyTorch，无需 CUDA）：

```bash
# 1. 先安装 CPU 版 PyTorch（约 200MB）
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu

# 2. 再安装项目及其余依赖
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

Web 页面里的运行任务、回填、保存配置、测试连接和重载 corpus 等修改性操作需要输入操作密码。默认密码在 `server.admin_password` 中配置，也可以通过环境变量 `ARXIV_DAILY_ADMIN_PASSWORD` 覆盖。

## 收藏到 Zotero

在 Settings 中填写 Zotero User ID、具有个人库读写权限的 API Key，以及可选的
Collection Key，然后点击推荐论文上的“加入 Zotero”。User ID 和 API Key 可在
https://www.zotero.org/settings/keys 获取。也可以配置：

```yaml
zotero:
  user_id: "1234567"
  api_key: "your-api-key"
  collection_key: ""  # 可选的八位 collection key
```

API Key 和 User ID 也可通过 `ARXIV_DAILY_ZOTERO_API_KEY` 和
`ARXIV_DAILY_ZOTERO_USER_ID` 环境变量设置，环境变量优先。

收藏会创建论文及 PDF 附件条目，并将 ZIP/PROP 附件保存到现有 WebDAV 来源。
设置了 `webdav.local_path` 时服务进程需要该目录的写权限；否则使用 HTTPS WebDAV
上传，目标目录须已存在。Zotero 客户端需要启用同一账号的数据同步和同一 WebDAV
的文件同步。这一流程只支持个人库。

成功收藏的论文立即进入 corpus，并在重载后保留。失败时点击“重试收藏”可继续，
已经创建的条目会复用。收藏状态保存在 arXiv Daily 数据目录中。

## 首页组件

`GET /api/widget` 提供最新推荐日期、推荐数量和排名靠前的论文标题、简短 TLDR、
评分与详情链接。默认返回 3 篇，`limit` 可设为 1 到 5，不需要操作密码。
浏览器跨域读取仅允许 `server.widget_allowed_origins` 中配置的网站；默认允许
`luolimasi.xyz`、`luolimasi.com` 及其 `www` 域名。响应可缓存 60 秒。

主页终端使用只读的 `GET /api/terminal`：默认返回最新一期前 5 篇推荐；`date`
可指定推荐日期，`q` 可搜索历史推荐的标题、作者、摘要和 TLDR，`limit` 为 1 到 10。
返回最近 30 个有推荐的日期和论文解读，不包含配置、收藏状态或反馈记录。
终端支持 `papers`、`papers search <关键词>`、`papers dates`、`papers date YYYY-MM-DD`、
`paper <序号>`、`papers open <序号>`、`papers pdf <序号>` 和 `papers refresh`。

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
# 1. 先安装 CPU 版 PyTorch（约 200MB）
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu

# 2. 再安装项目及开发依赖
pip install -e ".[dev]"
```

运行测试：

```bash
pytest
```

运行检查：

```bash
ruff check .
```
