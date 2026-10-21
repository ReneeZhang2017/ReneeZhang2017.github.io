# Renee’s Hugo Blog

极简个人博客，版式参考 Geoff Ruddock 的博客。使用 Hugo 生成静态网页、GitHub Actions 自动发布到 GitHub Pages。无需购买主题，无第三方主题依赖。

## 第一次上线（只做一次）

1. 在 GitHub 新建一个 **Public** 仓库，仓库名建议用 `你的GitHub用户名.github.io`。该命名用于个人主页。默认分支用 `main`。
2. 解压项目。在仓库中通过 **Add file → Upload files** 上传项目内容（不是最外层文件夹）。确保 `hugo.toml`、`content`、`layouts`、`static` 都在仓库根目录。
3. **特别注意 `.github/workflows/hugo.yaml`**：以点开头的文件夹可能在文件管理器里隐藏，浏览器上传也可能漏掉。最稳妥的方式是在 GitHub 点击 **Add file → Create new file**，文件名填写 `.github/workflows/hugo.yaml`，再粘贴本项目对应文件的全部内容并保存。Mac 上可以用 `Command + Shift + .` 显示隐藏文件。
4. 打开仓库 **Settings → Pages → Build and deployment → Source**，选 **GitHub Actions**。
5. 进入 **Actions → Build and deploy Hugo blog → Run workflow → main → Run workflow**。之后每次保存到 `main`，都会自动触发发布。
6. 等任务变绿，在 **Settings → Pages** 查看网址。首次发布可能需要几分钟。
7. 将 `hugo.toml` 中的 `baseURL` 改为你的实际网站网址（结尾保留 `/`），保存。自动发布时工作流也会使用 GitHub 返回的真实网址。

你也可以用普通仓库名（例如 `blog`）；工作流支持带 `/blog/` 的项目路径。

**公开仓库的源码也是公开的。** `draft: true` 仅阻止文章生成到网站，不会隐藏仓库里的文字。未公开内容请先放在自己的电脑上。

## 添加一篇文章（全部在 GitHub 网页完成）

1. 打开仓库，点击 **Add file → Create new file**。
2. 文件名填写 `content/blog/my-new-post.md`。文件名建议英文、数字、短横线，标题可用中文。
3. 写入以下内容，替换标题、日期和正文：

```markdown
---
title: "如何评估倾向得分模型"
date: 2026-10-21
draft: false
description: "从 overlap、balance 和 extreme weights 三方面检查。"
categories: ["Causal Inference"]
---

这里写开头。

## 为什么要检查

这里写正文。

- 第一条
- 第二条
```

4. 点击 **Commit changes**，选择直接提交到 `main`，确认保存。
5. 在 **Actions** 等待发布成功，再打开网站。文章会自动出现在首页，按日期排序并按年份分组。

不需要改 HTML、CSS 或文章列表。`date` 不要设置成未来日期，否则 Hugo 默认会暂不发布。

## 修改、撤回与删除

- **修改**：打开对应 `.md` 文件，点铅笔图标编辑，再 **Commit changes**。
- **暂时撤回**：把 `draft: false` 改成 `draft: true`，保存后等待发布。旧页面会从新构建中移除，但缓存或搜索结果可能暂时保留。
- **笔记**：新增到 `content/notes/`，操作完全相同。
- **个人介绍**：编辑 `content/about.md`。
- **项目介绍**：编辑 `content/projects.md`。
- **网站姓名和简介**：编辑 `hugo.toml` 的 `title`、`params.author` 和 `params.description`。

## 插入图片

推荐每篇有图片的文章使用一个文件夹：

```text
content/blog/my-project/index.md
content/blog/my-project/chart.png
```

在 `index.md` 中写 `![图表说明](chart.png)`。将图片与文章放在一起，不需要记住网站域名，在项目子路径下也能显示。

也可以统一上传到 `static/images/`，但在项目站点下要在 Markdown 中包含正确的仓库子路径；推荐上面的文章文件夹方式。

## 文章模板

`content/blog/my-first-post.md` 和 `content/notes/my-first-note.md` 是草稿模板，默认不会出现在正式网站。替换正文并设 `draft: false` 即可发布。

`content/blog/welcome.md` 是当前唯一的公开示例文章，你可以修改或删除。

## 可选：本地预览

先按照 https://gohugo.io/installation/ 安装 Hugo。项目锁定版本 `0.168.0`，GitHub 自动发布也使用此版本。

在项目目录运行：

```bash
hugo server -D
```

浏览器打开终端提示的本地网址。`-D` 会显示草稿；正式发布不包含草稿。

创建文章也可以使用：

```bash
hugo new content/blog/my-new-post.md
```

## 费用与维护

Hugo 免费；GitHub Pages 对免费账号的公开仓库可用，受其使用限制约束。独立域名可选，需要自行购买和续费。此项目不需要 API 密钥、数据库或付费主题。

如果发布失败，先看 GitHub 的 Actions 日志。未来升级 Hugo 或 Actions 时，先验证构建再修改版本号。

参考：
- https://gohugo.io/host-and-deploy/deploy-to-github-pages/
- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages

## 顶部导航与文章分类

- **ALL blogs**：汇总 `content/blog/` 和 `content/notes/` 中的已发布文章，按日期从新到旧显示。
- **Causal Inference**、**Model**、**Statistics**：按照文章开头的 `categories` 自动筛选。

例如，在文章最上方两条 `---` 之间加一行：

```yaml
categories: ["Causal Inference"]
```

分类名称请保持为 `Causal Inference`、`Model` 或 `Statistics`。
文章可以同时属于多个分类：

```yaml
categories: ["Causal Inference", "Statistics"]
```

未设置分类的文章仍显示在 ALL blogs。个人介绍与项目说明不计入文章列表，原链接保留。
