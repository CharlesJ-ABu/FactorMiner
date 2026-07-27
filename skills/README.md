# FactorMiner Skills

本目录收录可以独立使用、并能在 FactorMiner 仓库中渐进增强的 Agent Skills。

## 可用 Skills

### FactorMiner 因子研究设计师

路径：[`factorminer-research-architect/`](factorminer-research-architect/)

把交易直觉、公式、代码或现有实验转化为可执行、可审查、可复现的因子研究任务卡。

- **独立研究模式**：不要求安装 FactorMiner，输出框架无关的假设、特征、标签、数据切分、评价和泄漏审查方案。
- **FactorMiner 增强模式**：检测到兼容仓库后，进一步检查现有算子和扩展点，生成 `user_workspace` 配置或扩展，执行真实实验并使用 Inspector 复查。

调用示例：

```text
使用 $factorminer-research-architect：
我认为放量突破后，短期价格会延续。请把它设计成一个严谨的因子实验。
```

## 安装

克隆仓库后，将 Skill 目录复制到 Codex Skills 目录：

```bash
git clone https://github.com/CharlesJ-ABu/FactorMiner.git
cd FactorMiner
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/factorminer-research-architect "${CODEX_HOME:-$HOME/.codex}/skills/"
```

重新启动 Codex 或开启新任务后，可这样调用：

```text
使用 $factorminer-research-architect：
我认为放量突破后，短期价格会延续。请把它设计成一个严谨的因子实验。
```

### 更新

先在 FactorMiner 仓库执行 `git pull`，再替换已安装目录：

```bash
rm -rf "${CODEX_HOME:-$HOME/.codex}/skills/factorminer-research-architect"
cp -R skills/factorminer-research-architect "${CODEX_HOME:-$HOME/.codex}/skills/"
```

### 卸载

```bash
rm -rf "${CODEX_HOME:-$HOME/.codex}/skills/factorminer-research-architect"
```

也可以不安装，直接让 Agent 从仓库路径读取 `SKILL.md`。

## 兼容范围与运行边界

- **独立研究模式**：不依赖 FactorMiner、行情数据或 FactorMiner Python 环境；输出框架无关的研究任务卡，不声称已经运行实验。
- **FactorMiner 增强模式**：需要可验证的 FactorMiner 仓库、本地数据和项目依赖。配置化 forward-return 标签要求使用包含统一 Target Builder 的版本，基线为提交 `f97e249` 或后续版本。
- 当前 Skill 按 Codex `SKILL.md` 目录格式维护；其他 Agent 可以直接读取 Markdown，但其 Skill 发现和 UI 元数据兼容性取决于各自产品。
- 增强模式默认只在 `user_workspace/` 下生成配置或扩展；不会因为检测到仓库就自动运行昂贵实验。
- 真实执行前应检查依赖、数据覆盖、成本和授权范围。Skill 不包含行情、API 密钥或真实实验指标。
- Skill 本身不主动上传本地行情；实际上下文和文件如何处理取决于用户选择的 Agent、模型和运行环境。

## 发布验收

首个 Skill 已完成以下验证：

- 独立模式：从“放量突破后短期延续”的自然语言直觉生成完整、框架无关的研究任务卡；
- FactorMiner 增强模式：在最新 `main` 上使用公开 CLI 完成配置、受约束扩展、真实挖掘、快照和标准 Inspector 闭环；
- 标签一致性：`target`、元数据、parquet 快照和 Inspector 使用同一 forward-return 公式；
- 最小扩展：Target Builder 已支持的 3-bar next-open 标签不会生成自定义标签或 Inspector 适配器；
- 回归验证：提交 `f97e249` 对应测试集为 60/60 通过，Target 专项测试为 6/6 通过。

这些结果证明执行链路和研究约束按设计工作，不代表任何因子有效、可交易或能够获得未来收益。

## 贡献约定

- 每个 Skill 使用独立的短横线命名目录，并包含 `SKILL.md`。
- `SKILL.md` frontmatter 只包含 `name` 和 `description`。
- 详细方法和版本化接口放在 `references/`，输出模板放在 `assets/`。
- FactorMiner 适配内容必须以当前仓库代码为准。
- 不提交虚构指标、真实密钥、私有数据或本地实验产物。
- 提交前运行 `python skills/validate.py`。
