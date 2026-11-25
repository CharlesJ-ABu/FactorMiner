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

## 验证与使用边界

Skill 的静态结构可通过 `python skills/validate.py` 检查。FactorMiner 的安装与产品测试参照[产品指南](../docs/FactorMiner_Product_Report_ZH.md)，当前支持本机单用户源码 checkout；wheel 本身不包含完整工作区与前端。

独立模式输出研究设计，不声称已经执行。增强模式须核对实际源码、行情、配置、标签和产物；配置化标签与窗口应一致，测试期参与选择时不能继续称为未见 holdout。历史运行记录或静态校验通过都不证明因子有效、可交易或未来收益。

## 贡献约定

- 每个 Skill 使用独立的短横线命名目录，并包含 `SKILL.md`。
- `SKILL.md` frontmatter 只包含 `name` 和 `description`。
- 详细方法和版本化接口放在 `references/`，输出模板放在 `assets/`。
- FactorMiner 适配内容必须以当前仓库代码为准。
- 不提交虚构指标、真实密钥、私有数据或本地实验产物。
- 提交前运行 `python skills/validate.py`。
