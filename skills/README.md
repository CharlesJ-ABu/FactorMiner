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

将所需 Skill 目录复制或链接到 Codex Skills 目录：

```bash
cp -R skills/factorminer-research-architect ~/.codex/skills/
```

也可以直接让 Agent 从仓库路径读取 `SKILL.md`。Skill 的基础研究能力不依赖 FactorMiner；只有增强模式需要本地项目、数据和运行环境。

## 贡献约定

- 每个 Skill 使用独立的短横线命名目录，并包含 `SKILL.md`。
- `SKILL.md` frontmatter 只包含 `name` 和 `description`。
- 详细方法和版本化接口放在 `references/`，输出模板放在 `assets/`。
- FactorMiner 适配内容必须以当前仓库代码为准。
- 不提交虚构指标、真实密钥、私有数据或本地实验产物。
- 提交前运行 `python skills/validate.py`。

`skills/` 是可发布 Skill 的唯一维护来源；商业策划和发布运营材料不在这里维护。
