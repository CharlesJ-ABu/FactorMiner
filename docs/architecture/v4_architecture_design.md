# FactorMiner V4 架构设计：统一因子挖掘范式

## 核心设计理念

为了兼容 GP、RL、NN、LLM 四种因子挖掘范式，FactorMiner V4 将“挖掘”动作高度抽象为一个通用的**“搜索与优化”闭环过程**：`生成候选 -> 回测评价 -> 模型更新 -> 再生成`。

为了解决“不同流派特性差异巨大”带来的工程落地挑战（如类型污染、NN 端到端评测闭环差异等），V4 架构引入了统一的表达式抽象和谱系追踪机制。**在终端对客层面，系统将所有底层流派完全封装，通过唯一的统一网关 `FactorMinerDirector` 提供纯声明式的策略驱动体验。**

---

## 1. 统一中间体抽象 (FactorExpression)

为了避免 `candidates: List[Any]` 造成的类型污染和 Evaluator 中大量的 `if isinstance` 判断，引入 `FactorExpression` 协议，统一封装底层逻辑。

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Union, Optional

class FactorExpression(ABC):
    """
    统一的因子中间体表达式
    不论生成器是 GP、RL、NN 还是 LLM，最终吐出的都是该类的子类实例。
    """
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame, Any]:
        """
        统一计算接口。
        - AST派生类: 在这里遍历解析树并计算
        - Code派生类: 在这里使用 exec/eval 计算
        - DL派生类: 在这里执行 model.forward(data)
        """
        pass
        
    @abstractmethod
    def get_source(self) -> Any:
        """
        获取底层真正的逻辑表达（比如 ast dict，code 字符串，或 nn.Module）
        """
        pass

class LineageTrackableMixIn:
    """可追溯谱系的能力标签"""
    def get_lineage_parents(self) -> List[str]:
        return getattr(self, '_parent_ids', [])

class LLMReflectableMixIn:
    """具有大模型反思记忆的能力标签"""
    def get_reflection_history(self) -> str:
        return getattr(self, '_reflection_history', "")
```

---

## 2. 核心抽象接口设计 (BaseMiner)

为了彻底解决“NN 范式评价断层”、“大模型同质化”以及“演进状态丢失”等架构痛点，V4 引入了统一的评测反馈结构 `EvaluationFeedback` 与 `DiversityFilter`。

```python
from dataclasses import dataclass
import torch
import hashlib

@dataclass
class EvaluationFeedback:
    """统一的评测反馈体，兼容标量指标和张量计算图"""
    metrics: List[Dict[str, float]]          # 传统指标（IC, Sharpe等）
    raw_outputs: Optional[torch.Tensor] = None # DL流派前向传播直接吐出的 Tensor
    raw_targets: Optional[torch.Tensor] = None # 用于 DL 计算 Loss 的配置化未来收益标签

class DiversityFilter:
    """因子正交性与多样性过滤器"""
    def __init__(self, correlation_threshold: float = 0.7):
        self.threshold = correlation_threshold
        self.archive_hashes = set()

    def filter_redundant(self, candidates: List[FactorExpression], data: pd.DataFrame) -> List[FactorExpression]:
        unique_candidates = []
        for cand in candidates:
            # 1. 语法/语义层面的硬去重（通过 Hash 机制）
            expr_hash = hashlib.md5(str(cand.get_source()).encode()).hexdigest()
            if expr_hash in self.archive_hashes:
                continue
                
            # 2. 截面相关性软去重（可结合回测引擎动态计算）
            
            self.archive_hashes.add(expr_hash)
            unique_candidates.append(cand)
            
        return unique_candidates

class BaseFactorMiner(ABC):
    """
    因子挖掘的通用基类范式
    适用于 GP (遗传), RL (强化学习), NN (神经网络), LLM (大模型)
    """
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data           
        self.config = config       
        self.history = []          
        
    @abstractmethod
    def initialize_search_space(self) -> None:
        """
        1. 初始化搜索空间 / 环境
        """
        pass

    @abstractmethod
    def generate_candidates(self) -> List[FactorExpression]:
        """
        2. 生成候选因子 (Propose)
        @return: 严格返回 FactorExpression 对象的列表
        """
        pass

    @abstractmethod
    def evaluate_candidates(self, candidates: List[FactorExpression]) -> 'EvaluationFeedback':
        """
        3. 评价与打分 (Evaluate / Reward)
        - GP/LLM/RL: 返回传统的指标（IC、夏普等）。
        - NN: 携带计算图 Tensor 原样返回，供 update_model 梯度计算。
        """
        pass

    @abstractmethod
    def update_model(self, candidates: List[FactorExpression], feedback: 'EvaluationFeedback') -> None:
        """
        4. 反馈与模型更新 (Feedback & Learn)
        - GP: 更新种群 (self.state.population = new_trees)
        - LLM: 更新反思记忆 (self.state.failed_reflections.append(...))
        - NN: 执行反向传播 (loss.backward(); optimizer.step())
        """
        pass

    def mine(self, n_iterations: int, progress_callback=None) -> List[FactorExpression]:
        """
        5. 标准的主循环引擎 (Main Loop) - 集成去重拦截器
        """
        self.initialize_search_space()
        div_filter = DiversityFilter(self.config.get('max_corr', 0.7))
        
        for epoch in range(n_iterations):
            # 1. 提案
            raw_candidates = self.generate_candidates()
            
            # 1.5 拦截过滤：干掉高相关性/重复因子 (结合 FactorStorage 的全局历史 Hash 进行硬去重免疫)
            candidates = div_filter.filter_redundant(raw_candidates, self.data)
            if not candidates: continue
            
            # 2. 判卷 (集成健壮性告警屏蔽，如 GP 膨胀产生的常数无效因子)
            feedback = self.evaluate_candidates(candidates)
            
            # 3. 学习：update_model 负责更新内部的 self.state 或模型参数
            self.update_model(candidates, feedback)
            
            self._log_epoch(epoch, candidates, feedback.metrics)
            
        return self._get_best_factors()
        
    def _log_epoch(self, epoch: int, candidates: List[FactorExpression], metrics: List[Dict[str, float]]):
        pass
        
    def _get_best_factors(self) -> List[FactorExpression]:
        pass
```

---

## 3. 标准化存储与谱系追踪 (FactorStorage)

由于四个流派在生成因子的“本质形态”上存在巨大差异，如果我们只用粗暴的二进制（Blob）存储，因子的可读性、回溯性和实盘复现性就会彻底崩溃。在 FactorMiner V4 中，系统通过 `FactorMetadata` 承载因子的谱系灵魂，并通过统一的存储接口，对不同流派采用不同的底层异构落地策略，最终进行标准化的肉体存储。

### 3.1 谱系与元数据定义 (FactorMetadata)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class FactorMetadata:
    factor_id: str                   # 因子唯一全球标识 (例如: alp_llm_20260710_001)
    miner_type: str                  # 流派标签: 'GP', 'RL', 'NN', 'LLM'
    user_id: str                     # 提交/运行该任务的研究员ID
    
    # 【新增】因子生命周期状态机
    # 状态流转: DISCOVERED (刚挖出) -> INSPECTED (已人工审查) -> PAPER_TRADING (模拟盘) -> LIVE (实盘) -> RETIRED (衰减退役)
    lifecycle_status: str = field(default="DISCOVERED") 
    
    # 谱系追踪 (Lineage): 支撑因子的达尔文演进或反思链条
    parent_ids: List[str] = field(default_factory=list) # 遗传变异的父代ID，或LLM上一版报错的代码ID
    generation_config: Dict[str, Any] = field(default_factory=dict) # 产生该因子时的超参数快照
    
    # 灵魂索引 (Soul References): 不同流派指向其逻辑核心的钥匙
    logic_reference: Dict[str, Any] = field(default_factory=dict) 
    # 示例: 
    # GP -> {"storage_type": "json_ast", "path": "path/to/ast.json"}
    # NN -> {"storage_type": "model_channel", "model_version": "v4.1", "channel_idx": 42}
    
    # 评测指标 (肉体成绩单)
    metrics: Dict[str, float] = field(default_factory=dict) # 留存当时的评估指标 (IC, IR, Sharpe等)
    created_at: str = ""
```

### 3.2 异构化存储接口实现 (FactorStorageInterface)

针对四大流派的特性，存储接口在落地层面的具体做法如下：

```python
from abc import ABC, abstractmethod
import pandas as pd

class FactorStorageInterface(ABC):
    
    # ==============================================================
    # 层面一：逻辑与谱系存储 (存灵魂) -> 针对四大流派异构化实现
    # ==============================================================
    
    @abstractmethod
    def save_gp_factor(self, ast_dict: Dict, metadata: FactorMetadata) -> bool:
        """
        【GP 流派存储方案】
        - 做法：将不可导的公式树直接序列化为标准的 JSON AST 格式存储（存入 NoSQL 或文件系统）。
        - 优势：极度轻量，完美支持跨平台复现。生产实盘引擎可以直接读取 JSON 树并用 C++ 表达式解析器高效复现。
        """
        pass
        
    @abstractmethod
    def save_rl_factor(self, action_sequence: List[str], agent_snapshot_bytes: bytes, metadata: FactorMetadata) -> bool:
        """
        【RL 流派存储方案】
        - 做法：存储两部分。
          1. 最终翻译出来的显式数学公式代码字符串或 Action 序列（用于可解释性分析）。
          2. 产生该轨迹的 Policy 网络当前 Epoch 的权重快照（用于增量强化学习或行为回放）。
        """
        pass
        
    @abstractmethod
    def save_llm_factor(self, python_code: str, reflection_log: str, metadata: FactorMetadata) -> bool:
        """
        【LLM 流派存储方案】
        - 做法：
          1. 存储经过沙盒验证、可单独运行的纯 Python 源代码脚本（以 .py 文本资产存储）。
          2. 将大模型生成该因子时的“思考、反思、报错日志（Reflection Log）”一并存入数据库。
        - 优势：研究员可以直接在线 review 代码，甚至可以直接把生成的 py 文件挂载到实盘定时任务中。
        """
        pass
        
    @abstractmethod
    def save_model_weights(self, model_version_id: str, model_weights: bytes) -> bool:
        """
        动作一：独立上传模型权重大文件。
        【全局单次调用】内部做判重，如果已存在则跳过。
        """
        pass

    @abstractmethod
    def save_nn_factor_channel(self, model_version_id: str, channel_index: int, metadata: FactorMetadata) -> bool:
        """
        动作二：极其轻量级的通道索引存储。
        【For 循环内调用】每次只传字符串版本号和索引 int，数据量仅几百字节，无网络压力。
        """
        pass

    # ==============================================================
    # 层面二：数据存储 (存肉体) -> 全流派标准化统一
    # ==============================================================
    
    @abstractmethod
    def save_factor_values(self, factor_id: str, values_df: pd.DataFrame) -> bool:
        """
        【全流派标准化数据落地】
        - 做法：不管灵魂是代码还是模型，它们计算出来的历史矩阵（肉体）必须强制标准化为 **Parquet** 格式进行列式存储。
        - 规范：`values_df` 必须严格遵循截面矩阵标准（Index 为 Datetime，Columns 为 StockCode，Values 为因子截面得分）。
        - 目的：下游的组合优化器（Portfolio Optimizer）和执行算法不需要关心因子是怎么挖掘出来的，直接以极高的 I/O 速度读取 Parquet 矩阵，进行因子权重分配和线性组合。
        """
        pass
```

### 3.3 终极联动：完整用户脚本中的存储落地演示

当研究员配置并运行完任务后，系统是如何在底层无缝处理这些异构存储的？我们以大模型（LLM）流派在 `mine()` 循环最后的落地为例：

```python
# 当用户调用：best_alphas = director.run(n_iterations=50) 
# 底层 Miner 完成挖掘后，自动触发的存储流水线伪代码：

class LLMFactorMiner(BaseFactorMiner):
    def __init__(self, data, config, fitness_calculator, storage_client: FactorStorageInterface):
        super().__init__(data, config)
        self.storage = storage_client
        
    def _save_discovered_alphas(self, best_candidates: List[FactorExpression], final_feedback: EvaluationFeedback):
        for idx, cand in enumerate(best_candidates):
            metrics = final_feedback.metrics[idx]
            factor_id = f"alp_llm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"
            
            # 1. 组装谱系灵魂元数据 (结合 MixIn 多态安全检查)
            parent_ids = cand.get_lineage_parents() if isinstance(cand, LineageTrackableMixIn) else []
            metadata = FactorMetadata(
                factor_id=factor_id,
                miner_type="LLM",
                user_id=self.config.get("researcher_id", "system"),
                parent_ids=parent_ids, 
                generation_config=self.config,
                metrics=metrics,
                created_at=pd.Timestamp.now().isoformat()
            )
            
            # 2. 存灵魂：调用异构接口存储纯 Python 源代码和反思日志
            reflection_log = cand.get_reflection_history() if isinstance(cand, LLMReflectableMixIn) else ""
            self.storage.save_llm_factor(
                python_code=cand.get_source(), 
                reflection_log=reflection_log,
                metadata=metadata
            )
            
            # 3. 存肉体：就地利用当前行情数据计算出全历史矩阵，标准化存为 Parquet
            factor_matrix = cand.compute(self.data)
            self.storage.save_factor_values(factor_id=factor_id, values_df=factor_matrix)
            
            print(f"成功将因子 {factor_id} 存入量化中台（灵魂: 脚本代码, 肉体: Parquet矩阵）")
```

---

## 4. 并行评测引擎与去重缓存 (ParallelEvaluator)

在实际生产中，`evaluate_candidates` 不能是简单的 for 循环。GP 会产生数万个 AST，LLM 评估要等网络 I/O。我们需要一个异步/多进程的分发器，并且要支持“表达式去重与缓存”（避免重复计算相同因子的面值）。

引入一个单例的 `EvaluationEngine`，它内部维护一个计算缓存和多进程/分布式池（如使用 Ray 或 Celery）。

```python
import hashlib
import ray

@ray.remote
def remote_compute_and_evaluate(expression_source: Any, data_ref: ray.ObjectRef, eval_func_bytes: bytes) -> Dict[str, float]:
    """在 Ray 集群节点上并行计算因子并回测"""
    # 1. 恢复回测函数
    import cloudpickle
    eval_func = cloudpickle.loads(eval_func_bytes)
    # 2. 获取共享内存中的行情数据
    df = ray.get(data_ref)
    
    # 3. 计算面值 (假设表达式对象已经被序列化传过来)
    # 真实落地时，这里要包在 try-except 中防止计算报错导致进程崩溃
    try:
        factor_values = expression_source.compute(df)
        metrics = eval_func(factor_values, df)
        return metrics
    except Exception as e:
        return {"IC": 0.0, "IR": 0.0, "error": 1.0}

class ParallelEvaluator:
    def __init__(self, data: pd.DataFrame, eval_metric_func):
        # 将大数据放入 Ray 共享内存，避免多进程间重复复制数据
        self.data_ref = ray.put(data)
        import cloudpickle
        self.eval_func_bytes = cloudpickle.dumps(eval_metric_func)
        self.computed_cache = {} # 表达式 Hash -> Metrics

    def evaluate_batch(self, candidates: List[FactorExpression]) -> List[Dict[str, float]]:
        futures = []
        task_mapping = []
        
        for idx, cand in enumerate(candidates):
            # 对因子底层逻辑生成唯一的语义明文 Hash（防止换个变量名重复计算）
            expr_hash = hashlib.md5(str(cand.get_source()).encode()).hexdigest()
            
            if expr_hash in self.computed_cache:
                # 缓存命中，直接记录
                task_mapping.append((idx, self.computed_cache[expr_hash]))
            else:
                # 缓存未命中，提交异步计算
                future = remote_compute_and_evaluate.remote(cand, self.data_ref, self.eval_func_bytes)
                futures.append(future)
                task_mapping.append((idx, expr_hash)) # 记录未来需要回填的位置
        
        # 阻塞等待所有远程计算完成
        remote_results = ray.get(futures)
        
        # 组装结果并更新缓存
        final_results = [None] * len(candidates)
        remote_idx = 0
        for map_info in task_mapping:
            idx = map_info[0]
            if isinstance(map_info[1], dict): # 缓存命中
                final_results[idx] = map_info[1]
            else: # 远程返回
                res = remote_results[remote_idx]
                self.computed_cache[map_info[1]] = res
                final_results[idx] = res
                remote_idx += 1
        return final_results
```

---

## 5. 动态沙盒安全性 (RestrictedSandbox)

LLM 生成代码不能在 Miner、Evaluator 或 Inspector 进程中直接执行。当前实现位于
`core/evaluation/code_sandbox.py`，采用以下四层边界：

1. 使用 Python AST 白名单，只允许向量化的 pandas/NumPy 表达式和局部变量赋值；
2. 禁止 import、循环、函数/类定义、文件访问、动态解释执行、私有属性和魔术方法；
3. 在短生命周期的 `spawn` 子进程中执行，并设置墙钟超时、CPU、内存和文件描述符限制；
4. 返回前严格验证类型、索引、列、数值 dtype、有限值和输入对齐关系。

`FactorExpressionCode` 采用 fail-closed 语义：没有显式传入
`RestrictedSandbox` 时直接拒绝计算，不存在退回裸 `exec()` 的兼容路径。顺序模式要求
`factor` 是与输入索引完全一致的 `pandas.Series`；`cross_asset` 模式要求它是与所有特征
DataFrame 的时间索引和资产列完全一致的 `pandas.DataFrame`。

字符串关键字黑名单不是安全边界，也不得重新引入。新增 pandas/NumPy 能力时，应在 AST
属性白名单中逐项开放，并同时增加允许用例和逃逸负向测试。

---

## 6. 统一状态管理器 (MinerState)

在通用的循环 `update_model(candidates, evaluations)` 中，不同的流派需要完全不同的“历史记忆”交互：
- GP 需要更新当前的 Population（种群）。
- LLM 需要更新 Prompt 中的 Few-Shot Memory（成功与失败列表）。
- RL 需要更新 Replay Buffer。

为了保持 `BaseFactorMiner` 的纯净度，建议抽象出一个 `MinerState` 对象，作为各个派生类的内部存储枢纽。

```python
class MinerState:
    """
    流派状态管理器：统一承载不同算法的记忆和演进实体
    """
    def __init__(self):
        # 1. 适用于 GP
        self.population: List[FactorExpression] = []
        
        # 2. 适用于 LLM (成功和失败的经验池)
        self.successful_reflections: List[Dict] = []
        self.failed_reflections: List[Dict] = []
        
        # 3. 适用于 RL / NN
        self.model_weights_bytes: Optional[bytes] = None
        self.replay_buffer: List[Any] = []

    def get_llm_context_prompt(self) -> str:
        """给 LLM 流派快速组装 Few-shot 提示词的方法"""
        prompt = "Here are past successful examples:\n"
        for item in self.successful_reflections[-5:]: # 取最近5个
            prompt += f"Code: {item['code']}\nPerformance: {item['metrics']}\n"
        return prompt
```

---

## 7. 完整实现范例：LLMFactorMiner

结合所有补充后的完整流派实现雏形：以 LLM 派系为例。让我们看看整合了上述补充后，大模型挖掘器（`LLMFactorMiner`）在生产中是如何被极简实现的：

```python
class LLMFactorMiner(BaseFactorMiner):
    def __init__(self, data: pd.DataFrame, config: Dict):
        super().__init__(data, config)
        self.state = MinerState()
        self.sandbox = RestrictedSandbox()
        # 初始化外部专用评估引擎
        self.evaluator = ParallelEvaluator(self.data, eval_metric_func=config['metric_func'])

    def initialize_search_space(self) -> None:
        # 注入初始的基础 Prompt 指南
        self.system_prompt = "You are a top quant. Generate predictive formulas..."

    def generate_candidates(self) -> List[FactorExpression]:
        # 1. 从状态管理器中获取历史“记忆”，组装动态 Prompt
        dynamic_prompt = self.state.get_llm_context_prompt()
        
        candidates = []
        # 2. 批量调用大模型 (实际应当用异步 I/O 并行请求 API)
        # 【防御性设计】通过 API Key Pool / Token Bucket 机制，限制并发速率，防止触发云厂商限流封控
        async def _fetch_with_rate_limit():
            return await call_llm_api_with_pool(self.system_prompt, dynamic_prompt)
            
        # raw_codes = asyncio.run(batch_gather_with_limit(_fetch_with_rate_limit, self.config['batch_size']))
        raw_codes = ["mock_code_1", "mock_code_2"] # 伪代码示意
        
        # 3. 将生产的代码、沙盒和具体实现绑定到统一中间体
        for raw_code in raw_codes:
            cand = FactorExpressionCode(code_str=raw_code, sandbox=self.sandbox)
            candidates.append(cand)
        return candidates

    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        # 直接托管给高性能并行回测引擎，屏蔽底层细节
        metrics = self.evaluator.evaluate_batch(candidates)
        return EvaluationFeedback(metrics=metrics)

    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        # 收集本次迭代的反馈，存入记忆库，供下一轮 generate_candidates 组装 Prompt 使用
        for cand, eval_res in zip(candidates, feedback.metrics):
            log_entry = {"code": cand.get_source(), "metrics": eval_res}
            if eval_res.get("IC", 0) > self.config["ic_threshold"]:
                self.state.successful_reflections.append(log_entry)
            else:
                self.state.failed_reflections.append(log_entry)
```

---

## 8. 四大流派的演进逻辑与闭环落地

为了让这套多范式框架真正落地，我们将四个流派的核心思想在 FactorMiner V4 统一闭环（`initialize_search_space` -> `generate_candidates` -> `evaluate_candidates` -> `update_model`）中的具体执行步骤进行了彻底映射和归纳。

### 8.1 遗传规划 (Genetic Programming, GP)

**核心思想**: 
将数学公式视作抽象语法树 (AST)。初始随机生成大量树，保留表现好的“精英树”，通过交叉 (Crossover) 和变异 (Mutation) 繁衍下一代。这是一种典型的达尔文进化论思想。

**闭环流转**:
```text
                    [ 初始随机生成 1000 棵公式树 ]
                                  |
                                  v
                    [ 计算每棵树在历史数据上的 IC ]
                                  |
            +---------------------+---------------------+
            | (淘汰后 80%)                              | (保留前 20%)
            v                                           v
    [ 释放算力与内存 ]                          [ 进入精英保留区 ]
                                                        |
                                       +----------------+----------------+
                                       | (70% 概率)                      | (30% 概率)
                                       v                                 v
                               [ 挑选两棵树交叉 ]                 [ 随机选择单树变异 ]
                                       |                                 |
                                       +----------------+----------------+
                                                        |
                                                        v
                                          [ 组装合成 1000 棵新公式树 ]
```

**V4 接口映射**:
- **`initialize`**: 定义可用的常量范围、底层基础行情数据列（open, close）以及算子原语库。限制树的最大深度。
- **`generate`**: 从 `MinerState.population` 中读取上一代的“精英树”，通过交叉或变异来批量繁衍出一批新候选树。
- **`evaluate`**: 并行引擎反序列化 AST 树。每个节点计算出全截面时序面值，并与配置化未来收益标签计算 RankIC。
- **`update`**: 淘汰得分垫底的后 80%，将得分最高的前 20% 精英树存入 `MinerState.population`。

<details><summary>底层逻辑参考实现</summary>

```python
def genetic_programming_mining(data, num_generations=50, pop_size=1000):
    population = [random_tree_generator() for _ in range(pop_size)]
    for gen in range(num_generations):
        fitness_scores = [(tree, calculate_ic(evaluate_tree(tree, data), data['returns'])) for tree in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        elites = [x[0] for x in fitness_scores[:int(pop_size * 0.2)]]
        
        next_generation = elites.copy()
        while len(next_generation) < pop_size:
            if random() < 0.7:
                next_generation.append(crossover(random.choice(elites), random.choice(elites)))
            else:
                next_generation.append(mutate(random.choice(elites)))
        population = next_generation
    return elites
```
</details>

### 8.2 强化学习 (Reinforcement Learning, RL)

**核心思想**:
将生成公式看作一个“马尔可夫决策过程 (MDP)”。Agent 序列化地添加操作符或数据列。最终公式计算出因子的夏普比率作为 Reward，通过策略梯度（PPO/DQN）更新 Agent。

**闭环流转**:
```text
  +--------------------------------------------------------+
  |                   FactorRLAgent (Policy)               |
  +--------------------------------------------------------+
        |                                            ^
        | 动作 (Action): 选择 'ts_mean'               | 策略梯度更新
        v                                            | (PPO / DQN)
  +--------------------------------------------------------+
  |                    环境状态 (State)                     |
  |  当前前缀: "add(close, "  --> 拼接成完整的 Alpha 公式   |
  +--------------------------------------------------------+
        |
        v
  [ 计算该 Alpha 因子在回测期内的夏普比率 ] ------------> 产生奖励 (Reward)
```

**V4 接口映射**:
- **`initialize`**: 初始化智能体（如 PPO 网络）和训练环境（动作空间与状态空间）。
- **`generate`**: 从空公式开始，网络输入当前 State，输出下一个动作的概率分布。通过概率采样选出多个 Action，最后将其转换为 `FactorExpression` 列表并记录 `log_probs`。
- **`evaluate`**: 将采样出的公式送往回测引擎，计算出信息比率 (IR) 或多空组合夏普比率 (Sharpe) 作为环境赋予智能体的绝对奖励（Reward）。
- **`update`**: 提取 `log_probs` 和 Reward，计算优势函数，通过策略梯度算法执行反向传播，更新 Agent 神经网络权重。

<details><summary>底层逻辑参考实现</summary>

```python
class FactorRLAgent:
    def __init__(self):
        self.policy_network = build_neural_network()
        self.optimizer = Adam(self.policy_network.parameters())

def rl_factor_mining(data, num_episodes=1000):
    agent = FactorRLAgent()
    for episode in range(num_episodes):
        state, formula, log_probs = initial_state(), [], []
        while not is_complete(formula):
            action_probs = agent.policy_network(state)
            action = sample(action_probs)
            log_probs.append(log(action_probs[action]))
            formula.append(action)
            state = update_state(state, action)
            
        reward = calculate_sharpe(evaluate_formula(formula, data), data['returns'])
        
        loss = -sum(log_probs) * reward
        agent.optimizer.zero_grad()
    return agent
```

> **架构解耦与 PyTorch 无缝接入说明**
> 在初期的轻量级验证或轻资产环境中，用户完全可以使用纯 Python（如维护一个概率字典 `self.action_probs`）来实现 Policy Gradient 并更新模型。
> 而当需要接入笨重的深度学习网络（如真实的 PPO/DQN）时，由于 `update_model` 的完全解耦，用户只需在内部导入 PyTorch，将 Feedback 转为 Tensor Reward，然后调用标准的反向传播代码（`loss.backward(); optimizer.step()`）。底层数据流和回测引擎对这种切换完全“无感”，真正做到了算法实现与系统框架的 100% 分离。

</details>

### 8.3 大语言模型 (LLM-based Discovery)

**核心思想**:
利用大模型结合 In-context Learning 和反思机制 (Reflection)。将生成的因子及其回测表现喂给 LLM 当作上下文，让 LLM “总结失败教训”，避免重复犯错。

**闭环流转**:
```text
 +-----------------------------------------------------------+
 |                    大语言模型 (LLM API)                     |
 +-----------------------------------------------------------+
       |                                               ^
       | 1. 基于量化秘籍与历史教训                       | 3. 更新 Prompt Memory
       |    生成 Python 因子代码                        |    (成功/失败案例)
       v                                               |
 +-----------------------------------------------------------+
 |                     隔离沙箱执行与回测                      |
 |   计算因子 IC，拦截恶意命令或语法错误                         |
 +-----------------------------------------------------------+
       |
       +---> [ 成功: IC=0.06 ] ---> 提示词: "很好，请微调此思路"
       |
       +---> [ 失败: 运行报错 ] ---> 提示词: "报错由于维度不符，请修正"
```

**V4 接口映射**:
- **`initialize`**: 构筑知识库与系统 Prompt（量化方法论、因子的基本编写规范、可调用算子库文档）。
- **`generate`**: 从 `MinerState` 提取成功经验和失败教训拼接到 User Prompt 中（Few-shot）。大模型吐出 Markdown，提取纯 Python 代码，包装为沙箱安全的 `FactorExpressionCode`。
- **`evaluate`**: 将代码喂给 `RestrictedSandbox` 执行。如果报错直接将 Traceback 作为负反馈记录。如果运行成功计算 IC。所有状态均记录在返回的 `EvaluationFeedback` 中。
- **`update`**: 一步“无梯度、纯文本”的更新。将生成的代码与表现配对存入成功池或失败池。这些记忆将在下一轮改变 LLM 的上下文。

<details><summary>底层逻辑参考实现</summary>

```python
def llm_factor_mining(data, max_iterations=50):
    prompt_memory = ["你是一个顶级的量化专家。请生成代码。要求：不能高度相关。"]
    discovered_factors = []
    
    for i in range(max_iterations):
        llm_response = call_llm(messages=prompt_memory)
        factor_code = extract_code(llm_response)
        
        try:
            factor_values = execute_sandbox(factor_code, data)
            ic, icir = evaluate(factor_values, data['returns'])
            correlation = calc_correlation(factor_values, discovered_factors)
            if ic > 0.05 and correlation < 0.3:
                discovered_factors.append(factor_code)
                feedback = f"成功！IC={ic}。请继续发掘不同逻辑。"
            else:
                feedback = f"失败！相关性过高(corr={correlation})，请尝试新逻辑。"
        except Exception as e:
            feedback = f"代码执行报错: {e}。请修正语法或逻辑错误。"
            
        prompt_memory.append({"role": "assistant", "content": factor_code})
        prompt_memory.append({"role": "user", "content": feedback})
    return discovered_factors
```
</details>

### 8.4 NN：神经网络与深度表征学习

**核心思想**:
放弃寻找“明确的数学公式”。直接把全市场的股票数据切片张量，喂给深度神经网络。网络最后一层输出的隐变量 (Latent Variable) 直接作为因子的截面打分。

**闭环流转**:
```text
[ 全市场股票 20 天量价张量 [B, T, F] ] ---> [ 神经网络层 (Transformer / GRU) ]
                                                        |
                                                        v
                                          [ 隐空间输出 (因子截面打分) ]
                                                        |
                                                        v
[ 配置化未来收益标签 ] ------------> [ 计算 Batch IC / Rank Loss (目标函数) ]
                                                        |
                                                        v
                                          [ 梯度传导反向传播，更新模型权重 ]
```

**V4 接口映射**:
- **`initialize`**: 实例化深度时间序列网络（特征提取器），初始化网络权重。初始化训练优化器和 Loss 函数。
- **`generate`**: 将行情数据（张量切片）喂给神经网络执行一次前向传播 (Forward)。网络输出一个形状为 `[Batch, Seq, Output_Channels]` 的连续 Tensor。每一个 Channel 代表一个“黑盒因子表达式”。
- **`evaluate`**: 绝不能切断计算图。直接在 GPU 上让网络的输出 Tensor 与配置化未来收益 Tensor 计算目标 Loss。将 loss 张量对象和计算图原封不动装入 `EvaluationFeedback.raw_outputs` 回传。
- **`update`**: 取出带有计算图的 Loss，执行标准深度学习反向传播更新：`loss.backward()` 传导梯度，随后调用 `optimizer.step()` 更新模型权重。

### 8.4.1 训练张量与可保存因子的边界

NN 的训练中间产物不能直接当作因子落盘。以 `MyCustomNNMiner` 为例，`channel=-1` 表示完整模型输出，仅用于让 `ParallelEvaluator` 返回带梯度的原始张量并完成反向传播；它不是最终可查看或交易的单因子。

每轮权重更新后，Miner 将冻结的模型快照的每个输出通道物化为 `FactorExpressionTensor(model_version_id, channel_idx)`：单品种通道返回对齐行情索引的 `pd.Series`，跨资产通道返回时间 × 资产的 `pd.DataFrame`。模型仅在 `mine_period` 拟合权重和标准化参数，通道使用 `test_period` 的样本外 IC、RankIC、Turnover 与自定义 fitness 排名；跨轮候选还会按数值相关性过滤并保留全程 Top-K。

`model_version_id` 由完整模型包内容摘要生成；每个通道同时拥有由 `{"model_version", "channel"}` 导出的逻辑哈希。`FactorMinerDirector` 保存时会把可恢复模型包写入 `factor_db/models/<model_version>.npz`，其中包含权重、偏置、特征顺序、标准化参数、数据模式和 schema 版本；新通道使用 `{"type": "nn_channel", "model_version": ..., "channel": ...}`，并附加可选的 `model_file`、`model_format` 与特征元数据。历史 `dl_channel` 和 `factor_db/weights/<model_version>.pt` 裸权重档案继续可读。

<details><summary>底层逻辑参考实现</summary>

```python
class DeepAlphaModel(nn.Module):
    def __init__(self, num_features=6):
        self.encoder = TransformerEncoder(num_layers=3)
        self.fc = Linear(hidden_size, 1)
        
    def forward(self, x):
        hidden = self.encoder(x)
        score = self.fc(hidden[:, -1, :]) 
        return score

def dl_factor_mining(train_loader, epochs=100):
    model = DeepAlphaModel()
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = MSELoss()
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            alpha_scores = model(batch_x)
            loss = loss_fn(alpha_scores, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model 
```
</details>

---

### 💡 总结：V4 统一闭环下的四大流派对照表

| 流派 | `generate_candidates` 的物理产物 | `evaluate_candidates` 的核心职责 | `update_model` 的真实动作 | `MinerState` 记录的记忆实体 |
| :--- | :--- | :--- | :--- | :--- |
| **GP** | 随机或交叉变异产生的 AST 公式树 | 多进程跑树解析，计算标量 IC/IR | 精英留存，强制淘汰低分树 | 优质的基因树种群 (Population) |
| **RL** | Policy 网络采样出的一组 Action 轨迹 | 快速计算因子多空收益的夏普比率 | 计算优势函数，利用策略梯度更新网络 | 策略网络的权重与 Replay Buffer |
| **LLM** | 提取 Markdown 得到的 Python 源码 | 沙盒执行代码，拦截危险指令并测算 IC | 将表现或报错信息转化为文本反思日志 | 成功与失败的 Few-shot 案例库 |
| **NN** | 特征提取网络前向传播得到的 Tensor 隐通道 | 保持计算图不塌陷，计算 Batch IC Loss | 执行 `loss.backward()` 与权重更新 | 神经网络本身的参数字典 (.pt 权重) |

---

## 9. 统一数据流水线与跨市场对齐 (Data Pipeline)

在 V4 架构中，高质量的模型离不开强健的数据供养层。整个 `data_feed` 模块不仅负责历史数据的拉取，还需要在内存中完成无缝的时区对齐、断层填补以及前端元数据的级联分发。

### 9.1 基于流动性感知的元数据路由
前端通过 `/api/exchange_meta` 实时获取 CCXT 所支持的 `Exchange -> TradeType -> Symbol`。后端在此阶段会额外抓取 24 小时 `quoteVolume` 并进行倒序排序，这能确保前端下拉框中优先展示市场上流动性最好的主力合约/现货（如 BTC, ETH, SOL）。这一机制同时也能避免多市场混合下载时产生的错位（如现货独有的山寨币被意外放入期货下载队列）。

### 9.2 底层持久化与查询的一致性保证
由于 CCXT 在不同市场下返回的 Symbol 格式差异巨大（如期货 `1000CAT/USDT:USDT`，现货 `1000CAT/USDT`），在持久化层面，V4 严格统一了从内存符号到物理文件名的转换规范：
`safe_symbol = symbol.replace('/', '_').replace(':', '_')`

唯一允许的落盘格式为：`{safe_symbol}-{timeframe}-{trade_type}.feather`。例如，永续合约 `BTC/USDT:USDT` 必须落盘为 `BTC_USDT_USDT-1m-futures.feather`；读取器不会回退匹配旧命名。
无论是批处理下载器（`Batch Downloader`）、增量下载器，还是引擎读取器（`Real Data Client`）、断层修补器（`Gap Filler`），都必须通过上述绝对规则进行映射，保证了从硬盘扫描到内存加载的数据资产 100% 对齐。

### 9.3 沉浸式数据补全 Console
对于量化研究员而言，数据的完整性检查是一个“黑盒”。V4 架构在 Web 端集成了基于 WebSocket 的长连接控制台（Execution Console），实现了 CCXT 单步请求、K 线拉取重试、断层 Gap 合并等后端日志向前端 Drawer 的实时穿透，将枯燥的 ETL 过程转化为了极具安全感的可视化反馈。

---

## 10. 算子库解耦与全局策略驱动 (Config-Strategy Driven)

为了实现彻底的控制反转 (IoC)，并让终端研究员摆脱繁重的“类继承”与“框架底层耦合”，FactorMiner V4 提供了一个全局的 `OperatorRegistry` 算子注册中心，并通过**唯一对外网关 `FactorMinerDirector`** 进行声明式调度。

### 10.1 全局算子注册中心 (OperatorRegistry)

为了防止沙盒拦截自定义算子，也为了让 GP 和 RL 能够自动感知用户写了哪些新算子，V4 抛弃了“在 Miner 类里写函数”的做法，转而采用全局字典池的设计。

```python
# ==========================================
# 框架侧：提供全局注册表
# ==========================================
    _registry = {}

---

## 10. 数据供养层与 Config-Driven CLI (RealDataClient)

在 FactorMiner V4 中，为了实现工业级的数据回测和策略挖掘分离，我们彻底剥离了代码中硬编码的 Mock 数据，转而采用与 Freqtrade 类似的 **Config-Driven（配置驱动） CLI 工作流**。

### 10.1 统一配置入口 (config.json)

所有的静态参数、运行边界、以及与挖掘流派无关的基础环境配置全部抽离至 `config.json`，核心结构包含时间片切割机制：

```json
{
    "data_feeds": {
        "exchange": "binance",
        "instrument_type": "futures",
        "timeframe": "1m",
        "pairs": ["BTC/USDT:USDT", "SUI/USDT:USDT"],
        "mine_period": [["2025-07-20", "2025-08-01"]],
        "test_period": [["2025-08-02", "2025-08-15"]],
        "mining_mode": "sequential_single"
    }
}
```

### 10.2 真实数据源切片 (RealDataClient)

系统内部提供 `RealDataClient` 组件接管底层 `data/` 目录中高速 `.feather` 格式的访问权限：
- **动态寻址**：根据 CCXT 原始标的自动定位 `data/{exchange}/{instrument_type}/{safe_symbol}-{timeframe}-{instrument_type}.feather`。
- **时间段切片与拼接**：自动遍历 `mine_period` 中的多个时段，提取并过滤掉不需要的震荡市或噪声数据段，通过 `pd.concat` 无缝拼接出一块纯净的训练集传给底层挖掘器。
- **模式切换**：支持 `sequential_single` 串行单品种挖掘模式，避免在 GP 等基于一维序列的算子引擎中发生不同资产数据的错位滚动。

### 10.3 高级批量数据下载与元数据引擎 (Advanced Batch Downloader)

为了彻底解决数据准备的痛点，V4 架构引入了基于 CCXT 强驱动的独立批处理下载引擎。
- **智能元数据探针与网络降级 (Meta Fallback)**：系统启动时动态探查交易所 (如 Binance) 的存活状态和可下载特征维度 (`symbols`, `timeframes`, `trade_types`)。当遭受网络封锁（如 HTTP 451）时，底层探测器将无缝降级 (Fallback) 返回内置的 36+ 种高优主流代币白名单，保障平台在极端网络下依然可运行。
- **笛卡尔积排列并发引擎**：通过 `POST /api/batch_data_coverage`，底层系统支持接受多维度数组特征传入。在后端运用 `itertools.product` 将 `Symbols × TradeTypes × Timeframes` 动态解包为微任务队列，并通过 `WebSockets` 实现在前端对每一个排列组合 (Permutation) 本地数据的精确覆盖率探查与下载状态广播。
- **智能合并与对齐策略**：支持 `merge`（去重合并）、`fill_gap`（自动嗅探断点并双向延展填充缺口）以及 `overwrite`（强覆盖），最大程度减少网络 IO。

### 10.4 工业级 CLI 入口 (main.py)

利用 `main.py` 提供的标准化 CLI 入口，研究员完全不需要修改引擎的任何源码，只需指定外挂的配置和 Miner 插件目录即可完成无休止的后台计算：

```bash
python main.py --miner MyCustomGP --config user_workspace/configs/config.json --user-dir user_workspace
```    

### 10.5 算子维度与数据特征的解耦 (Arity & Streams)

在使用 GP 等依托于数学语法树的挖掘流派时，系统在设计上实现了**“暴露特征（Leaves）”与“算子维度（Arity）”的完全解耦**：

1. **`required_streams` (特征暴露)**：
   在 `config.json` 中配置的 `required_streams: ["close", "volume"]` 仅代表向算法环境注入的基础变量特征（即树的底层叶子节点）。如果底层 `.feather` 文件中拥有 `open, high, low, close, volume, funding_rate`，则完全可以将它们全部配置在此处。算法会在生成公式时随机抓取这些数据列作为基础计算元素。
   
2. **算子输入维度 (Operator Arity)**：
   系统不限制算子只能计算两个特征。算子能吞吐的变量数量完全取决于其注册时的 `arity` 属性：
   - **一元算子 (Unary, arity=1)**：如 `ts_delay(x)`, `log(x)`。
   - **二元算子 (Binary, arity=2)**：如 `add(x, y)`, `corr(x, y)`。
   - **多元算子 (N-ary, arity=N)**：如 `condition(A>B, C, D)` (arity=3)，或复杂的 `vwap(high, low, close, volume)` (arity=4)。
   
   只需通过 `@OperatorRegistry.register(name="my_n_ary_op", arity=N)` 在 `user_workspace` 中进行外挂注册，底层架构便能自动感知并生成拥有多条分支的巨大语法树。

---

    @classmethod
    def register(cls, name=None, arity=1):
        def decorator(func):
            op_name = name or func.__name__
            cls._registry[op_name] = {"func": func, "arity": arity}
            return func
        return decorator

class EvaluatorRegistry:
    _registry = {}
    
    @classmethod
    def register_fitness_hook(cls, hook_name):
        def decorator(func):
            cls._registry[hook_name] = func
            return func
        return decorator
```

### 9.2 用户层极致的声明式脚本体验 (Freqtrade 范式)

终端研究员不再需要了解底层框架的代码，我们将研发模式升级为了开源量化界极其优雅的 **“Freqtrade 范式” (物理隔离 + 动态加载)**。

**研发流 (User Workflow)：**

1. **配置编写**: 研究员在 `user_workspace/configs/` 中编写 JSON 或 YAML 配置。
2. **算子与挖掘机开发**: 研究员在 `user_workspace/custom_miners/` 和 `user_workspace/custom_operators/` 目录中放入自定义的 `.py` 文件，通过 `@MinerRegistry.register` 或 `@OperatorRegistry.register` 进行声名。

```python
# 文件：user_workspace/custom_miners/my_gp.py
from core.miner.registry import MinerRegistry
from core.miner.paradigms.gp_miner import GPFactorMiner

@MinerRegistry.register("MyCustomGP")
class FirstGPMiner(GPFactorMiner):
    def generate_candidates(self):
        # 用户的自定义突变逻辑
        return super().generate_candidates()
```

3. **CLI 命令行即插即用**: 
   直接在根目录执行命令，底层 `DynamicLoader` 会在启动时自动扫盘加载 `user_workspace/` 下的所有文件，实现“写完即跑，框架零入侵”。

```bash
factorminer mine --miner MyCustomGP --config user_workspace/configs/demo_config.json --user-dir user_workspace
```

这种架构彻底实现了**平台开发者（维护核心 `core/`）**与**策略研究员（只需维护 `user_workspace/`）**之间的职责与物理隔离。

**扩展的结果契约：**

- 自定义 Miner 仍需实现初始化、候选生成、评估和更新四个阶段；更新策略、提示词或模型权重后，必须把最终已评分候选写入 `MinerState.population` 或 `replay_buffer`，否则 Director 没有可持久化的研究产物。
- `OperatorRegistry` 中的内置与用户算子由统一 AST runtime 消费。GP/RL 根据 arity 生成一元或二元节点，`FactorExpressionAST.compute()` 再按节点名分发到同一注册表；因此注册算子会实际参与候选生成和求值，而不会仅停留在发现阶段。
- `MyCustomGP` 的默认时序搜索调色板包含 `custom_ts_decay`、`ts_zscore_20`、`ts_delta_5`、`ts_rank_20` 和 `ts_volatility_20`，并可由 `search_space.allowed_operators` 显式收窄或扩展。
- `EvaluatorRegistry.register_fitness_hook(name)` 的 Hook 接收 `factor_values, returns, base_metrics`，可返回数值或含 `fitness_score` 的字典；任务配置通过 `fitness.hook` 引用该名称。
- NN 需将训练阶段的临时张量模型与可评估的输出通道区分开：只有被物化、评估并保留的通道表达式会成为因子档案，权重作为其可追溯附件保存。

**启动校验与验证闭环：**

- DynamicLoader 返回模块加载报告；CLI 和 WebUI Launchpad 在创建数据客户端前汇总模块导入错误、Miner 存在性、data_feeds 基础字段、算子注册/arity、Fitness Hook 名称。
- 算子注册只允许 arity 为 1 或 2，函数签名必须覆盖声明的参数数量；Fitness Hook 必须接收 factor_values、returns、base_metrics。未知配置会失败并返回可行动的错误，而非回退为零因子或默认评分。
- tests/test_cli_smoke.py 在临时目录写入最小 Feather 数据，实际调用公开 CLI 验证 MyCustomGP、MyCustomRL、MyCustomLLM、NN 均能完成一轮并落盘因子元数据。
- 真实运行的档案指标采用稳定键名 `IC`、`RankIC`、`Turnover`、`fitness_score`。前端摘要、任务结果和 Inspector 必须直接消费该契约；不能以不存在的小写别名读取并将缺失值误呈现为零。变更 Python 执行链路后需重启 FastAPI；重启只丢弃 `TaskManager` 的内存 Tracker，不影响 `LocalFactorStorage` 的 metadata 和 values 快照。

README 的目录地图以该边界为准：`core/` 只承担研究执行，`api/` 负责服务与任务可观测性，`web/` 负责研究工作台，`user_workspace/` 是策略、算子和评分的用户扩展面，`factor_db/` 只保存可追溯的研究产物。

---

## 10. 因子实盘落地与白盒审查 (Compiler & Inspector)

统一闭环跑完后，面对海量的因子，系统还需要解决“怎么看”和“怎么用”的问题。为此，FactorMiner V4 引入了 `FactorInspector` 和 `FactorCompiler`。

### 10.1 补齐“怎么用”：FactorCompiler (实盘推理编译器)

目前因子被存为了 AST 字典、Python 源码或 PyTorch 权重。但在实盘高频交易系统中，我们不可能每次收到一个新的 Tick 数据，就去解析一次 JSON AST 或者跑一次沙盒 `exec()`。实盘需要的是极低延迟的流式计算。

`FactorCompiler` 的职责是：在研究员决定将某个因子上线实盘时，把因子的“灵魂（元数据）”编译成底层 C++ 或高性能引擎能直接调用的流式算子。

```python
class FactorCompiler:
    """
    因子上线编译器：将存储的因子逻辑转化为实盘极速推理模块
    """
    def __init__(self, storage_client, sandbox):
        self.storage = storage_client
        self.sandbox = sandbox

    def compile_for_live_trading(self, factor_id: str):
        metadata = self.storage.get_metadata(factor_id)
        logic = metadata.logic_reference

        # 按稳定的 logic_reference.type 分流，而不是按可自定义的 Miner 名称分流。
        if logic["type"] == "json_ast":
            return FactorExpressionAST(logic["ast"]).compute
        if logic["type"] == "python_source":
            code = self.storage.load_llm_source(logic["source_file"])
            self.sandbox.validate_code(code)
            return FactorExpressionCode(code, sandbox=self.sandbox).compute
        if logic["type"] in {"nn_channel", "dl_channel"}:
            model = load_portable_model(self.storage, logic)
            return lambda data: model.predict_channel(data, logic["channel"])
        raise NotImplementedError("该产物不能重建为可执行因子")

    def deploy_to_live_server(self, factor_id: str, server_target: str):
        self.compile_for_live_trading(factor_id)  # 只验证能否重建
        raise NotImplementedError("尚未实现部署传输层，因子没有被推送")
```

当前仓库只负责把持久化产物安全地重建为本地 callable。真正的镜像构建、签名、审批、
推送、回滚和远端健康检查尚未实现，因此接口不得返回“部署成功”。接入真实传输层后，
只有收到远端节点确认并记录部署版本时才能转为成功状态。

### 10.2 补齐“怎么看”：FactorInspector (白盒化审查台)

研究员跑完 `director.run()` 之后，如果只看到控制台打印“捕获 50 个圣杯因子”，这在真实量化团队里是过不了风控的。研究员需要“打开黑盒”，看看这个因子长什么样、各分位数收益如何、是否发生过拟合。

`FactorInspector` 增加了一层专业的可视化白盒审查能力，它对接下游的 Alphalens 等专业量化分析库，不仅把 Parquet 文件里的面值画成图，还能把因子的“逻辑”反向翻译成人话。

```python
class FactorInspector:
    """因子白盒化审查与可视化面板"""
    def __init__(self, storage_client: FactorStorageInterface):
        self.storage = storage_client
        
    def show_tearsheet(self, factor_id: str):
        """生成因子的全身体检报告 (Tearsheet)"""
        # 1. 加载肉体 (Parquet 面值)
        factor_values = self.storage.load_factor_values(factor_id)
        
        # 2. 调用 Alphalens 等库生成 分层收益图、IC 衰减图、换手率图
        self._plot_quant_metrics(factor_values)
        
        # 3. 还原灵魂 (可读性展示)
        metadata = self.storage.get_metadata(factor_id)
        if metadata.miner_type == "GP":
            # 将 AST 树反向解析为人类可读的数学公式 (如使用 SymPy 渲染 LaTeX)
            print("因子数学公式: ", self._ast_to_formula(factor_id))
        elif metadata.miner_type == "LLM":
            print("生成代码:\n", self.storage.get_llm_logic(factor_id))
            print("\n大模型反思日志:\n", self.storage.get_llm_reflection(factor_id))
        elif metadata.logic_reference.get("type") in {"nn_channel", "dl_channel"}:
            print(f"NN 提取特征通道: {metadata.logic_reference['channel_index']}")
            # 可拓展特征归因分析 (Feature Importance / SHAP 值)
```

---

## 11. 工程落地状态 (Implementation Status)

基于上述 V4 架构设计的理念，底层四大挖掘范式**已全部成功落地，并完成了端到端验证闭环**：

- **GP (MyCustomGPMiner)**: 验证了基于抽象语法树的进化、变异和交叉，能够根据配置进行因子繁衍。
- **RL (MyCustomRLMiner)**: 验证了策略梯度思想，通过概率权重字典实现了无 PyTorch 依赖的轻量级强化学习寻优闭环。
- **LLM (MyCustomLLMMiner)**: 成功实现了大语言模型反思记忆（Reflection Loop）驱动的代码生成。验证了并行执行沙盒 `RestrictedSandbox` 的安全性，并在缺乏真实 API Key 时，验证了系统的容灾限流及优雅降级机制。
- **NN (MyCustomNNMiner)**: 基于纯 NumPy 构建了包含反向传播（Backpropagation）能力的微型张量机制 `MockTensor`。训练完成后，模型输出会按通道物化、评分、保留 Top-K，并将权重与通道元数据一同持久化，已打通 CLI 与 Web 的统一结果链路。
- **并行评估与沙盒引擎 (ParallelEvaluator)**: 全面打通了回测评分闭环。验证了 `EvaluatorRegistry` 对于外部自定义挂钩 (`custom_fitness`) 的无缝动态注入（如实现了惩罚换手率的 `my_bear_market_hunter`），保证了回测评价维度的无限扩展。
- **因子归因与审查引擎 (FactorInspectorEngine)**: 实现了 `core/inspector/` 独立审查模块与 `factorminer inspect` CLI 命令。支持从 Factor ID、AST 字典字符串或 Python 源码解析因子，在多币种与样本外 (OOS) 时间段计算 Coverage（有效覆盖率）、Pearson/RankIC (Mean/Std/IR/t-stat)、Lag 1..10 延迟衰减、5-Quantile 组间收益与换手率，并利用 Rich 库输出终端卡片面板。

以上所有的完整实现均无缝外挂于 `user_workspace/custom_miners/` 和 `user_workspace/configs/` 目录下，实现了引擎核心代码的**零入侵**，完美兑现了“Config-Driven 配置驱动与逻辑插件化”的架构愿景。

---

## 12. 现代化 Web UI 与前后端分离架构

在彻底完善后端引擎的基础上，FactorMiner V4 引入了基于 **React (Vite) + FastAPI** 的纯正前后端分离控制台，替代了原有的 CLI 与 Flask 方案。

### 12.1 页面空间与布局重构
采用**顶部 Tab 导航栏**替代传统的左侧边栏，将屏幕的横向空间极限释放，为展现宽表数据、超长生命周期 ECharts 及长堆栈信息提供充足的视野支撑。

### 12.2 全局指挥中心 (Command Center)
- **无感心跳**：提供 `Engine Online` 状态的探测。
- **全局看板**：由 FastAPI `/api/stats` 接口驱动，宏观统计并展示 Task 总量、落盘因子的总量、以及综合成功率。
- **活动时间线**：Timeline 组件记录后端异步调度任务的轨迹。

现行 Research Dashboard 进一步使用 `GET /api/dashboard` 作为单次聚合快照：任务部分来自内存中的 `TaskManager`（运行中任务和近期执行记录），因子部分来自持久化 `factor_db/metadata`（总量、Miner 分布、生命周期覆盖率和按 fitness 排序的候选）。因此服务重启后历史任务时间线会按既有内存策略清空，但因子档案和研究质量概览仍然保留。前端只展示这两类真实来源，并将因子链接到 Inspector、任务入口链接到 Launchpad。

### 12.3 异步调度与 WebSocket 实时流推送
传统的 Web 提交长耗时任务往往会导致请求超时或阻塞主线程。V4 采用真正的异步池化策略：
1. **任务分发**：前端提交 JSON (Miner & Config) 至 `/api/launch`，FastAPI 借助 `BackgroundTasks` 分配后台守护线程启动 `FactorMinerDirector.run()`，不阻塞 HTTP 响应。
2. **状态挂载**：任务挂载至全局内存态字典 `TaskManager`。
3. **实时推送**：注入回调勾子 `progress_callback` 进底层演化循环（`epoch` 级别）。每一次种群繁衍，底层直接透过 `ws_manager.broadcast` 将状态压入 WebSocket 管道，前端通过 `Launchpad` 的 Drawer 实现在无感刷新下的微秒级进度条呈现。

### 12.4 Factor Inspector Phase II：可复现快照、Tearsheet 与批量审查

Inspector 不能读取 `TaskManager` 中的临时任务结果作为因子档案：服务重启后该内存会丢失，且 CLI 产生的因子也不会出现。Phase I 以 `LocalFactorStorage` 中的 `factor_db/metadata/*.json` 为唯一目录来源，并提供以下 API：

- `GET /api/factors`：按创建时间、IC 或 fitness 返回可搜索/过滤的因子摘要；
- `GET /api/factors/{factor_id}`：返回元数据、按存储类型还原的逻辑引用，以及因子值审查快照是否存在；
- `PATCH /api/factors/{factor_id}/lifecycle`：持久化 `DISCOVERED → INSPECTED → PAPER_TRADING → LIVE / RETIRED` 等人工审查状态。

前端根据 `logic_reference.type` 而不是笼统的 Miner 名称呈现白盒内容：`json_ast` 绘制 AST 树并展示表达式，`python_source` 读取受控源码文件，`rl_actions` 展示动作轨迹，`nn_channel` 展示模型版本、输出通道和权重工件是否存在；历史 `dl_channel` 会在 API 层归一化为 `nn_channel`。Launchpad 的已保存因子 ID 使用 `/inspector?factor=<id>` 直接跳转到对应档案。

Phase II 在 Director 落盘逻辑工件后，重新以该候选和同一数据客户端物化因子值，并和 `RealDataClient.get_returns()` 返回的未来收益对齐，写入 `factor_db/values/<factor_id>.parquet`。标准 schema 为 `timestamp, asset（截面模式）, factor, forward_return`；`FactorMetadata.data_lineage` 同时记录数据源、交易对、市场类型、周期、输入流、样本范围、规范化 `target` 和精确收益公式。统一 Target Builder 支持 `current_close` / `next_open` 入场、`close` 出场、可配置 `horizon_bars` 以及 simple / log return；省略配置时保持 `close[t+1] / close[t] - 1` 的历史行为。每个数据区间独立生成标签，Inspector 按 factor ID 默认继承入库时的目标定义。

对 MyCustomGP 的真实 Feather 回归应检查完整闭环：配置允许的用户算子会生成符合 arity 的 AST；AST runtime 产生可对齐的序列；Evaluator 写入上述四项指标；Director 最后将同一候选的逻辑、指标、数据血缘及 values/forward-return 快照持久化。只看到模块成功导入或任务完成并不足以证明该链路有效。

Tearsheet API 只分析上述 parquet：单标的计算时序滚动 IC、逐期因子差分换手和全样本分位平均未来收益；截面模式计算逐期截面 IC 的滚动均值、资产矩阵换手和逐期分位组合的平均未来收益。没有快照会返回明确的重新挖掘提示，绝不以模拟曲线替代。

Inspector 支持选择 2–5 个具有快照的因子调用比较 API，以同一口径对齐并叠加其滚动 IC；同时提供批量生命周期更新 API。测试套件以临时 FactorStorage 和 FastAPI TestClient 覆盖快照 schema、顺序/截面分析、比较和批量审查端点。

### 12.5 三语界面与研究工件边界

前端采用轻量本地 i18n Context，支持 `zh`、`en`、`de` 三种界面语言。初始语言由浏览器语言推断，用户通过顶部导航选择后写入 `localStorage`；翻译词典集中在 `web/src/i18n.tsx`，避免页面中散落条件判断。第一批覆盖全局导航、Research Dashboard 与 Factor Inspector，后续页面按相同键空间扩展。

语言层只负责操作性 UI 文案：导航、按钮、状态说明、筛选项、空态及解释性提示。因子 ID、逻辑哈希、交易对、AST/表达式、代码、模型版本、权重文件，以及 IC、RankIC 等研究指标缩写均作为不可翻译的原始工件保留。后端保持返回稳定枚举与结构化数据，绝不向 API 注入某一种自然语言的业务文案。
