# FactorMiner V4 Architecture: A Unified Factor Mining Framework

[简体中文](v4_architecture_design.md)

Updated 2026-11-25. The current release is a local, single-user research tool run from a source checkout. It requires Python ≥3.10 (tested with 3.12) and Node.js ≥22.12. See the [product guide](../FactorMiner_Product_Report_EN.md) for installation, testing, and file management. The wheel provides the Python engine, API, and CLI; the workspace, examples, and frontend come from the source checkout.

This document includes interface designs and illustrations for teaching the algorithms. Pseudocode marked “design illustration” or “planned” is not guaranteed to run directly. The source code and the table below define the current contract:

| Capability | Current implementation | Not yet delivered |
| --- | --- | --- |
| Candidate generation | GP mutation and elitism, reference RL weight updates, NumPy NN, LLM code and feedback | Full GP crossover, built-in Actor-Critic or general GPU computation graphs |
| Evaluation | 1–64 local threads, IC/RankIC/factor changes, source-hash deduplication | Ray/Celery, correlation-based deduplication across all Miners, portfolio optimization |
| Archiving | AST/source/NN artifacts, snapshots, data lineage, declared parent references and model provenance | Full search-state and RL policy restoration for arbitrary Miners |
| Inspection | CLI reevaluation; Web snapshots, trees, and comparison; manual lifecycle labels | Actual trading equity curves, NN 3D or importance plots, automated causal review |
| Execution | Reconstruct a local Python callable | C++ compilation, trading gateways, remote deployment |


## Core Design Principles

To support GP, RL, NN, and LLM factor mining, FactorMiner V4 abstracts mining into a general **search and optimization loop**: `generate candidates -> backtest and evaluate -> update the model -> generate again`.

The V4 architecture introduces a unified expression abstraction and lineage tracking to address the engineering challenges posed by these different approaches, such as mixed types and the distinct end-to-end evaluation loop required by neural networks. **At the user-facing layer, the system encapsulates the underlying approaches behind a single gateway, `FactorMinerDirector`, offering a declarative, strategy-driven workflow.**

---

## 1. A Unified Intermediate Representation (FactorExpression)

The `FactorExpression` protocol encapsulates the underlying logic in a common interface, avoiding the mixed types of `candidates: List[Any]` and repeated `if isinstance` checks in the Evaluator.

```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Union, Optional

class FactorExpression(ABC):
    """
    Unified intermediate representation for factors.
    GP, RL, NN, and LLM generators all return instances of its subclasses.
    """

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame, Any]:
        """
        Common computation interface.
        - AST subclasses: traverse and evaluate the parse tree here.
        - Code subclasses: compute with exec/eval here.
        - NN subclasses: execute model.forward(data) here.
        """
        pass

    @abstractmethod
    def get_source(self) -> Any:
        """
        Return the underlying representation (e.g., an AST dict, code string, or nn.Module).
        """
        pass

class LineageTrackableMixIn:
    """Capability marker for traceable lineage."""
    def get_lineage_parents(self) -> List[str]:
        return getattr(self, '_parent_ids', [])

class LLMReflectableMixIn:
    """Capability marker for LLM reflection memory."""
    def get_reflection_history(self) -> str:
        return getattr(self, '_reflection_history', "")
```

---

## 2. Core Abstract Interface (BaseMiner, Design Illustration)

V4 introduces a common evaluation feedback structure, `EvaluationFeedback`, and a `DiversityFilter` to address architectural issues: the disconnect in NN evaluation, homogeneous LLM candidates, and lost evolutionary state.

```python
from dataclasses import dataclass
import torch
import hashlib

@dataclass
class EvaluationFeedback:
    """Common evaluation feedback supporting scalar metrics and tensor computation graphs."""
    metrics: List[Dict[str, float]]          # Conventional metrics (IC, RankIC, Turnover, fitness)
    raw_outputs: Optional[torch.Tensor] = None # Tensor returned directly by the DL forward pass
    raw_targets: Optional[torch.Tensor] = None # Configurable forward-return labels for the DL loss

class DiversityFilter:
    """Filter for factor orthogonality and diversity."""
    def __init__(self, correlation_threshold: float = 0.7):
        self.threshold = correlation_threshold
        self.archive_hashes = set()

    def filter_redundant(self, candidates: List[FactorExpression], data: pd.DataFrame) -> List[FactorExpression]:
        unique_candidates = []
        for cand in candidates:
            # 1. Hash-based hard deduplication at the syntax/semantics level
            expr_hash = hashlib.md5(str(cand.get_source()).encode()).hexdigest()
            if expr_hash in self.archive_hashes:
                continue

            # 2. Soft deduplication by cross-sectional correlation (can use the backtest engine)

            self.archive_hashes.add(expr_hash)
            unique_candidates.append(cand)

        return unique_candidates

class BaseFactorMiner(ABC):
    """
    Common base class for factor mining.
    Supports GP (genetic programming), RL (reinforcement learning), NN, and LLMs.
    """

    def __init__(self, data: pd.DataFrame, config: Dict):
        self.data = data
        self.config = config
        self.history = []

    @abstractmethod
    def initialize_search_space(self) -> None:
        """
        1. Initialize the search space / environment.
        """
        pass

    @abstractmethod
    def generate_candidates(self) -> List[FactorExpression]:
        """
        2. Generate candidate factors (Propose).
        @return: A list containing only FactorExpression objects.
        """
        pass

    @abstractmethod
    def evaluate_candidates(self, candidates: List[FactorExpression]) -> 'EvaluationFeedback':
        """
        3. Evaluate and score (Evaluate / Reward).
        - GP/LLM/RL: return conventional metrics (IC, Sharpe ratio, etc.).
        - NN: return tensors with their computation graph intact for gradients in update_model.
        """
        pass

    @abstractmethod
    def update_model(self, candidates: List[FactorExpression], feedback: 'EvaluationFeedback') -> None:
        """
        4. Apply feedback and update the model (Feedback & Learn).
        - GP: update the population (self.state.population = new_trees).
        - LLM: update reflection memory (self.state.failed_reflections.append(...)).
        - NN: backpropagate (loss.backward(); optimizer.step()).
        """
        pass

    def mine(self, n_iterations: int, progress_callback=None) -> List[FactorExpression]:
        """
        5. Standard main loop with integrated deduplication.
        """
        self.initialize_search_space()
        div_filter = DiversityFilter(self.config.get('max_corr', 0.7))

        for epoch in range(n_iterations):
            # 1. Propose
            raw_candidates = self.generate_candidates()

            # 1.5 Filter highly correlated/duplicate factors using global historical hashes from FactorStorage
            candidates = div_filter.filter_redundant(raw_candidates, self.data)
            if not candidates: continue

            # 2. Evaluate (with robustness-warning suppression, e.g., invalid constant factors from GP bloat)
            feedback = self.evaluate_candidates(candidates)

            # 3. Learn: update_model updates the internal self.state or model parameters
            self.update_model(candidates, feedback)

            self._log_epoch(epoch, candidates, feedback.metrics)

        return self._get_best_factors()

    def _log_epoch(self, epoch: int, candidates: List[FactorExpression], metrics: List[Dict[str, float]]):
        pass

    def _get_best_factors(self) -> List[FactorExpression]:
        pass
```

---

## 3. Standardized Storage and Lineage Tracking (FactorStorage)

Each approach stores its AST, source code, or model artifact, then uses a common snapshot format for factor values and labels. Metadata records data lineage, selected configuration, parent references declared by the Miner, and model provenance. Parent references are not necessarily archived Factor IDs, nor do they imply that the full search process can be restored. The interface sketch below illustrates the division of responsibilities; actual fields are defined in `core/miner/entities.py` and the storage implementation.

### 3.1 Lineage and Metadata (FactorMetadata)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class FactorMetadata:
    factor_id: str                   # Globally unique factor ID (e.g., alp_llm_20260710_001)
    miner_type: str                  # Approach: 'GP', 'RL', 'NN', 'LLM'
    user_id: str                     # ID of the researcher submitting/running this task

    # [New] Factor lifecycle state machine
    # Manual status labels; PAPER_TRADING/LIVE do not start paper or live trading
    lifecycle_status: str = field(default="DISCOVERED")

    # Lineage: support evolutionary ancestry or reflection chains
    parent_ids: List[str] = field(default_factory=list) # Parent IDs for mutation, or the previous failed LLM code ID
    generation_config: Dict[str, Any] = field(default_factory=dict) # Hyperparameter snapshot at generation time

    # Soul References: pointers to the underlying logic for each approach
    logic_reference: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    # GP -> {"storage_type": "json_ast", "path": "path/to/ast.json"}
    # NN -> {"storage_type": "model_channel", "model_version": "v4.1", "channel_idx": 42}

    # Evaluation metrics (results for the computed values)
    metrics: Dict[str, float] = field(default_factory=dict) # Metrics recorded at evaluation time (IC, RankIC, Turnover, fitness)
    created_at: str = ""
```

### 3.2 Heterogeneous Storage Interface (FactorStorageInterface)

The storage interface handles the characteristics of the four approaches as follows:

```python
from abc import ABC, abstractmethod
import pandas as pd

class FactorStorageInterface(ABC):

    # ==============================================================
    # Layer 1: logic and lineage (the "soul") -> approach-specific storage
    # ==============================================================

    @abstractmethod
    def save_gp_factor(self, ast_dict: Dict, metadata: FactorMetadata) -> bool:
        """
        [GP storage]
        - Method: serialize the nondifferentiable expression tree as a standard JSON AST (in NoSQL or the filesystem).
        - Current use: save expressions restorable by the local AST runtime; a C++ parser is planned.
        """
        pass

    @abstractmethod
    def save_rl_factor(self, action_sequence: List[str], agent_snapshot_bytes: bytes, metadata: FactorMetadata) -> bool:
        """
        [RL storage]
        - Method: store two components.
          1. The resulting explicit mathematical expression as code or an action sequence (for interpretability).
          2. The interface accepts policy weights, but the current Director action branch does not supply a full policy checkpoint; the reference RL implementation saves an AST.
        """
        pass

    @abstractmethod
    def save_llm_factor(self, python_code: str, reflection_log: str, metadata: FactorMetadata) -> bool:
        """
        [LLM storage]
        - Method:
          1. Store sandbox-validated, standalone Python source scripts as .py text artifacts.
          2. Store the LLM reasoning, reflection, and error logs (Reflection Log) alongside them in the database.
        - Current use: researchers review source and reflections; restored execution still uses the sandbox and is not live deployment.
        """
        pass

    @abstractmethod
    def save_model_weights(self, model_version_id: str, model_weights: bytes) -> bool:
        """
        Step 1: upload the large model-weight artifact separately.
        [One global call] Deduplicate internally and skip if it already exists.
        """
        pass

    @abstractmethod
    def save_nn_factor_channel(self, model_version_id: str, channel_index: int, metadata: FactorMetadata) -> bool:
        """
        Step 2: store a lightweight channel index.
        [Called inside a for loop] Pass only a version string and an integer index each time: a few hundred bytes with minimal network overhead.
        """
        pass

    # ==============================================================
    # Layer 2: computed data (the "body") -> standardized across all approaches
    # ==============================================================

    @abstractmethod
    def save_factor_values(self, factor_id: str, values_df: pd.DataFrame) -> bool:
        """
        [Standardized data storage for all approaches]
        - Method: whether the logic is code or a model, store the historical matrix it computes in standardized columnar **Parquet** format.
        - Contract: `values_df` must follow the cross-sectional matrix format (Datetime index, StockCode columns, factor scores as values).
        - Purpose: downstream Portfolio Optimizers and execution algorithms can read Parquet matrices with high I/O throughput for weighting and linear combinations, without needing to know how factors were mined.
        """
        pass
```

### 3.3 Putting It Together: Storage in a Complete User Script

How does the system handle heterogeneous storage once a researcher has configured and run a task? The following example shows LLM storage at the end of the `mine()` loop:

```python
# When the user calls: best_alphas = director.run(n_iterations=50)
# Pseudocode for the storage pipeline automatically triggered after the Miner finishes:

class LLMFactorMiner(BaseFactorMiner):
    def __init__(self, data, config, fitness_calculator, storage_client: FactorStorageInterface):
        super().__init__(data, config)
        self.storage = storage_client

    def _save_discovered_alphas(self, best_candidates: List[FactorExpression], final_feedback: EvaluationFeedback):
        for idx, cand in enumerate(best_candidates):
            metrics = final_feedback.metrics[idx]
            factor_id = f"alp_llm_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"

            # 1. Assemble logic and lineage metadata, with safe polymorphic MixIn checks
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

            # 2. Save the logic: use the approach-specific interface for Python source and reflection logs
            reflection_log = cand.get_reflection_history() if isinstance(cand, LLMReflectableMixIn) else ""
            self.storage.save_llm_factor(
                python_code=cand.get_source(),
                reflection_log=reflection_log,
                metadata=metadata
            )

            # 3. Save the values: compute the full historical matrix from current market data and store as Parquet
            factor_matrix = cand.compute(self.data)
            self.storage.save_factor_values(factor_id=factor_id, values_df=factor_matrix)

            print(f"Saved factor {factor_id} to the quant platform (logic: source script; values: Parquet matrix)")
```

---

## 4. Parallel Evaluation and Deduplication Cache (ParallelEvaluator)

The current `ParallelEvaluator` uses a local ThreadPoolExecutor with 8 threads by default. `evaluation.max_workers` accepts 1–64; the actual thread count does not exceed the candidate count. General deduplication uses source-code MD5 hashes, while the reference NN implementation also filters by output correlation. This does not eliminate all semantically duplicate factors.

The `EvaluationEngine` / Ray code below is an unimplemented distributed design illustration, not a current dependency, installation step, or entry point. Ray/Celery scheduling and a shared computation-cache service are not implemented.

```python
import hashlib
import ray

@ray.remote
def remote_compute_and_evaluate(expression_source: Any, data_ref: ray.ObjectRef, eval_func_bytes: bytes) -> Dict[str, float]:
    """Compute and backtest factors in parallel on Ray cluster nodes."""
    # 1. Restore the backtest function
    import cloudpickle
    eval_func = cloudpickle.loads(eval_func_bytes)
    # 2. Retrieve market data from shared memory
    df = ray.get(data_ref)

    # 3. Compute factor values (assuming the expression object has been serialized and passed in)
    # A production implementation needs try-except here to keep computation errors from crashing the process
    try:
        factor_values = expression_source.compute(df)
        metrics = eval_func(factor_values, df)
        return metrics
    except Exception as e:
        return {"IC": 0.0, "IR": 0.0, "error": 1.0}

class ParallelEvaluator:
    def __init__(self, data: pd.DataFrame, eval_metric_func):
        # Put large datasets in Ray shared memory to avoid copying them across processes
        self.data_ref = ray.put(data)
        import cloudpickle
        self.eval_func_bytes = cloudpickle.dumps(eval_metric_func)
        self.computed_cache = {} # Expression hash -> metrics

    def evaluate_batch(self, candidates: List[FactorExpression]) -> List[Dict[str, float]]:
        futures = []
        task_mapping = []

        for idx, cand in enumerate(candidates):
            # Hash the underlying factor logic as a unique semantic text key (intended to avoid recomputation after renaming variables)
            expr_hash = hashlib.md5(str(cand.get_source()).encode()).hexdigest()

            if expr_hash in self.computed_cache:
                # Cache hit: record the result directly
                task_mapping.append((idx, self.computed_cache[expr_hash]))
            else:
                # Cache miss: submit an asynchronous computation
                future = remote_compute_and_evaluate.remote(cand, self.data_ref, self.eval_func_bytes)
                futures.append(future)
                task_mapping.append((idx, expr_hash)) # Record where to insert the result later

        # Block until all remote computations finish
        remote_results = ray.get(futures)

        # Assemble results and update the cache
        final_results = [None] * len(candidates)
        remote_idx = 0
        for map_info in task_mapping:
            idx = map_info[0]
            if isinstance(map_info[1], dict): # Cache hit
                final_results[idx] = map_info[1]
            else: # Remote result
                res = remote_results[remote_idx]
                self.computed_cache[map_info[1]] = res
                final_results[idx] = res
                remote_idx += 1
        return final_results
```

---

## 5. Dynamic Code Sandbox Safety (RestrictedSandbox)

LLM-generated code must not execute directly in the Miner, Evaluator, or Inspector process. The current implementation is in
`core/evaluation/code_sandbox.py` and enforces four layers of protection:

1. A Python AST allowlist permits only vectorized pandas/NumPy expressions and local variable assignments;
2. Imports, loops, function/class definitions, file access, dynamic code execution, private attributes, and magic methods are prohibited;
3. Execution takes place in a short-lived, separate `subprocess` with wall-clock, CPU, memory, and file-descriptor limits;
4. Before returning, the sandbox strictly validates the type, index, columns, numeric dtype, finite values, and alignment with the input.

`FactorExpressionCode` fails closed: it refuses to compute unless a
`RestrictedSandbox` is explicitly supplied. There is no compatibility fallback to bare `exec()`. Sequential mode requires
`factor` to be a `pandas.Series` with exactly the input index; `cross_asset` mode requires a `pandas.DataFrame`
whose time index and asset columns exactly match those of every feature DataFrame.

The sandbox entry point uses an absolute script path, so calls from other working directories do not depend on the project `PYTHONPATH`. Subprocess creation copies the environment and limits BLAS/OMP threads to 1 only in the child, preventing multithreaded numerical-library imports from consuming the budget; the parent environment is unchanged. Defaults remain 3 seconds of cumulative CPU time (including imports), 5 seconds of wall time, and 1024 MiB of memory, subject to the existing platform conditions for resource limits. No extra CPU allowance is granted to the factor. Timeout and other exception paths reclaim the process and pipes.

A string-keyword denylist is not a security boundary and must not be reintroduced. New pandas/NumPy capabilities must be enabled individually in the AST
attribute allowlist, with both permitted-use tests and negative sandbox-escape tests.

---

## 6. Unified State Management (MinerState)

Within the common `update_model(candidates, evaluations)` loop, each approach requires different interactions with historical state:
- GP updates the current population.
- LLM updates the few-shot memory in its prompt (lists of successes and failures).
- RL updates the replay buffer.

To keep `BaseFactorMiner` focused, a separate `MinerState` object is proposed as the internal state container for each subclass.

```python
class MinerState:
    """
    State manager: a common container for algorithm memory and evolutionary state.
    """
    def __init__(self):
        # 1. For GP
        self.population: List[FactorExpression] = []

        # 2. For LLM (a pool of successful and failed experiences)
        self.successful_reflections: List[Dict] = []
        self.failed_reflections: List[Dict] = []

        # 3. For RL / NN
        self.model_weights_bytes: Optional[bytes] = None
        self.replay_buffer: List[Any] = []

    def get_llm_context_prompt(self) -> str:
        """Quickly assemble a few-shot prompt for the LLM approach."""
        prompt = "Here are past successful examples:\n"
        for item in self.successful_reflections[-5:]: # Use the latest 5
            prompt += f"Code: {item['code']}\nPerformance: {item['metrics']}\n"
        return prompt
```

---

## 7. LLMFactorMiner Interface Design Illustration

The pseudocode below illustrates the division between LLM generation, evaluation, and feedback. The runnable reference implementation is `user_workspace/custom_miners/my_custom_llm.py`; do not copy the illustration as a production implementation.

```python
class LLMFactorMiner(BaseFactorMiner):
    def __init__(self, data: pd.DataFrame, config: Dict):
        super().__init__(data, config)
        self.state = MinerState()
        self.sandbox = RestrictedSandbox()
        # Initialize the dedicated external evaluation engine
        self.evaluator = ParallelEvaluator(self.data, eval_metric_func=config['metric_func'])

    def initialize_search_space(self) -> None:
        # Inject the initial base prompt instructions
        self.system_prompt = "You are a top quant. Generate predictive formulas..."

    def generate_candidates(self) -> List[FactorExpression]:
        # 1. Retrieve historical memory from the state manager and build the dynamic prompt
        dynamic_prompt = self.state.get_llm_context_prompt()

        candidates = []
        # 2. Call the LLM in batches (use asynchronous I/O for parallel API requests in practice)
        # [Defensive design] Limit concurrency with an API key pool / token bucket to avoid provider rate limits or blocks
        async def _fetch_with_rate_limit():
            return await call_llm_api_with_pool(self.system_prompt, dynamic_prompt)

        # raw_codes = asyncio.run(batch_gather_with_limit(_fetch_with_rate_limit, self.config['batch_size']))
        raw_codes = ["mock_code_1", "mock_code_2"] # Pseudocode illustration

        # 3. Bind the generated code, sandbox, and implementation to the common representation
        for raw_code in raw_codes:
            cand = FactorExpressionCode(code_str=raw_code, sandbox=self.sandbox)
            candidates.append(cand)
        return candidates

    def evaluate_candidates(self, candidates: List[FactorExpression]) -> EvaluationFeedback:
        # Delegate to the parallel backtest engine, hiding its internal details
        metrics = self.evaluator.evaluate_batch(candidates)
        return EvaluationFeedback(metrics=metrics)

    def update_model(self, candidates: List[FactorExpression], feedback: EvaluationFeedback) -> None:
        # Store this iteration's feedback in memory for the next generate_candidates prompt
        for cand, eval_res in zip(candidates, feedback.metrics):
            log_entry = {"code": cand.get_source(), "metrics": eval_res}
            if eval_res.get("IC", 0) > self.config["ic_threshold"]:
                self.state.successful_reflections.append(log_entry)
            else:
                self.state.failed_reflections.append(log_entry)
```

---

## 8. Evolution and Feedback Loops Across the Four Approaches

To make the framework concrete, this section maps the core ideas of all four approaches onto the FactorMiner V4 loop: `initialize_search_space` -> `generate_candidates` -> `evaluate_candidates` -> `update_model`.

### 8.1 Genetic Programming (GP)

The current `MyCustomGP` implements random trees, mutation, and elitism. Crossover probabilities and population proportions in the diagram and pseudocode below explain general GP; they are neither current defaults nor an implemented crossover strategy.

**Core idea**:
Represent mathematical formulas as abstract syntax trees (ASTs). Generate many random trees initially, retain the best as elites, and produce the next generation through crossover and mutation—a Darwinian evolutionary process.

**Feedback loop**:
```text
                      [ Generate 1000 random formula trees ]
                                        |
                                        v
                      [ Compute each tree's historical IC ]
                                        |
                 +----------------------+---------------------+
                 | (discard bottom 80%)                       | (keep top 20%)
                 v                                            v
    [ Release compute and memory ]                   [ Retain elite trees ]
                                                              |
                                             +----------------+----------------+
                                             | (70% probability)               | (30% probability)
                                             v                                 v
                                 [ Cross over two trees ]             [ Mutate one random tree ]
                                             |                                 |
                                             +----------------+----------------+
                                                              |
                                                              v
                                            [ Build 1000 new formula trees ]
```

**Mapping to V4 interfaces**:
- **`initialize`**: Define the allowed constant range, base market-data columns (open, close), and operator primitives. Limit maximum tree depth.
- **`generate`**: Read the previous generation's elite trees from `MinerState.population` and produce batches of new candidates through crossover or mutation.
- **`evaluate`**: The parallel engine deserializes each AST. Nodes compute factor values across assets and time, then calculate RankIC against the configured forward-return labels.
- **`update`**: Discard the bottom 80% and store the top 20% of elite trees in `MinerState.population`.

<details><summary>Reference implementation of the underlying logic</summary>

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

### 8.2 Reinforcement Learning (RL)

The current `MyCustomRL` adjusts operator and field probability weights based on scores; the built-in `RL` is a skeleton with random actions and placeholder updates. The policy network, advantage function, and backpropagation below illustrate an extension design; Actor-Critic/PPO/DQN are not currently delivered.

**Core idea**:
Treat formula generation as a Markov decision process (MDP). An agent adds operators or data columns sequentially. The final formula's factor Sharpe ratio becomes the reward, and policy-gradient methods (PPO/DQN) update the agent.

**Feedback loop**:
```text
  +---------------------------------------------------------------------+
  |                       FactorRLAgent (Policy)                        |
  +---------------------------------------------------------------------+
        |                                                      ^
        | Action: select 'ts_mean'                              | Policy-gradient update
        v                                                      | (PPO / DQN)
  +---------------------------------------------------------------------+
  |                         Environment state                           |
  | Current prefix: "add(close, " --> assemble a complete alpha formula  |
  +---------------------------------------------------------------------+
        |
        v
  [ Compute alpha Sharpe ratio over the backtest period ] -----> Reward
```

**Mapping to V4 interfaces**:
- **`initialize`**: Initialize the agent (e.g., a PPO network) and training environment (action and state spaces).
- **`generate`**: Start with an empty formula. The network receives the current state and outputs the probability distribution for the next action. Sample actions, convert the completed formulas into a list of `FactorExpression` objects, and record `log_probs`.
- **`evaluate`**: Send sampled formulas to the backtest engine. Compute the information ratio (IR) or long–short portfolio Sharpe ratio as the absolute reward from the environment.
- **`update`**: Retrieve `log_probs` and rewards, compute the advantage function, and backpropagate using a policy-gradient algorithm to update the agent's neural-network weights.

<details><summary>Reference implementation of the underlying logic</summary>

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

> **Model extension limits**
> The reference RL implementation uses probability dictionaries for score feedback. Integrating PyTorch/PPO/DQN requires the user to implement training, candidate materialization, scoring, and restoration. An extensible interface does not mean these algorithms are implemented, and adding an import alone does not ensure compatibility.

</details>

### 8.3 Large Language Models (LLM-based Discovery)

**Core idea**:
Combine LLM in-context learning with reflection. Feed generated factors and their backtest performance back to the LLM as context, allowing it to learn from failures and avoid repeating mistakes.

**Feedback loop**:
```text
 +-----------------------------------------------------------------------+
 |                     Large language model (LLM API)                     |
 +-----------------------------------------------------------------------+
       |                                                       ^
       | 1. Generate Python factor code                        | 3. Update prompt memory
       |    from quant methods and past lessons                |    (success/failure cases)
       v                                                       |
 +-----------------------------------------------------------------------+
 |               Isolated sandbox execution and backtesting               |
 |      Compute factor IC; reject malicious commands or syntax errors     |
 +-----------------------------------------------------------------------+
       |
       +---> [ Success: IC=0.06 ] ---> Prompt: "Good; refine this idea."
       |
       +---> [ Failure: runtime error ] ---> Prompt: "Fix the dimension mismatch."
```

**Mapping to V4 interfaces**:
- **`initialize`**: Build the knowledge base and system prompt (quantitative methods, basic factor-writing rules, and documentation for available operators).
- **`generate`**: Add successes and failures from `MinerState` to the user prompt as few-shot examples. Extract plain Python code from the LLM's Markdown and wrap it in a sandboxed `FactorExpressionCode`.
- **`evaluate`**: Execute the code in `RestrictedSandbox`. On error, record the traceback as negative feedback; on success, compute IC. Record every outcome in the returned `EvaluationFeedback`.
- **`update`**: Make a text-only update without gradients. Pair generated code with its performance and store it in the success or failure pool. These memories change the context for the next LLM call.

<details><summary>Reference implementation of the underlying logic</summary>

```python
def llm_factor_mining(data, max_iterations=50):
    prompt_memory = ["You are an expert quant. Generate code. Requirement: factors must not be highly correlated."]
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
                feedback = f"Success! IC={ic}. Continue exploring different logic."
            else:
                feedback = f"Failure! Correlation is too high (corr={correlation}); try new logic."
        except Exception as e:
            feedback = f"Code execution failed: {e}. Fix the syntax or logic errors."

        prompt_memory.append({"role": "assistant", "content": factor_code})
        prompt_memory.append({"role": "user", "content": feedback})
    return discovered_factors
```
</details>

### 8.4 NN: Neural Networks and Deep Representation Learning

The deep time-series network diagram and PyTorch code below illustrate possible extensions. The current reference implementation is a NumPy MLP: it does not require a GPU by default and does not rely on the shared framework to retain a full PyTorch autograd graph.

**Core idea**:
Instead of searching for an explicit mathematical formula, feed tensor slices of market-wide stock data directly into a deep neural network. Latent variables from its final layer become cross-sectional factor scores.

**Feedback loop**:
```text
[ Market-wide 20-day price/volume tensor [B, T, F] ] ---> [ Neural layers (Transformer / GRU) ]
                                                                          |
                                                                          v
                                                       [ Latent output: factor scores ]
                                                                          |
                                                                          v
[ Configured forward-return labels ] ----------------> [ Batch IC / Rank Loss (objective) ]
                                                                          |
                                                                          v
                                                       [ Backpropagate and update weights ]
```

**Mapping to V4 interfaces**:
- **`initialize`**: Instantiate a deep time-series network (feature extractor), initialize its weights, and set up the training optimizer and loss function.
- **`generate`**: Feed market data (tensor slices) through the network in a forward pass. It outputs a continuous tensor of shape `[Batch, Seq, Output_Channels]`. Each channel represents a black-box factor expression.
- **`evaluate`**: Keep the computation graph intact. Compute the objective loss directly on the GPU from the output tensor and the configured forward-return tensor. Return the loss tensor and its graph unchanged in `EvaluationFeedback.raw_outputs`.
- **`update`**: Retrieve the loss with its computation graph and perform standard backpropagation: `loss.backward()` propagates gradients, then `optimizer.step()` updates the model weights.

### 8.4.1 Training Tensors Versus Persistable Factors

Intermediate NN training outputs cannot be saved directly as factors. The reference `MyCustomNNMiner` enters its training branch using `channel=-1` and the `NNTrainingTensor.requires_grad` flag. This is a training payload: the NumPy model updates parameters in its own train method, rather than retaining a general autograd graph.

After each weight update, the Miner materializes each output channel of a frozen model snapshot as `FactorExpressionTensor(model_version_id, channel_idx)`. A single-instrument channel returns a `pd.Series` aligned with the market-data index; a cross-asset channel returns a time × asset `pd.DataFrame`. The model fits weights and normalization parameters only on `mine_period`. If test_period is provided, its metrics select channels; otherwise the training period is used. Because test_period participates in selection, it serves as a validation set; final conclusions still require independent data excluded from selection. Candidates across iterations are filtered by output correlation, retaining the Top-K.

`model_version_id` is derived from a digest of the complete model package. Each channel also has a logic hash derived from `{"model_version", "channel"}`. On save, `FactorMinerDirector` writes the restorable package to `factor_db/models/<model_version>.npz`, containing weights, biases, feature order, normalization parameters, data mode, and schema version. New channels use `{"type": "nn_channel", "model_version": ..., "channel": ...}`, with optional `model_file`, `model_format`, and feature metadata. Historical `dl_channel` metadata remains readable; old bare weights without a portable model package cannot be restored by guessing.

<details><summary>Reference implementation of the underlying logic</summary>

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

### 💡 Summary: The Four Approaches in the V4 Loop

| Approach | Output of `generate_candidates` | Main role of `evaluate_candidates` | Action in `update_model` | State held in `MinerState` |
| :--- | :--- | :--- | :--- | :--- |
| **Reference GP** | Random/mutated ASTs | Compute IC/RankIC/fitness in a thread pool | Elite selection | Population; archives retain logic and declared parent references |
| **MyCustomRL** | ASTs with action records | Common correlation-based scoring | Update operator/field probability weights | Runtime sampling weights and elites; no policy-checkpoint guarantee |
| **LLM** | Python source extracted from Markdown | Run sandboxed code, reject unsafe operations, compute IC | Turn performance or errors into textual reflections | Few-shot examples of successes and failures |
| **Reference NN** | Training payload, then materialized output channels | Score channels on the current evaluation slice | Train the NumPy model | Portable model package, channels, and input definitions |

---

## 9. Unified Data Pipeline and Cross-Market Alignment

In V4, reliable models depend on a robust data layer. The `data_feed` module retrieves historical data, aligns time zones in memory, fills gaps, and supplies dependent metadata choices to the frontend.

### 9.1 Liquidity-Aware Metadata Routing
The frontend obtains USDT instruments and timeframes from `/api/exchange_meta`. When metadata and tickers are available, it sorts by 24h quoteVolume and returns at most 200 instruments; otherwise it uses a sorted or static list. A static list does not prove exchange connectivity, instrument support, or a current liquidity ranking. The Web UI allows one market type and multiple instruments and timeframes.

### 9.2 Consistency Between Persistence and Retrieval
CCXT symbol formats vary by market (e.g., futures `1000CAT/USDT:USDT` versus spot `1000CAT/USDT`). V4 therefore defines a strict conversion from in-memory symbols to filenames:
`safe_symbol = symbol.replace('/', '_').replace(':', '_')`

The only permitted storage format is `{safe_symbol}-{timeframe}-{trade_type}.feather`. For example, the perpetual contract `BTC/USDT:USDT` must be saved as `BTC_USDT_USDT-1m-futures.feather`; the reader does not fall back to legacy filenames.
The `Batch Downloader`, incremental downloader, `Real Data Client`, and `Gap Filler` must all use this rule to keep download and read paths consistent. Actual data quality still requires health checks.

The current downloader interprets date arguments and timezone-naive market timestamps as UTC. Timezone-aware timestamps in old files are converted to UTC before removing timezone information; gap detection, requests, and merging retain the same real instants. Gap filling requests only `[first missing candle, next existing candle)`, reuses the download task's original CCXT instance, and merges/saves under that instance's exchange identity and the task's market type. Three consecutive responses without forward progress raise `DataGapFillError`. The download entry point returns failure and stops before saving, preserving the original file; it does not swallow the error and repeat automatic repair.

Standard and batch downloads share pagination logic, advancing from the last received candle to explicit UTC `[start, end)` request boundaries. Before saving, completeness is checked against the expected candle count for that interval, including missing leading and trailing candles. This download-completeness denominator differs from the research definition of factor Coverage. Actual OHLC consistency counts contribute to the health score; invalid price relationships cannot pass the final save gate. Both downloaders write to unique temporary Feather files in the destination directory and replace the target atomically. Write failures propagate the original error, remove the temporary file, and preserve existing data. Old market data is normalized to UTC using the same `date → time → timestamp(ms)` priority as the reader and merged before health checks. Unreadable files, unrecognized timestamps, or missing times stop the operation. Only validated data is published after the health gate: unchecked old rows are not mixed back in, and a failed merge is not saved elsewhere and reported as success.

Atomic replacement guarantees completeness of a single file publication only. To preserve intervals from concurrent merges, both download entry points share an in-process `RLock` keyed by the target path after `Path.resolve()`. The lock is held from the first read of old data through merging, gap filling, health checks, and publication, and is released on exceptions and failure returns. The main download request runs outside the lock, allowing different files to proceed in parallel. Each call fixes its save path from the actual exchange.id, symbol, timeframe, and trade_type, then passes it to merge/save helpers so another thread cannot redirect storage by changing identity fields on a shared downloader. This protects threads within one process only; no lock files or cross-process/cross-machine write protocol have been added.

`DataDownloader.download_ohlcv` accepts UTC date or timestamp boundaries and selects the opening times of fixed-interval candles within the range. The reader converts an inclusive period into the half-open request `ceil(start)` to `floor(end) + interval` and passes the actual exchange; existing slicing and label semantics are unchanged. `overwrite` branches before the first merge so old intervals cannot enter the result. Nonfinite values in required OHLCV fields are rejected independently of the missing-data threshold. Fixed-duration timeframes currently extend through 1d; weekly/monthly calendar semantics are outside this change.

### 9.3 Integrated Data Completion Console
Data Center displays progress, gap filling, and error summaries within the page. Download combinations within a batch run sequentially; independent tasks may run in parallel. The console is not a complete per-candle log archive, and percentages do not substitute for outcome status.

The download backend captures the event loop in the main coroutine, and worker threads submit progress through thread-safe scheduling. For Web mining in `sequential_single` mode, each pair gets its own client and Director; results are combined and overall progress is scaled by the number of instruments. `cross_asset` uses one joint run. These are the only valid values for `data_feeds.mining_mode`, which defaults to `sequential_single` when omitted. Shared launch validation and RealDataClient construction reject unknown strings and wrong types before reading market data, preventing a run from reading only the first instrument while recording multi-instrument lineage. Data reading, RealDataClient / Director construction, and execution occur within one worker-thread call so market-data reads or automatic downloads do not block the API event loop. Each Director's configuration contains only its actual input pairs, keeping lineage accurate.

Mining tasks also drain worker-thread broadcasts before publishing a terminal status. Task logs are assigned by context, and the evaluator copies the calling context for each concurrent candidate. The shared logger's level is restored after the last active task finishes, avoiding cross-task logs or premature log loss. Candidate results retain submission order, and the evaluation thread count follows evaluation.max_workers.

Download tasks have unique `task_id` values and statuses `running / completed / failed / partial_failed`. They wait for submitted thread-progress broadcasts before publishing a terminal status. The frontend consumes only events for the current task and uses explicit terminal statuses to display success or failure. CLI downloads aggregate results across combinations and exit with code `1` if any fail.

WebSocket support includes an optional per-event callback while preserving other pages' `lastMessage` interface. During HTTP submission, the download page buffers events by task ID. After the response, it replays only the current task's events and clears the buffer. Late progress cannot overwrite a confirmed terminal status. Submission errors or retries clear old state.

Launchpad likewise processes task, log, and scatter events individually so React batch rendering does not retain only the last message. Its initial HTTP snapshot adds rows by task ID only when they are not already present, preserving live state received during the request. Late progress cannot overwrite completed / failed / completed_empty in the table or selected task. Download events update the latest snapshot in `TaskManager.download_tasks` before broadcasting. On connection, WebSocket replays those `download_progress` snapshots and existing `TaskManager.tasks` as normal `task_update` events, restoring terminal statuses missed during disconnection. The download monitor reports disconnection and reconnection without changing task success/failure. State lives only in the current API process; task history is not restored after restart, and no persistent task service has been added.

The current effect owns the connection: on unmount or URL change, it closes that connection, cancels its reconnect timer, and ignores late events. Ordinary disconnections continue to reconnect under the existing policy.

---

## 10. Operator Extensions, Data Configuration, and CLI

### 10.1 Registries and Input Contracts

`OperatorRegistry` dispatches unary/binary operators, and `EvaluatorRegistry` registers scoring functions. The example shows their public interfaces without duplicating the core registry implementation:

```python
from core.miner.registry import OperatorRegistry, EvaluatorRegistry

@OperatorRegistry.register(arity=1)
def rolling_mean_10(series):
    return series.rolling(10).mean()

@EvaluatorRegistry.register_fitness_hook("rankic_score")
def rankic_score(factor_values, returns, base_metrics):
    return abs(base_metrics["RankIC"]) * 100
```

Operators support only arity=1 or 2, with matching signatures; multi-argument ASTs are not implemented. Outputs must preserve input indices, asset columns, and dimensionality. Normal warm-up NaNs may remain. `required_streams` supplies candidate terminals, model features, or LLM prompt fields for reference Miners; it is not a global read allowlist.

### 10.2 Data Configuration and Public Entry Points

`RealDataClient` reads real data from Feather files under `data/`, using exchange, market, symbol, and timeframe. Sequential mode runs each instrument separately; cross-sectional mode aligns time and assets. Missing files may trigger automatic downloads. Ordinary factor computation can run offline when all required local data is present.

The following is a data configuration fragment. See the [product guide](../FactorMiner_Product_Report_EN.md) for complete runnable examples:

```json
{
  "data_feeds": {
    "exchange": "binance",
    "instrument_type": "futures",
    "timeframe": "1h",
    "pairs": ["BTC/USDT:USDT"],
    "required_streams": ["close", "volume"],
    "mine_period": [["2024-01-01 00:00:00", "2024-01-07 23:00:00"]],
    "test_period": [["2024-01-08 00:00:00", "2024-01-10 23:00:00"]],
    "mining_mode": "sequential_single"
  }
}
```

Times are interpreted as UTC. Mining/test slices include the final candle, and labels are constructed independently within each interval. The CLI entry point is `factorminer`, or equivalently `python -m core.cli`. Once the required market data and extensions are ready:

```bash
python -m core.cli mine --miner MyCustomGP --config user_workspace/configs/config.json --user-dir user_workspace
```

Data Center shows the file range and record count for each combination. API batches process downloads sequentially, with WebSocket status broadcasts. merge combines data, overwrite replaces it, and fill_gap requests missing data; see the pipeline section above for current fixed-timeframe and atomic-write constraints.

### 10.3 User Workspace and Declarative Configuration

Users can define configuration and extensions in the workspace without changing core scheduling. Extensions are dynamically imported into the main process, so load only trusted code; separate directories do not provide security isolation.

**User workflow:**

1. **Write configuration**: Researchers create JSON files in `user_workspace/configs/`; current entry points do not parse YAML.
2. **Develop operators and Miners**: Place custom `.py` files in `user_workspace/custom_miners/` and `user_workspace/custom_operators/`, registering them with `@MinerRegistry.register` or `@OperatorRegistry.register`.

```python
# File: user_workspace/custom_miners/my_gp.py
from core.miner.registry import MinerRegistry
from core.miner.paradigms.gp_miner import GPFactorMiner

@MinerRegistry.register("MyCustomGP")
class FirstGPMiner(GPFactorMiner):
    def generate_candidates(self):
        # User-defined mutation logic
        return super().generate_candidates()
```

3. **Run through the CLI**:
   Run the command from the repository root. At startup, `DynamicLoader` scans and loads files under `user_workspace/`, allowing extensions to run without modifying the framework.

```bash
factorminer mine --miner MyCustomGP --config user_workspace/configs/demo_config.json --user-dir user_workspace
```

This architecture separates the responsibilities and directories of **platform developers (maintaining `core/`)** and **strategy researchers (maintaining `user_workspace/`)**.

**Extension result contracts:**

- Custom Miners must implement initialization, candidate generation, evaluation, and updating. After updating a strategy, prompt, or model weights, they must place final scored candidates in `MinerState.population` or `replay_buffer`; otherwise the Director has no research artifacts to persist.
- The common AST runtime consumes both built-in and user operators from `OperatorRegistry`. GP/RL generates unary or binary nodes based on arity, and `FactorExpressionAST.compute()` dispatches node names through the same registry. Registered operators therefore participate in candidate generation and evaluation, rather than merely being discovered.
- The default time-series search palette for `MyCustomGP` includes `custom_ts_decay`, `ts_zscore_20`, `ts_delta_5`, `ts_rank_20`, and `ts_volatility_20`. It can be explicitly narrowed or expanded with `search_space.allowed_operators`.
- Hooks registered with `EvaluatorRegistry.register_fitness_hook(name)` receive `factor_values, returns, base_metrics` and may return a number or a dictionary containing `fitness_score`. Task configuration selects the hook through `fitness.hook`.
- NN implementations must distinguish temporary training tensor models from evaluable output channels. Only materialized, evaluated, and retained channel expressions become factor archives; their weights are stored as traceable attachments.

**Startup validation and verification:**

- DynamicLoader returns a module-loading report. Before creating a data client, CLI and WebUI Launchpad aggregate module-import errors and validate Miner availability, basic data_feeds fields, operator registration/arity, and Fitness Hook names.
- Extension modules are registered in `sys.modules` before execution. Every import-exception path restores or removes the entry, supporting dataclasses with postponed annotations. Ordinary errors are aggregated and prevent startup; `SystemExit` / `KeyboardInterrupt` propagate unchanged after cleanup. AST references to nonexistent input streams raise explicit errors. Numeric terminal `0` and normal rolling warm-up NaNs retain their existing semantics.
- Operators may register only arity 1 or 2, and function signatures must accept the declared number of arguments. Fitness Hooks must accept factor_values, returns, and base_metrics. Unknown configuration fails with actionable errors rather than falling back to zero factors or default scoring.
- tests/test_cli_smoke.py writes minimal Feather data in temporary directories and invokes the public CLI to verify that MyCustomGP, MyCustomRL, MyCustomLLM, and NN each complete one iteration and persist factor metadata.
- Archived metrics from actual runs use stable keys: `IC`, `RankIC`, `Turnover`, and `fitness_score`. Frontend summaries, task results, and Inspector must consume this contract directly, never read nonexistent lowercase aliases and misrepresent missing values as zero. Restart FastAPI after Python execution-path changes. Restarting discards only in-memory `TaskManager` trackers; `LocalFactorStorage` metadata and values snapshots remain.
- A scalar candidate succeeds only if `fitness_score` is finite. NaN, Infinity, or missing scores from a custom Hook enter the existing per-candidate error-feedback path. Zero scores, negative scores, and additional diagnostic metrics retain their values; the training-stage gradient-tensor branch retains its existing interface.
- Default HTTP responses, WebSocket broadcasts, and connection replay share recursive JSON conversion: nonfinite floating-point diagnostic values become `null` in the transport copy, while finite values in dictionaries, lists, and nested structures remain unchanged. Conversion does not mutate in-memory tasks or original archives. Scoring still requires finite `fitness_score`; display needs do not relax that gate.

The README directory map follows these responsibilities: `core/` runs research, `api/` provides services and task observability, `web/` is the research workbench, `user_workspace/` contains user extensions for strategies, operators, and scoring, and `factor_db/` stores traceable research artifacts only.

---

## 11. Factor Restoration and White-Box Inspection (Compiler & Inspector)

After the common loop produces a collection of factors, users still need to inspect and use them. FactorMiner V4 provides `FactorInspector` and `FactorCompiler` for these tasks.

### 11.1 FactorCompiler: Restoring Local Callables

Currently restorable archives include valid ASTs, restricted Python source, and complete NN model artifacts with a loader contract. Restoration does not mean connection to live trading.

`FactorCompiler` reconstructs these archives as local Python callables. C++ compilation, streaming tick engines, and trading gateways are outside the current implementation.

```python
class FactorCompiler:
    """
    Factor restorer: reconstruct persisted logic as a local callable.
    """
    def __init__(self, storage_client, sandbox):
        self.storage = storage_client
        self.sandbox = sandbox

    def compile_for_live_trading(self, factor_id: str):
        metadata = self.storage.get_metadata(factor_id)
        logic = metadata.logic_reference

        # Dispatch on stable logic_reference.type, not the customizable Miner name.
        if logic["type"] == "json_ast":
            return FactorExpressionAST(logic["ast"]).compute
        if logic["type"] == "python_source":
            code = self.storage.load_llm_source(logic["source_file"])
            self.sandbox.validate_code(code)
            return FactorExpressionCode(code, sandbox=self.sandbox).compute
        if logic["type"] in {"nn_channel", "dl_channel"}:
            model = load_portable_model(self.storage, logic)
            return lambda data: model.predict_channel(data, logic["channel"])
        raise NotImplementedError("This artifact cannot be reconstructed as an executable factor")

    def deploy_to_live_server(self, factor_id: str, server_target: str):
        self.compile_for_live_trading(factor_id)  # Only verify reconstruction
        raise NotImplementedError("Deployment transport is not implemented; the factor was not pushed")
```

The current repository only reconstructs persisted artifacts safely as local callables. Image builds, signing, approvals,
delivery, rollback, and remote health checks are not implemented, so the interface must not report “deployment successful.” Once a real transport layer is added,
success may be reported only after the remote node acknowledges the deployment and its version is recorded.

AST restoration accepts field strings, numeric constants, and valid operator trees, with structural validation and runtime execution using the same operator registry. The CLI Resolver can use a registered `model_format` to load complete `nn_channel` / `dl_channel` artifacts and infer the saved channel. Old bare weights, missing files, invalid channels, and unknown formats fail explicitly. The current CLI Inspector supplies one instrument at a time and rejects cross-asset NN reevaluation. Direct Resolver calls still require the complete input structure expected by the model. Displaying stored Web snapshots does not require new inference.

Inspector exchange and market arguments override configuration only when explicitly supplied; Binance/futures defaults apply afterward. Real-data reads and slice boundaries use UTC consistently to maintain label alignment. Per-instrument failures return `error`, and reports display unavailable results with reasons rather than inventing zero metrics. Other instruments continue; any failure makes the CLI return `1`.

Inspector reuses the evaluator's numeric and axis-consistency checks, then requires at least 10 jointly valid observations. No valid output, Infinity, misalignment, or insufficient samples produces an unavailable result while preserving metric formulas and the Coverage denominator. The inspect entry point consumes the dynamic-module loading report and exits before constructing the engine or reading data if imports fail. An explicit configuration must be readable and a valid JSON object or execution likewise stops; defaults apply only when configuration is omitted. The Resolver checks whether a direct AST argument is `None`, so numeric terminal `0` is not lost.

Inspector passes the requested periods to Lag computation, constructs each horizon's labels independently per interval, and then concatenates them. Even adjacent intervals retain their own ending NaNs when explicitly separate. Grouping still uses the original qcut algorithm. Missing groups and Long_Short values depending on them are `None`, displayed as `N/A` in both Rich and plain text; no zero returns are invented, while actual zeros remain. Cross-asset group returns are not implemented, and their former placeholders are also marked unavailable.

CLI Inspector ranks independently within each rolling Spearman window: the latest 20 jointly valid observations, with a minimum of 10. It does not use full-period ranks for historical windows. IR uses a fixed sqrt(252) multiplier and is not calibrated to the timeframe or trading calendar. Default Lags are 1, 2, 3, 5, and 10. Output Coverage, jointly valid Coverage, and file coverage are different concepts; see the scoring guide.

### 11.2 FactorInspector: White-Box Inspection

After `director.run()`, a console message claiming “50 holy-grail factors found” is not enough for risk review in a real quant team. Researchers need to open the black box, inspect the factor's structure and quantile returns, and assess overfitting.

The current CLI uses `core/inspector/`; the Web UI analyzes snapshots through `core/analysis/factor_analytics.py`. The code below is an early visualization design illustration. Alphalens, SHAP, and the private plotting methods shown here are not integrated; available features are listed in the Web section below.

```python
class FactorInspector:
    """White-box factor inspection and visualization panel."""
    def __init__(self, storage_client: FactorStorageInterface):
        self.storage = storage_client

    def show_tearsheet(self, factor_id: str):
        """Generate a full factor diagnostics report (tearsheet)."""
        # 1. Load computed values (Parquet)
        factor_values = self.storage.load_factor_values(factor_id)

        # 2. Use libraries such as Alphalens to plot quantile returns, IC decay, and turnover
        self._plot_quant_metrics(factor_values)

        # 3. Restore the logic for a readable display
        metadata = self.storage.get_metadata(factor_id)
        if metadata.miner_type == "GP":
            # Render the AST as a human-readable formula (e.g., LaTeX through SymPy)
            print("Factor formula: ", self._ast_to_formula(factor_id))
        elif metadata.miner_type == "LLM":
            print("Generated code:\n", self.storage.get_llm_logic(factor_id))
            print("\nLLM reflection log:\n", self.storage.get_llm_reflection(factor_id))
        elif metadata.logic_reference.get("type") in {"nn_channel", "dl_channel"}:
            print(f"NN feature channel: {metadata.logic_reference['channel_index']}")
            # Can be extended with feature attribution (Feature Importance / SHAP values)
```

---

## 12. Implementation Status

The following reference entry points are implemented. They do not constitute acceptance of every algorithm illustration, planned diagram, or trading capability described above:

- **GP (MyCustomGPMiner)**: Implements random ASTs, mutation, and elite selection. Full crossover strategies remain an extension direction.
- **RL (MyCustomRLMiner)**: Demonstrates the policy-gradient idea through a lightweight reinforcement-learning optimization loop based on probability-weight dictionaries, with no PyTorch dependency.
- **LLM (MyCustomLLMMiner)**: Provides code generation, sandbox feedback, and reflection. Real experiments use require_live_api=true and allow_fallback=false. Missing credentials prevent startup; failed requests are not passed off as model candidates. Offline tests do not establish access to a real model service.
- **NN (MyCustomNNMiner)**: Uses a NumPy MLP and training payloads, updating weights inside the model. After training, outputs are materialized by channel, scored, and filtered to retain Top-K. Weights and channel metadata are persisted together, completing the common CLI and Web result path.
- **Parallel evaluation and sandbox**: Common metrics, a finite primary-score gate, and custom Hooks. Source allowlists and separate processes do not replace reviews for look-ahead bias, overfitting, or platform security.
- **Factor attribution and inspection engine (FactorInspectorEngine)**: Implements the standalone `core/inspector/` module and `factorminer inspect` CLI command. It resolves factors from Factor IDs, AST dictionary strings, or Python source; computes Coverage, Pearson/RankIC (Mean/Std/IR/t-stat), decay at default Lags 1, 2, 3, 5, and 10, 5-Quantile group returns, and turnover across multiple coins and out-of-sample (OOS) periods; and renders terminal cards with Rich.

User extensions and configuration live in user_workspace; the core handles scheduling, scoring, and archiving. See the [product guide](../FactorMiner_Product_Report_EN.md#validation-and-current-limits) for complete test commands. Offline tests do not demonstrate trading returns or validate every private extension.

---

## 13. Web UI and Separate Frontend/Backend Architecture

FactorMiner V4 uses a **React (Vite) + FastAPI** console with separate frontend and backend, retaining the current CLI and replacing the historical Flask/Streamlit interfaces.

### 13.1 Page Space and Layout
A **top tab bar** replaces the traditional left sidebar, freeing horizontal space for wide tables, long-history ECharts, and lengthy stack information.

### 13.2 Command Center
- **Background heartbeat**: Detects the `Engine Online` status.
- **Overview dashboard**: Powered by FastAPI `/api/dashboard`, showing tasks, factor inventory, lifecycle states, and high-scoring candidates.
- **Activity timeline**: A Timeline component records the progress of asynchronously scheduled backend tasks.

The current Research Dashboard uses `GET /api/dashboard` as a single aggregated snapshot. Task data comes from the in-memory `TaskManager` (running tasks and recent execution history); factor data comes from persisted `factor_db/metadata` (total count, Miner distribution, lifecycle coverage, and candidates ranked by fitness). After a service restart, the historical task timeline is cleared under the existing in-memory policy, but factor archives and the research-quality overview remain. The frontend displays only these two actual sources, links factors to Inspector, and links task entry points to Launchpad.

### 13.3 Asynchronous Scheduling and Real-Time WebSocket Events
Long-running tasks submitted through conventional Web handlers can time out or block the main thread. V4 uses asynchronous dispatch with a thread pool:
1. **Dispatch**: The frontend posts JSON (Miner & Config) to `/api/launch`. FastAPI BackgroundTasks calls asyncio.to_thread, placing synchronous data reading, initialization, and execution in a worker thread.
2. **Track state**: Tasks are registered in the global in-memory `TaskManager` dictionary.
3. **Push live events**: Epoch callbacks broadcast status and the best candidate's IC/actual complexity through the event loop. Launchpad modal details consume each event. Color maps to complexity, not lineage or returns. There is no microsecond-latency guarantee.

### 13.4 Factor Inspector Phase II: Reproducible Snapshots, Tearsheets, and Batch Review

Inspector must not treat temporary results in `TaskManager` as factor archives: that memory disappears on service restart and excludes factors produced by the CLI. Phase I uses `factor_db/metadata/*.json` in `LocalFactorStorage` as the sole catalog source and provides these APIs:

- `GET /api/factors`: Return searchable/filterable factor summaries ordered by creation time, IC, or fitness;
- `GET /api/factors/{factor_id}`: Return metadata, logic references reconstructed by storage type, and whether a factor-value inspection snapshot exists;
- `PATCH /api/factors/{factor_id}/lifecycle`: Persist manual review states such as `DISCOVERED → INSPECTED → PAPER_TRADING → LIVE / RETIRED`.

The frontend renders white-box content by `logic_reference.type`, not a broad Miner name: `json_ast` displays the AST tree and expression; `python_source` reads a controlled source file; `rl_actions` shows the action trace; `nn_channel` shows model version, output channel, and weight-artifact availability. Historical `dl_channel` is normalized to `nn_channel` in the API layer. Saved factor IDs in Launchpad link directly to their archives via `/inspector?factor=<id>`.

In Phase II, after the Director persists a candidate's logic artifact, it materializes factor values again using the same candidate and data client, aligns them with forward returns from `RealDataClient.get_returns()`, and writes `factor_db/values/<factor_id>.parquet`. The standard schema is `timestamp, asset (cross-sectional mode), factor, forward_return`. `FactorMetadata.data_lineage` also records the data source, pairs, market type, timeframe, input streams, sample range, normalized `target`, and exact return formula. The unified Target Builder supports `current_close` / `next_open` entry, `close` exit, configurable `horizon_bars`, and simple / log returns. When configuration is omitted, it preserves the historical `close[t+1] / close[t] - 1` behavior. Labels are generated independently within each data interval; Inspector defaults to the archived target definition when using a factor ID.

A real-Feather regression for MyCustomGP should verify the full loop: configured user operators generate ASTs with valid arity; the AST runtime produces aligned series; the Evaluator writes the four metrics above; and the Director persists the same candidate's logic, metrics, data lineage, and values/forward-return snapshot. A successful module import or completed task alone does not establish that this path works.

If a candidate snapshot fails, diagnostic metadata is retained, and `pending` / `failed` archives cannot be analyzed. Other candidates in the batch continue, and completed archives remain. The CLI exits with failure and lists successful and failed IDs in local logs. The Web task reports failure with sanitized public errors. Older archives use their existing snapshots.

The Tearsheet API analyzes only these Parquet snapshots. For a single instrument it computes rolling time-series IC, period-to-period factor-change turnover, and full-sample quantile mean forward returns. For cross-sectional mode it computes rolling means of per-period cross-sectional IC, asset-matrix turnover, and mean forward returns of per-period quantile portfolios. Missing snapshots produce an explicit instruction to mine again; simulated curves are never substituted.

Inspector can compare 2–5 factors with snapshots through the comparison API, aligning and overlaying rolling IC on a common basis. It also provides a batch lifecycle-update API. Tests use temporary FactorStorage and FastAPI TestClient instances to cover snapshot schemas, sequential/cross-sectional analysis, comparison, and batch-review endpoints.

### 13.5 Three-Language UI and Research Artifact Boundaries

The frontend uses a lightweight local i18n Context supporting `zh`, `en`, and `de`. The initial language is inferred from the browser; a choice in the top navigation is saved to `localStorage`. Translation dictionaries are centralized in `web/src/i18n.tsx` to avoid scattered conditional logic. Initial coverage includes global navigation, Research Dashboard, and Factor Inspector; later pages extend the same key space.

The language layer handles operational UI text only: navigation, buttons, status descriptions, filters, empty states, and explanatory hints. Factor IDs, logic hashes, pairs, ASTs/expressions, code, model versions, weight files, and research metric abbreviations such as IC and RankIC remain untranslated artifacts. Backend status enums stay stable. Some logs and error messages retain their original language; UI translation does not change research artifacts.
