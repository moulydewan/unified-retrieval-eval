# src/managers/strategy_manager.py
import yaml

from src.experiments.rag import run_bm25_rag
from src.experiments.denseretrieval import run_dense_retriever
from src.experiments.human_proxy_rag import run_human_proxy_rag
from src.experiments.agent_peer import run_peer_agents
from src.experiments.agent_hierarchy import run_agent_hierarchy_strategy
from src.experiments.agent_functional import run_functional_agents


class StrategyManager:
    def __init__(self):
        self.registry = {
            "rag": run_bm25_rag,
            "denseretrieval": run_dense_retriever,
            "human_proxy_rag": run_human_proxy_rag,
            "peer_agents": run_peer_agents,
            "hierarchy_agents": run_agent_hierarchy_strategy,
            "functional_agents": run_functional_agents
        }
        self._persona_db = None  # lazy cache

    def _load_persona_db(self, personas_path="configs/personas.yaml") -> dict:
        if self._persona_db is None:
            with open(personas_path, "r") as f:
                self._persona_db = yaml.safe_load(f) or {}
        return self._persona_db

    def _normalize_persona(self, p: dict) -> dict:
        p = dict(p)
        if isinstance(p.get("persona_roles"), str):
            lines = [ln.strip() for ln in p["persona_roles"].splitlines() if ln.strip()]
            p["persona_roles"] = [
                ln[2:].strip() if ln.startswith("- ") else ln for ln in lines
            ]
        return p

    def _load_persona(self, persona_key, personas_path="configs/personas.yaml"):
        if persona_key is None:
            return None

        if isinstance(persona_key, str):
            keys = [p.strip() for p in persona_key.split(",") if p.strip()]
        elif isinstance(persona_key, list):
            keys = persona_key
        else:
            raise ValueError("persona_key must be a string or list of strings")

        persona_db = self._load_persona_db(personas_path)

        missing = [k for k in keys if k not in persona_db]
        if missing:
            raise ValueError(f"Persona(s) not found in configs/personas.yaml: {missing}")

        personas = [self._normalize_persona(persona_db[k]) for k in keys]
        return personas[0] if len(personas) == 1 else personas

    def run_strategy(self, name, bundle, model_cfg, strategy_cfg, dataset_cfg, top_k):
        """
        All strategies take IRDatasetBundle.
        """
        if name not in self.registry:
            raise ValueError(f"Unknown strategy '{name}'")

        pipeline_fn = self.registry[name]

        # Copy to avoid accidental cross-run mutation
        strategy_cfg = dict(strategy_cfg) if strategy_cfg else {}
        dataset_cfg = dict(dataset_cfg) if dataset_cfg else {}

        # If dataset_cfg carries a per-dataset pyserini index, let it satisfy rag's requirement.
        if name in {"rag", "human_proxy_rag", "functional_agents"}:
            if "pyserini_index_dir" not in strategy_cfg and "pyserini_index_dir" in dataset_cfg:
                strategy_cfg["pyserini_index_dir"] = dataset_cfg["pyserini_index_dir"]
            if "pyserini_prebuilt_index" not in strategy_cfg and "pyserini_prebuilt_index" in dataset_cfg:
                strategy_cfg["pyserini_prebuilt_index"] = dataset_cfg["pyserini_prebuilt_index"]

        kwargs = dict(
            bundle=bundle,
            model_cfg=model_cfg,
            strategy_cfg=strategy_cfg,
            dataset_cfg=dataset_cfg,
            top_k=top_k
        )

        if name == "human_proxy_rag":
            persona_key = strategy_cfg.get("persona_key")
            personas_path = strategy_cfg.get("personas_path", "configs/personas.yaml")
            if not persona_key:
                raise ValueError("human_proxy_rag requires persona_key in strategies.yaml")
            kwargs["persona"] = self._load_persona(persona_key, personas_path)

        return pipeline_fn(**kwargs)
