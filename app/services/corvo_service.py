import sys
import os
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, List

# Import Config from corvo-api first (before modifying sys.path)
from app.config import Config

# Add Corvo to Python path BEFORE importing corvo modules
CORVO_PATH = Config.CORVO_PATH
corvo_path_str = str(CORVO_PATH)
if corvo_path_str not in sys.path:
    sys.path.insert(0, corvo_path_str)

# Temporarily remove corvo-api's 'app' module from sys.modules to avoid conflicts
# Save it first so we can restore if needed
_api_app_module = sys.modules.get('app')
_api_config_module = sys.modules.get('app.config')
if 'app' in sys.modules:
    del sys.modules['app']
if 'app.config' in sys.modules:
    del sys.modules['app.config']

# Now import from corvo system's app module
try:
    from app.utils import STEmbedding
    from app.normalization import normalize_vendors
    from app.categorization import (
        categorize_transactions, 
        build_pairs, 
        build_corpus_from_taxonomy,
        RuleEngine,
        rule_based_classify,
        ml_categorize,
        llm_categorize
    )
    from app.database import get_milvus_collection
    from app.config import (
        NEW_DB, MAIN_DB, EMBED_MODEL_PATH,
        HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
        HDBSCAN_ALLOW_SINGLE_CLUSTER, VECTOR_TOP_K, VECTOR_MIN_SIM,
        ALIAS_BOOST_STRONG, ALIAS_BOOST_WEAK, ALIAS_THRESHOLD,
        TAXONOMY_FILE, IR_FILE, CATEGORY_BATCH_SIZE, CATEGORY_TOP_K,
        CATEGORY_USE_ML, CATEGORY_USE_LLM, CATEGORY_VENDOR_COL, CATEGORY_DESC_COL,
        CLUSTER_CONFIDENCE_THRESHOLD, NORMALIZATION_CONFIDENCE_THRESHOLD,
        USE_OUTLIER_SCORES, NOISE_BASELINE_CONFIDENCE,
        MILVUS_BATCH  # Added for vectordb.py
    )
finally:
    # Restore corvo-api's app module if it existed
    if _api_app_module is not None:
        sys.modules['app'] = _api_app_module
    if _api_config_module is not None:
        sys.modules['app.config'] = _api_config_module


class CorvoService:
    """Service layer that wraps Corvo functionality for API access."""
    
    _instance = None
    _embedder = None
    _milvus_client = None
    _rules_engine = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CorvoService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print(f"Initializing CorvoService...", flush=True)
        print(f"Corvo path: {CORVO_PATH}", flush=True)
        
        # Initialize embedder
        model_path = CORVO_PATH / EMBED_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model not found: {model_path}")
        
        print(f"Loading embedding model from: {model_path}", flush=True)
        print("This may take several minutes...", flush=True)
        self._embedder = STEmbedding(str(model_path))
        print(f"Embedding model loaded. Dimension: {self._embedder.embedding_dimension}", flush=True)
        
        self._initialized = True
    
    @property
    def embedder(self):
        return self._embedder
    
    def _get_milvus_client(self):
        if self._milvus_client is None:
            self._milvus_client = get_milvus_collection(
                uri=Config.MILVUS_URI,
                token=Config.MILVUS_TOKEN,
                collection_name=Config.MILVUS_COLLECTION,
                vector_dim=self._embedder.embedding_dimension,
            )
        return self._milvus_client
    
    def _get_rules_engine(self, taxonomy_file: Optional[str] = None):
        """Lazy load rules engine"""
        if self._rules_engine is None or taxonomy_file:
            taxonomy_path = taxonomy_file or str(CORVO_PATH / TAXONOMY_FILE)
            self._rules_engine = RuleEngine()
            self._rules_engine.load_csv(taxonomy_path)
        return self._rules_engine
    
    def normalize_vendors(
        self,
        df: pd.DataFrame,
        vendor_column: str,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        top_k: Optional[int] = None,
        min_sim: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Normalize vendor names in a DataFrame."""
        client = self._get_milvus_client()
        
        return normalize_vendors(
            df_raw=df,
            milvus_client=client,
            collection_name=Config.MILVUS_COLLECTION,
            vendor_column=vendor_column,
            embedder=self._embedder,
            min_cluster_size=min_cluster_size or HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=min_samples or HDBSCAN_MIN_SAMPLES,
            allow_single_cluster=HDBSCAN_ALLOW_SINGLE_CLUSTER,
            top_k=top_k or VECTOR_TOP_K,
            min_sim=min_sim or VECTOR_MIN_SIM,
            alias_boost_strong=ALIAS_BOOST_STRONG,
            alias_boost_weak=ALIAS_BOOST_WEAK,
            alias_threshold=ALIAS_THRESHOLD,
            cluster_conf_threshold=kwargs.get('cluster_conf_threshold', CLUSTER_CONFIDENCE_THRESHOLD),
            normalization_conf_threshold=kwargs.get('normalization_conf_threshold', NORMALIZATION_CONFIDENCE_THRESHOLD),
            use_outlier_scores=kwargs.get('use_outlier_scores', USE_OUTLIER_SCORES),
            noise_baseline_conf=kwargs.get('noise_baseline_conf', NOISE_BASELINE_CONFIDENCE),
        )
    
    def categorize_transactions(
        self,
        df: pd.DataFrame,
        vendor_col: str = CATEGORY_VENDOR_COL,
        description_col: str = CATEGORY_DESC_COL,
        taxonomy_file: Optional[str] = None,
        ir_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
        use_ml: Optional[bool] = None,
        use_llm: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Categorize transactions using full cascading pipeline."""
        taxonomy_path = taxonomy_file or str(CORVO_PATH / TAXONOMY_FILE)
        ir_path = ir_file or str(CORVO_PATH / IR_FILE)
        
        return categorize_transactions(
            df=df,
            embedder=self._embedder,
            taxonomy_file=taxonomy_path,
            ir_file=ir_path,
            vendor_col=vendor_col,
            description_col=description_col,
            batch_size=batch_size or CATEGORY_BATCH_SIZE,
            top_k=top_k or CATEGORY_TOP_K,
            use_ml=use_ml if use_ml is not None else CATEGORY_USE_ML,
            use_llm=use_llm if use_llm is not None else CATEGORY_USE_LLM,
        )
    
    def categorize_rules_only(
        self,
        df: pd.DataFrame,
        vendor_col: str = "vendor_name",
        description_col: str = "GL_DESCRIPTION",
        taxonomy_file: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run rules-based categorization only (Flag 0 - Fully Confident)."""
        rules_engine = self._get_rules_engine(taxonomy_file)
        
        results = []
        for _, row in df.iterrows():
            record = {
                "vendor_name": str(row.get(vendor_col, "")) if vendor_col in row else "",
                "description": str(row.get(description_col, "")) if description_col in row else "",
                "gl_desc": str(row.get(description_col, "")) if description_col in row else "",
            }
            result = rule_based_classify(record, rules_engine)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        df_result = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Rename columns to match expected format
        if 'L0' in df_result.columns:
            df_result['L0_CATEGORY'] = df_result['L0']
        if 'L1' in df_result.columns:
            df_result['L1_CATEGORY'] = df_result['L1']
        if 'L2' in df_result.columns:
            df_result['L2_CATEGORY'] = df_result['L2']
        
        return df_result
    
    def categorize_ml(
        self,
        df: pd.DataFrame,
        vendor_col: str = "PREDICTED_NORMALIZED_VENDOR_NAME",
        description_col: str = "GL_DESCRIPTION",
        ir_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run ML categorization on records (Flag 1 - Medium Confidence)."""
        ir_path = ir_file or str(CORVO_PATH / IR_FILE)
        
        result_df = ml_categorize(
            records=df,
            embedder=self._embedder,
            ir_file=ir_path,
            vendor_col=vendor_col,
            description_col=description_col,
            batch_size=batch_size or CATEGORY_BATCH_SIZE,
            top_k=top_k or CATEGORY_TOP_K,
        )
        
        return result_df
    
    def categorize_llm(
        self,
        df: pd.DataFrame,
        vendor_col: str = "PREDICTED_NORMALIZED_VENDOR_NAME",
        description_col: str = "GL_DESCRIPTION",
        taxonomy_file: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """Run LLM categorization on records (Flag 2 - Low Confidence)."""
        taxonomy_path = taxonomy_file or str(CORVO_PATH / TAXONOMY_FILE)
        
        result_df = llm_categorize(
            records=df,
            taxonomy_file=taxonomy_path,
            confidence_threshold=confidence_threshold,
            vendor_col=vendor_col,
            description_col=description_col,
        )
        
        return result_df
    
    def categorize_by_flag(
        self,
        df: pd.DataFrame,
        flag: int,
        vendor_col: str = "vendor_name",
        description_col: str = "GL_DESCRIPTION",
        taxonomy_file: Optional[str] = None,
        ir_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
        llm_confidence_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        Categorize based on flag value:
        - flag 0: Rules-only (fully confident, CAT_STATUS=0)
        - flag 1: ML categorization (medium confidence, CAT_STATUS=1)
        - flag 2: LLM categorization (low confidence, CAT_STATUS=2)
        - flag -1: Full cascading pipeline (all stages)
        """
        if flag == 0:
            return self.categorize_rules_only(
                df=df,
                vendor_col=vendor_col,
                description_col=description_col,
                taxonomy_file=taxonomy_file,
            )
        elif flag == 1:
            return self.categorize_ml(
                df=df,
                vendor_col=vendor_col,
                description_col=description_col,
                ir_file=ir_file,
                batch_size=batch_size,
                top_k=top_k,
            )
        elif flag == 2:
            return self.categorize_llm(
                df=df,
                vendor_col=vendor_col,
                description_col=description_col,
                taxonomy_file=taxonomy_file,
                confidence_threshold=llm_confidence_threshold,
            )
        elif flag == -1:
            return self.categorize_transactions(
                df=df,
                vendor_col=vendor_col,
                description_col=description_col,
                taxonomy_file=taxonomy_file,
                ir_file=ir_file,
                batch_size=batch_size,
                top_k=top_k,
                use_ml=True,
                use_llm=True,
            )
        else:
            raise ValueError(f"Invalid flag: {flag}. Must be 0, 1, 2, or -1")
    
    def normalize_and_categorize(
        self,
        df: pd.DataFrame,
        vendor_column: str,
        flag: int,
        description_col: str = "GL_DESCRIPTION",
        taxonomy_file: Optional[str] = None,
        ir_file: Optional[str] = None,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        norm_top_k: Optional[int] = None,
        min_sim: Optional[float] = None,
        batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
        llm_confidence_threshold: float = 0.7,
    ) -> pd.DataFrame:
        """
        Normalize vendors then categorize with specific flag.
        
        Args:
            flag: 0=Rules only, 1=ML, 2=LLM
        """
        # Step 1: Normalize
        df_normalized = self.normalize_vendors(
            df=df,
            vendor_column=vendor_column,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            top_k=norm_top_k,
            min_sim=min_sim,
        )
        
        # Step 2: Categorize with flag
        df_result = self.categorize_by_flag(
            df=df_normalized,
            flag=flag,
            vendor_col=CATEGORY_VENDOR_COL,  # Use normalized vendor column
            description_col=description_col,
            taxonomy_file=taxonomy_file,
            ir_file=ir_file,
            batch_size=batch_size,
            top_k=top_k,
            llm_confidence_threshold=llm_confidence_threshold,
        )
        
        return df_result
    
    def get_taxonomy_schema(self, taxonomy_file: Optional[str] = None) -> Dict[str, Any]:
        """Get taxonomy schema/structure."""
        taxonomy_path = taxonomy_file or str(CORVO_PATH / TAXONOMY_FILE)
        rules_engine = self._get_rules_engine(taxonomy_path)
        
        schema = {
            "L0_categories": {},
            "total_categories": 0,
            "total_rules": len(rules_engine.rules),
        }
        
        for rule in rules_engine.rules:
            l0 = rule.l0 or "Uncategorized"
            l1 = rule.l1 or "Uncategorized"
            l2 = rule.l2 or "Uncategorized"
            
            if l0 not in schema["L0_categories"]:
                schema["L0_categories"][l0] = {
                    "L1_categories": {},
                    "count": 0
                }
            
            if l1 not in schema["L0_categories"][l0]["L1_categories"]:
                schema["L0_categories"][l0]["L1_categories"][l1] = {
                    "L2_categories": [],
                    "count": 0
                }
            
            if l2 not in schema["L0_categories"][l0]["L1_categories"][l1]["L2_categories"]:
                schema["L0_categories"][l0]["L1_categories"][l1]["L2_categories"].append(l2)
                schema["L0_categories"][l0]["L1_categories"][l1]["count"] += 1
                schema["L0_categories"][l0]["count"] += 1
                schema["total_categories"] += 1
        
        return schema


# Global service instance
corvo_service = CorvoService()


