from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TransactionRecord(BaseModel):
    """Single transaction record"""
    vendor_name: str = Field(..., description="Vendor name")
    description: Optional[str] = Field(None, description="Transaction description")
    amount: Optional[float] = Field(None, description="Transaction amount")
    date: Optional[str] = Field(None, description="Transaction date")
    gl_code: Optional[str] = Field(None, description="GL code")
    gl_description: Optional[str] = Field(None, description="GL description")


class NormalizeRequest(BaseModel):
    """Request for vendor normalization"""
    transactions: List[TransactionRecord]
    vendor_column: str = Field("vendor_name", description="Column name for vendor")
    min_cluster_size: Optional[int] = Field(None, description="HDBSCAN min cluster size")
    min_samples: Optional[int] = Field(None, description="HDBSCAN min samples")
    top_k: Optional[int] = Field(None, description="Vector retrieval top-K")
    min_sim: Optional[float] = Field(None, description="Minimum similarity threshold")


class CategorizeRequest(BaseModel):
    """Request for transaction categorization"""
    transactions: List[TransactionRecord]
    vendor_column: str = Field("PREDICTED_NORMALIZED_VENDOR_NAME", description="Vendor column name")
    description_column: str = Field("GL_DESCRIPTION", description="Description column name")
    taxonomy_file: Optional[str] = Field(None, description="Path to taxonomy file")
    ir_file: Optional[str] = Field(None, description="Path to IR pack file")
    batch_size: Optional[int] = Field(None, description="Batch size for ML")
    top_k: Optional[int] = Field(None, description="Top-K predictions")
    use_ml: bool = Field(True, description="Use ML categorization")
    use_llm: bool = Field(False, description="Use LLM categorization")


class ProcessRequest(BaseModel):
    """Request for full pipeline (normalize + categorize)"""
    transactions: List[TransactionRecord]
    vendor_column: str = Field("vendor_name", description="Vendor column name")
    # Normalization params
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    top_k: Optional[int] = None
    min_sim: Optional[float] = None
    # Categorization params
    batch_size: Optional[int] = None
    category_top_k: Optional[int] = None
    use_ml: bool = True
    use_llm: bool = False


class CategorizeByFlagRequest(BaseModel):
    """Request for categorization by flag"""
    transactions: List[TransactionRecord]
    flag: int = Field(..., description="Flag: 0=Rules only, 1=ML, 2=LLM, -1=Full pipeline")
    vendor_column: str = Field("vendor_name", description="Vendor column name")
    description_column: str = Field("GL_DESCRIPTION", description="Description column name")
    taxonomy_file: Optional[str] = Field(None, description="Path to taxonomy file")
    ir_file: Optional[str] = Field(None, description="Path to IR pack file (for ML)")
    batch_size: Optional[int] = Field(None, description="Batch size for ML")
    top_k: Optional[int] = Field(None, description="Top-K predictions for ML")
    llm_confidence_threshold: float = Field(0.7, description="Confidence threshold for LLM")


class NormalizeAndCategorizeRequest(BaseModel):
    """Request for normalization + categorization with specific flag"""
    transactions: List[TransactionRecord]
    flag: int = Field(..., description="Flag: 0=Rules only, 1=ML, 2=LLM")
    vendor_column: str = Field("vendor_name", description="Vendor column name")
    description_column: str = Field("GL_DESCRIPTION", description="Description column name")
    # Normalization params
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    norm_top_k: Optional[int] = None
    min_sim: Optional[float] = None
    # Categorization params
    taxonomy_file: Optional[str] = None
    ir_file: Optional[str] = None
    batch_size: Optional[int] = None
    top_k: Optional[int] = None
    llm_confidence_threshold: float = 0.7


class ProcessResponse(BaseModel):
    """Response with processed transactions"""
    success: bool
    records_processed: int
    results: List[Dict[str, Any]]
    message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    flag: Optional[int] = None
    cat_status_distribution: Optional[Dict[str, int]] = None


class TaxonomySchemaResponse(BaseModel):
    """Response with taxonomy schema"""
    success: bool
    schema: Dict[str, Any]
    total_categories: int
    total_rules: int
    message: Optional[str] = None

