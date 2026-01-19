from flask import Blueprint, request, jsonify
from app.models.schemas import NormalizeRequest, ProcessRequest, ProcessResponse
from app.services.corvo_service import corvo_service
import pandas as pd
import time

normalization_bp = Blueprint('normalization', __name__)


@normalization_bp.route('/api/v1/normalize', methods=['POST'])
def normalize():
    """Normalize vendor names in transaction records."""
    try:
        data = request.get_json()
        req = NormalizeRequest(**data)
        
        df = pd.DataFrame([t.dict() for t in req.transactions])
        
        start_time = time.time()
        result_df = corvo_service.normalize_vendors(
            df=df,
            vendor_column=req.vendor_column,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            top_k=req.top_k,
            min_sim=req.min_sim,
        )
        processing_time = time.time() - start_time
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            message="Normalization complete"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500

