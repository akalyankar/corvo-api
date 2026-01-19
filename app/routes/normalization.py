from flask import Blueprint, request, jsonify
from app.models.schemas import NormalizeRequest, ProcessRequest, ProcessResponse
from app.services.corvo_service import corvo_service
import pandas as pd
import time

normalization_bp = Blueprint('normalization', __name__)


@normalization_bp.route('/api/v1/process', methods=['POST'])
def process():
    """Full pipeline: Normalize + Categorize transactions."""
    try:
        data = request.get_json()
        req = ProcessRequest(**data)
        
        # Convert transactions to DataFrame and map field names
        records = []
        for t in req.transactions:
            record = t.dict()
            # Map gl_description to GL_DESCRIPTION for corvo system compatibility
            if 'gl_description' in record and record['gl_description']:
                record['GL_DESCRIPTION'] = record['gl_description']
            elif 'description' in record and record['description']:
                record['GL_DESCRIPTION'] = record['description']
            records.append(record)
        
        df = pd.DataFrame(records)
        
        start_time = time.time()
        
        # Step 1: Normalize vendors
        df_normalized = corvo_service.normalize_vendors(
            df=df,
            vendor_column=req.vendor_column,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            top_k=req.top_k,
            min_sim=req.min_sim,
        )
        
        # Step 2: Categorize transactions
        # Ensure GL_DESCRIPTION exists for categorization
        if 'GL_DESCRIPTION' not in df_normalized.columns:
            if 'gl_description' in df_normalized.columns:
                df_normalized['GL_DESCRIPTION'] = df_normalized['gl_description']
            elif 'description' in df_normalized.columns:
                df_normalized['GL_DESCRIPTION'] = df_normalized['description']
            else:
                df_normalized['GL_DESCRIPTION'] = ''  # Empty string if not provided
        
        result_df = corvo_service.categorize_transactions(
            df=df_normalized,
            vendor_col="PREDICTED_NORMALIZED_VENDOR_NAME",  # Use normalized vendor column
            description_col="GL_DESCRIPTION",
            batch_size=req.batch_size,
            top_k=req.category_top_k,
            use_ml=req.use_ml,
            use_llm=req.use_llm,
        )
        
        processing_time = time.time() - start_time
        
        # Get CAT_STATUS distribution
        cat_status_dist = {}
        if 'CAT_STATUS' in result_df.columns:
            status_counts = result_df['CAT_STATUS'].value_counts().to_dict()
            cat_status_dist = {
                'status_0_high_confidence': int(status_counts.get(0, 0)),
                'status_1_medium_confidence': int(status_counts.get(1, 0)),
                'status_2_low_confidence': int(status_counts.get(2, 0)),
            }
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            cat_status_distribution=cat_status_dist,
            message="Full pipeline (Normalize + Categorize) complete"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


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

