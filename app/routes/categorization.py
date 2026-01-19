from flask import Blueprint, request, jsonify
from app.models.schemas import (
    CategorizeByFlagRequest,
    CategorizeRequest,
    ProcessRequest,
    NormalizeAndCategorizeRequest,
    ProcessResponse,
    TaxonomySchemaResponse
)
from app.services.corvo_service import corvo_service
import pandas as pd
import time

categorization_bp = Blueprint('categorization', __name__)


@categorization_bp.route('/api/v1/categorize', methods=['POST'])
def categorize():
    """Categorize transactions using full cascading pipeline."""
    try:
        data = request.get_json()
        req = CategorizeRequest(**data)
        
        # Convert transactions to DataFrame and map field names
        records = []
        for t in req.transactions:
            record = t.dict()
            # Map gl_description to GL_DESCRIPTION if needed
            if 'gl_description' in record and record['gl_description']:
                record['GL_DESCRIPTION'] = record['gl_description']
            elif 'description' in record and record['description'] and 'GL_DESCRIPTION' not in record:
                record['GL_DESCRIPTION'] = record['description']
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Use GL_DESCRIPTION if description_column is not explicitly set or is default
        desc_col = req.description_column
        if desc_col == "GL_DESCRIPTION" and 'GL_DESCRIPTION' not in df.columns:
            # Try to map from gl_description or description
            if 'gl_description' in df.columns:
                df['GL_DESCRIPTION'] = df['gl_description']
            elif 'description' in df.columns:
                df['GL_DESCRIPTION'] = df['description']
        
        start_time = time.time()
        result_df = corvo_service.categorize_transactions(
            df=df,
            vendor_col=req.vendor_column,
            description_col=desc_col,
            taxonomy_file=req.taxonomy_file,
            ir_file=req.ir_file,
            batch_size=req.batch_size,
            top_k=req.top_k,
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
            message="Categorization complete"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/categorize/by-flag', methods=['POST'])
def categorize_by_flag():
    """Categorize transactions by flag (0=Rules, 1=ML, 2=LLM, -1=Full)."""
    try:
        data = request.get_json()
        req = CategorizeByFlagRequest(**data)
        
        df = pd.DataFrame([t.dict() for t in req.transactions])
        
        start_time = time.time()
        result_df = corvo_service.categorize_by_flag(
            df=df,
            flag=req.flag,
            vendor_col=req.vendor_column,
            description_col=req.description_column,
            taxonomy_file=req.taxonomy_file,
            ir_file=req.ir_file,
            batch_size=req.batch_size,
            top_k=req.top_k,
            llm_confidence_threshold=req.llm_confidence_threshold,
        )
        processing_time = time.time() - start_time
        
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
            flag=req.flag,
            cat_status_distribution=cat_status_dist
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/categorize/rules-only', methods=['POST'])
def categorize_rules_only():
    """Rules-only categorization (Flag 0 - Fully Confident)."""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        vendor_column = data.get('vendor_column', 'vendor_name')
        description_column = data.get('description_column', 'GL_DESCRIPTION')
        taxonomy_file = data.get('taxonomy_file', None)
        
        df = pd.DataFrame(transactions)
        
        start_time = time.time()
        result_df = corvo_service.categorize_rules_only(
            df=df,
            vendor_col=vendor_column,
            description_col=description_column,
            taxonomy_file=taxonomy_file,
        )
        processing_time = time.time() - start_time
        
        # Filter to only CAT_STATUS=0 (fully confident)
        if 'CAT_STATUS' in result_df.columns:
            flag_0_df = result_df[result_df['CAT_STATUS'] == 0].copy()
        else:
            flag_0_df = result_df
        
        results = flag_0_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            flag=0,
            message="Rules-only categorization (Flag 0 - Fully Confident)"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/categorize/ml', methods=['POST'])
def categorize_ml():
    """ML categorization (Flag 1 - Medium Confidence)."""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        vendor_column = data.get('vendor_column', 'PREDICTED_NORMALIZED_VENDOR_NAME')
        description_column = data.get('description_column', 'GL_DESCRIPTION')
        ir_file = data.get('ir_file', None)
        batch_size = data.get('batch_size', None)
        top_k = data.get('top_k', None)
        
        df = pd.DataFrame(transactions)
        
        start_time = time.time()
        result_df = corvo_service.categorize_ml(
            df=df,
            vendor_col=vendor_column,
            description_col=description_column,
            ir_file=ir_file,
            batch_size=batch_size,
            top_k=top_k,
        )
        processing_time = time.time() - start_time
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            flag=1,
            message="ML categorization (Flag 1 - Medium Confidence)"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/categorize/llm', methods=['POST'])
def categorize_llm():
    """LLM categorization (Flag 2 - Low Confidence)."""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        vendor_column = data.get('vendor_column', 'PREDICTED_NORMALIZED_VENDOR_NAME')
        description_column = data.get('description_column', 'GL_DESCRIPTION')
        taxonomy_file = data.get('taxonomy_file', None)
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        df = pd.DataFrame(transactions)
        
        start_time = time.time()
        result_df = corvo_service.categorize_llm(
            df=df,
            vendor_col=vendor_column,
            description_col=description_column,
            taxonomy_file=taxonomy_file,
            confidence_threshold=confidence_threshold,
        )
        processing_time = time.time() - start_time
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            flag=2,
            message="LLM categorization (Flag 2 - Low Confidence)"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/categorize/full', methods=['POST'])
def categorize_full():
    """Full cascading pipeline (Flag -1)."""
    try:
        data = request.get_json()
        req = CategorizeByFlagRequest(**data)
        req.flag = -1
        
        df = pd.DataFrame([t.dict() for t in req.transactions])
        
        start_time = time.time()
        result_df = corvo_service.categorize_by_flag(
            df=df,
            flag=-1,
            vendor_col=req.vendor_column,
            description_col=req.description_column,
            taxonomy_file=req.taxonomy_file,
            ir_file=req.ir_file,
            batch_size=req.batch_size,
            top_k=req.top_k,
            llm_confidence_threshold=req.llm_confidence_threshold,
        )
        processing_time = time.time() - start_time
        
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
            flag=-1,
            cat_status_distribution=cat_status_dist,
            message="Full cascading pipeline (Rules → ML → LLM)"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/normalize-and-categorize/rules-only', methods=['POST'])
def normalize_and_categorize_rules_only():
    """Normalize + Rules-only categorization (Flag 0)."""
    try:
        data = request.get_json()
        req = NormalizeAndCategorizeRequest(**data)
        req.flag = 0
        
        df = pd.DataFrame([t.dict() for t in req.transactions])
        
        start_time = time.time()
        result_df = corvo_service.normalize_and_categorize(
            df=df,
            vendor_column=req.vendor_column,
            flag=0,
            description_col=req.description_column,
            taxonomy_file=req.taxonomy_file,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            norm_top_k=req.norm_top_k,
            min_sim=req.min_sim,
        )
        processing_time = time.time() - start_time
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            flag=0,
            message="Normalization + Rules-only categorization"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/normalize-and-categorize/ml', methods=['POST'])
def normalize_and_categorize_ml():
    """Normalize + ML categorization (Flag 1)."""
    try:
        data = request.get_json()
        req = NormalizeAndCategorizeRequest(**data)
        req.flag = 1
        
        df = pd.DataFrame([t.dict() for t in req.transactions])
        
        start_time = time.time()
        result_df = corvo_service.normalize_and_categorize(
            df=df,
            vendor_column=req.vendor_column,
            flag=1,
            description_col=req.description_column,
            taxonomy_file=req.taxonomy_file,
            ir_file=req.ir_file,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            norm_top_k=req.norm_top_k,
            min_sim=req.min_sim,
            batch_size=req.batch_size,
            top_k=req.top_k,
        )
        processing_time = time.time() - start_time
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            flag=1,
            message="Normalization + ML categorization"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/normalize-and-categorize/llm', methods=['POST'])
def normalize_and_categorize_llm():
    """Normalize + LLM categorization (Flag 2)."""
    try:
        data = request.get_json()
        req = NormalizeAndCategorizeRequest(**data)
        req.flag = 2
        
        df = pd.DataFrame([t.dict() for t in req.transactions])
        
        start_time = time.time()
        result_df = corvo_service.normalize_and_categorize(
            df=df,
            vendor_column=req.vendor_column,
            flag=2,
            description_col=req.description_column,
            taxonomy_file=req.taxonomy_file,
            min_cluster_size=req.min_cluster_size,
            min_samples=req.min_samples,
            norm_top_k=req.norm_top_k,
            min_sim=req.min_sim,
            llm_confidence_threshold=req.llm_confidence_threshold,
        )
        processing_time = time.time() - start_time
        
        results = result_df.to_dict(orient='records')
        
        return jsonify(ProcessResponse(
            success=True,
            records_processed=len(results),
            results=results,
            processing_time_seconds=processing_time,
            flag=2,
            message="Normalization + LLM categorization"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'records_processed': 0,
            'results': []
        }), 500


@categorization_bp.route('/api/v1/taxonomy/schema', methods=['GET'])
def get_taxonomy_schema():
    """Get taxonomy schema/structure."""
    try:
        taxonomy_file = request.args.get('taxonomy_file', None)
        
        schema = corvo_service.get_taxonomy_schema(taxonomy_file=taxonomy_file)
        
        return jsonify(TaxonomySchemaResponse(
            success=True,
            schema=schema['L0_categories'],
            total_categories=schema['total_categories'],
            total_rules=schema['total_rules'],
            message="Taxonomy schema retrieved successfully"
        ).dict())
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'schema': {},
            'total_categories': 0,
            'total_rules': 0
        }), 500

