"""
DVB-T Visualization Web Application

Flask application providing a web interface for visualizing
the DVB-T encoding pipeline.
"""

import os
import sys
import json
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from flask import (
    Flask, 
    render_template, 
    request, 
    jsonify, 
    send_file,
    session
)
from werkzeug.utils import secure_filename
import numpy as np

# Add parent directory to path for dvb imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Handle both direct script execution and module import
try:
    from .pipeline import DVBPipelineViz, DVBTParams, PipelineResults
    from .media_converter import convert_to_ts, check_ffmpeg, generate_test_ts
    from .visualizations import get_visualization, get_all_visualizations, VISUALIZERS
except ImportError:
    from pipeline import DVBPipelineViz, DVBTParams, PipelineResults
    from media_converter import convert_to_ts, check_ffmpeg, generate_test_ts
    from visualizations import get_visualization, get_all_visualizations, VISUALIZERS


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dvb-t-viz-secret-key-change-in-prod')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'},
    'video': {'mp4', 'avi', 'mkv', 'mov', 'webm', 'flv', 'm4v'},
}

# In-memory storage for job results (use Redis/database in production)
jobs: Dict[str, Dict[str, Any]] = {}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS['image'] or ext in ALLOWED_EXTENSIONS['video']


def serialize_results(results: PipelineResults) -> dict:
    """Convert PipelineResults to JSON-serializable dict."""
    return {
        'input_packets': results.input_packets,
        'pid_distribution': results.pid_distribution,
        'sample_rate': results.sample_rate,
        'data_rate': results.data_rate,
        'duration': results.duration,
        'params': results.params.to_dict(),
        'stages': list(VISUALIZERS.keys()),
    }


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/params', methods=['GET'])
def get_params():
    """Get available DVB-T parameter options."""
    return jsonify({
        'modes': ['2K', '8K'],
        'constellations': ['QPSK', '16QAM', '64QAM'],
        'code_rates': ['1/2', '2/3', '3/4', '5/6', '7/8'],
        'guard_intervals': ['1/4', '1/8', '1/16', '1/32'],
        'bandwidths': ['6MHz', '7MHz', '8MHz'],
        'defaults': {
            'mode': '2K',
            'constellation': 'QPSK',
            'code_rate': '1/2',
            'guard_interval': '1/4',
            'bandwidth': '8MHz',
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload():
    """
    Upload media file and process through DVB-T pipeline.
    
    Accepts multipart form with:
    - file: Image or video file
    - mode, constellation, code_rate, guard_interval, bandwidth: DVB-T params
    
    Returns job_id for retrieving results.
    """
    # Check for file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get DVB-T parameters from form
    params = DVBTParams(
        mode=request.form.get('mode', '2K'),
        constellation=request.form.get('constellation', 'QPSK'),
        code_rate=request.form.get('code_rate', '1/2'),
        guard_interval=request.form.get('guard_interval', '1/4'),
        bandwidth=request.form.get('bandwidth', '8MHz'),
    )
    
    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'created': datetime.now().isoformat(),
        'filename': secure_filename(file.filename),
        'params': params.to_dict(),
    }
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)
        
        try:
            # Convert to Transport Stream
            ts_data, media_info = convert_to_ts(tmp_path, max_duration=5.0)
            
            # Process through DVB-T pipeline
            pipeline = DVBPipelineViz(params)
            results = pipeline.process(ts_data)
            
            # Store results
            jobs[job_id]['status'] = 'complete'
            jobs[job_id]['media_info'] = media_info
            jobs[job_id]['results'] = results
            jobs[job_id]['summary'] = serialize_results(results)
            
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
        
        return jsonify({
            'job_id': job_id,
            'status': 'complete',
            'summary': jobs[job_id]['summary'],
        })
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo', methods=['POST'])
def demo():
    """
    Generate demo data without file upload.
    
    Creates synthetic Transport Stream for demonstration.
    """
    # Get DVB-T parameters
    data = request.get_json() or {}
    params = DVBTParams(
        mode=data.get('mode', '2K'),
        constellation=data.get('constellation', 'QPSK'),
        code_rate=data.get('code_rate', '1/2'),
        guard_interval=data.get('guard_interval', '1/4'),
        bandwidth=data.get('bandwidth', '8MHz'),
    )
    
    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'created': datetime.now().isoformat(),
        'filename': 'demo_stream.ts',
        'params': params.to_dict(),
    }
    
    try:
        # Generate test Transport Stream (1000 packets = ~188KB)
        ts_data, media_info = generate_test_ts(num_packets=1000)
        
        # Process through DVB-T pipeline
        pipeline = DVBPipelineViz(params)
        results = pipeline.process(ts_data)
        
        # Store results
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['media_info'] = media_info
        jobs[job_id]['results'] = results
        jobs[job_id]['summary'] = serialize_results(results)
        
        return jsonify({
            'job_id': job_id,
            'status': 'complete',
            'summary': jobs[job_id]['summary'],
        })
        
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'filename': job.get('filename'),
        'params': job.get('params'),
    }
    
    if job['status'] == 'complete':
        response['summary'] = job.get('summary')
    elif job['status'] == 'error':
        response['error'] = job.get('error')
    
    return jsonify(response)


@app.route('/api/results/<job_id>', methods=['GET'])
def results(job_id: str):
    """Get job results summary."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Job not complete', 'status': job['status']}), 400
    
    return jsonify({
        'job_id': job_id,
        'filename': job.get('filename'),
        'media_info': job.get('media_info'),
        'summary': job.get('summary'),
    })


@app.route('/api/viz/<job_id>/<stage>', methods=['GET'])
def visualization(job_id: str, stage: str):
    """
    Get Plotly visualization for a specific pipeline stage.
    
    Args:
        job_id: Job identifier
        stage: Pipeline stage (input, scrambler, reed_solomon, etc.)
        
    Returns:
        Plotly figure as JSON
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Job not complete'}), 400
    
    if stage not in VISUALIZERS:
        return jsonify({'error': f'Unknown stage: {stage}'}), 400
    
    try:
        fig = get_visualization(stage, job['results'])
        return jsonify(fig)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reprocess/<job_id>', methods=['POST'])
def reprocess(job_id: str):
    """
    Re-process with different DVB-T parameters.
    
    Uses cached Transport Stream data to avoid re-uploading.
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    old_job = jobs[job_id]
    if 'results' not in old_job:
        return jsonify({'error': 'Original job has no results'}), 400
    
    # Get new parameters
    data = request.get_json() or {}
    params = DVBTParams(
        mode=data.get('mode', old_job['params'].get('mode', '2K')),
        constellation=data.get('constellation', old_job['params'].get('constellation', 'QPSK')),
        code_rate=data.get('code_rate', old_job['params'].get('code_rate', '1/2')),
        guard_interval=data.get('guard_interval', old_job['params'].get('guard_interval', '1/4')),
        bandwidth=data.get('bandwidth', old_job['params'].get('bandwidth', '8MHz')),
    )
    
    # Create new job
    new_job_id = str(uuid.uuid4())
    jobs[new_job_id] = {
        'status': 'processing',
        'created': datetime.now().isoformat(),
        'filename': old_job.get('filename'),
        'params': params.to_dict(),
        'media_info': old_job.get('media_info'),
    }
    
    try:
        # Get original TS data from old results
        ts_data = old_job['results'].input_data
        
        # Process with new parameters
        pipeline = DVBPipelineViz(params)
        results = pipeline.process(ts_data)
        
        # Store results
        jobs[new_job_id]['status'] = 'complete'
        jobs[new_job_id]['results'] = results
        jobs[new_job_id]['summary'] = serialize_results(results)
        
        return jsonify({
            'job_id': new_job_id,
            'status': 'complete',
            'summary': jobs[new_job_id]['summary'],
        })
        
    except Exception as e:
        jobs[new_job_id]['status'] = 'error'
        jobs[new_job_id]['error'] = str(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<job_id>', methods=['GET'])
def download(job_id: str):
    """Download I/Q output file."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Job not complete'}), 400
    
    results = job['results']
    if len(results.iq_samples) == 0:
        return jsonify({'error': 'No I/Q data available'}), 400
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cf32') as tmp:
        results.iq_samples.astype(np.complex64).tofile(tmp.name)
        tmp_path = tmp.name
    
    filename = f"dvbt_output_{job_id[:8]}.cf32"
    
    return send_file(
        tmp_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/octet-stream'
    )


@app.route('/api/check', methods=['GET'])
def check():
    """Check system capabilities."""
    return jsonify({
        'ffmpeg_available': check_ffmpeg(),
        'stages': list(VISUALIZERS.keys()),
    })


def main():
    """Run the DVB-T visualization server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DVB-T Pipeline Visualizer - Web GUI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port to run the server on'
    )
    parser.add_argument(
        '--host', '-H',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (use 0.0.0.0 for all interfaces)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode with auto-reload'
    )
    
    args = parser.parse_args()
    
    print(f"Starting DVB-T Visualizer at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

