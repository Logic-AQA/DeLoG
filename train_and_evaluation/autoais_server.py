"""
AutoAIS NLI Model Server
========================
Standalone web service for AutoAIS NLI inference.

Usage:
    AUTOAIS_MODEL_PATH=<local-autoais-model> CUDA_VISIBLE_DEVICES=6 python train_and_evaluation/autoais_server.py --port 8001

API:
    POST /nli
    Body: {"premise": "...", "hypothesis": "..."}
    Returns: {"result": 0 or 1}
"""

import argparse
import gc
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Global model and tokenizer
autoais_model = None
autoais_tokenizer = None
request_count = 0
empty_cache_every = 1

AUTOAIS_MODEL_PATH = os.environ.get("AUTOAIS_MODEL_PATH", "")


def release_cuda_cache_if_needed():
    """Return unused CUDA cache to the driver after requests."""
    global request_count
    request_count += 1
    if empty_cache_every <= 0 or request_count % empty_cache_every != 0:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    """Load AutoAIS model on startup."""
    global autoais_model, autoais_tokenizer
    if not AUTOAIS_MODEL_PATH:
        raise ValueError("Set AUTOAIS_MODEL_PATH to a local AutoAIS model checkpoint")
    print("Loading AutoAIS model...")
    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
        AUTOAIS_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    autoais_tokenizer = AutoTokenizer.from_pretrained(
        AUTOAIS_MODEL_PATH,
        use_fast=False,
        local_files_only=True
    )
    print("AutoAIS model loaded successfully!")


@app.route('/nli', methods=['POST'])
def nli_inference():
    """
    Run NLI inference.

    Request body:
        {
            "premise": "text of the premise",
            "hypothesis": "text of the hypothesis"
        }

    Response:
        {
            "result": 0 or 1  (0 = not entailed, 1 = entailed)
        }
    """
    try:
        data = request.get_json()
        premise = data.get('premise', '')
        hypothesis = data.get('hypothesis', '')

        if not premise or not hypothesis:
            return jsonify({"error": "Both premise and hypothesis are required"}), 400

        # Format input
        input_text = f"premise: {premise} hypothesis: {hypothesis}"
        input_ids = None
        outputs = None
        try:
            input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(
                autoais_model.device
            )

            # Run inference
            with torch.inference_mode():
                outputs = autoais_model.generate(input_ids, max_new_tokens=10)
            result_text = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
        finally:
            del input_ids
            del outputs
            release_cuda_cache_if_needed()

        # Parse result
        result = 1 if result_text == "1" else 0

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        status = {
            "status": "ok",
            "model_loaded": autoais_model is not None,
            "tokenizer_loaded": autoais_tokenizer is not None
        }
        print(f"Health check: {status}")  # Debug log
        return jsonify(status), 200
    except Exception as e:
        print(f"Health check error: {e}")  # Debug log
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8001, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument(
        '--empty-cache-every',
        type=int,
        default=1,
        help='Call torch.cuda.empty_cache() every N NLI requests; <=0 disables it.'
    )
    args = parser.parse_args()
    empty_cache_every = args.empty_cache_every

    # Load model before starting server
    load_model()

    # Start Flask server with threaded=False to avoid threading issues
    print(f"Starting AutoAIS server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
