import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinn.api import PINNService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pinn_service.log')
    ]
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize PINN Service
pinn_service = PINNService()


@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'PINN Equation Solver API is running',
        'version': '1.0.0'
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'PINN Equation Solver',
        'version': '1.0.0'
    }), 200


@app.route('/api/initialize', methods=['POST'])
def initialize():
    try:
        config = request.json
        if not config:
            logging.warning('No configuration provided')
            return jsonify({
                'status': 'error',
                'message': 'No configuration provided'
            }), 400

        # Validate config
        if not isinstance(config, dict):
            logging.error('Invalid configuration format')
            return jsonify({
                'status': 'error',
                'message': 'Invalid configuration format'
            }), 400

        result = pinn_service.initialize_model(config)
        logging.info(f'Model initialized: {result}')

        return jsonify({
            'status': 'success',
            'data': result
        })
    except json.JSONDecodeError:
        logging.error('Invalid JSON format')
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON format'
        }), 400
    except Exception as e:
        logging.error(f'Initialization error: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        if not data:
            logging.warning('No solving data provided')
            return jsonify({
                'status': 'error',
                'message': 'No solving data provided'
            }), 400

        result = pinn_service.solve_equation(data)
        logging.info('Equation solved successfully')

        return jsonify({
            'status': 'success',
            'data': result
        })
    except json.JSONDecodeError:
        logging.error('Invalid JSON format for solving request')
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON format for solving request'
        }), 400
    except Exception as e:
        logging.error(f'Solving error: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def page_not_found(e):
    logging.warning(f'Page not found: {request.url}')
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(Exception)
def handle_global_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'status': 'error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )
