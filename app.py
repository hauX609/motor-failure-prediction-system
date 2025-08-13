import joblib
import numpy as np
import pandas as pd
import sqlite3
import os
from functools import wraps
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model, Model # pyright: ignore[reportMissingImports]
from datetime import datetime
import shap
import warnings
import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import threading

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

# Configuration
DB_FILE = "motors.db"
SECRET_API_KEY = os.getenv('MOTOR_API_KEY', "8da8847a3160d7e48e66efc235b65ba3ef9688b8325e09b172291fb68040008c")
MAX_BATCH_SIZE = 100
DB_TIMEOUT = 30
REQUIRED_SEQUENCE_LENGTH = 50

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model assets
model = None
feature_scaler = None
rul_scaler = None
feature_cols = None
status_map = {0: 'Optimal', 1: 'Degrading', 2: 'Critical'}
classification_explainer = None
regression_explainer = None

# Thread lock for model operations
model_lock = threading.Lock()

def load_model_assets():
    """Load all necessary model assets with proper error handling."""
    global model, feature_scaler, rul_scaler, feature_cols
    
    required_files = {
        'motor_model_multi.keras': 'model',
        'scaler.pkl': 'feature_scaler', 
        'rul_scaler.pkl': 'rul_scaler',
        'feature_columns.pkl': 'feature_cols'
    }
    
    try:
        # Check if all required files exist
        missing_files = []
        for filename in required_files.keys():
            if not os.path.exists(filename):
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Load assets
        model = load_model('motor_model_multi.keras')
        feature_scaler = joblib.load('scaler.pkl')
        rul_scaler = joblib.load('rul_scaler.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        
        # Validate model outputs
        if len(model.outputs) != 2:
            raise ValueError("Model must have exactly 2 outputs (classification and regression)")
        
        # Validate feature columns
        if not isinstance(feature_cols, list) or len(feature_cols) == 0:
            raise ValueError("feature_cols must be a non-empty list")
        
        logger.info("✅ All model assets loaded successfully.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model assets: {e}")
        return False

def initialize_explainers():
    """Initialize SHAP explainers for both classification and regression outputs."""
    global classification_explainer, regression_explainer, model
    
    if model is None:
        logger.error("Cannot initialize explainers: model not loaded")
        return False
    
    logger.info("Initializing SHAP explainers...")
    try:
        if not os.path.exists('shap_background.pkl'):
            logger.warning("SHAP background data not found. Explainers will not be available.")
            return False
        
        background_data = joblib.load('shap_background.pkl')
        classification_model_for_shap = Model(inputs=model.inputs, outputs=model.outputs[0])
        regression_model_for_shap = Model(inputs=model.inputs, outputs=model.outputs[1])
        classification_explainer = shap.GradientExplainer(classification_model_for_shap, background_data)
        regression_explainer = shap.GradientExplainer(regression_model_for_shap, background_data)
        
        logger.info("✅ SHAP explainers for both outputs initialized successfully.")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize SHAP explainers: {e}")
        return False

# Database connection management
@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling."""
    conn = None
    try:
        if not os.path.exists(DB_FILE):
            raise FileNotFoundError(f"Database file {DB_FILE} not found")
        
        conn = sqlite3.connect(DB_FILE, timeout=DB_TIMEOUT)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        yield conn
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Validation functions
def validate_motor_id(motor_id: str) -> bool:
    """Validate motor ID format with enhanced checks."""
    if not motor_id or not isinstance(motor_id, str):
        return False
    
    motor_id = motor_id.strip()
    if len(motor_id) == 0 or len(motor_id) > 50:  # Reasonable length limit
        return False
    
    # Check for basic SQL injection patterns
    dangerous_chars = ["'", '"', ';', '--', '/*', '*/']
    if any(char in motor_id for char in dangerous_chars):
        return False
    
    return True

def validate_severity(severity: str) -> bool:
    """Validate alert severity - Updated to match database constraints."""
    # These should match your database CHECK constraint exactly
    valid_severities = ['Degrading', 'Critical', 'Warning']  # Removed 'Optimal'
    return severity in valid_severities

def get_alert_severity_for_status(status: str) -> str:
    """Convert model status to valid alert severity for database."""
    # Map model statuses to valid database severities
    status_to_severity_map = {
        'Optimal': 'Warning',      # Optimal status maps to Warning for alerts
        'Degrading': 'Degrading',
        'Critical': 'Critical'
    }
    return status_to_severity_map.get(status, 'Warning')

# Authentication decorator
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != SECRET_API_KEY:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized. Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated_function

# Database query functions
def get_latest_sequence_from_db(motor_id: str) -> pd.DataFrame:
    """Get latest sequence from database with validation."""
    if not validate_motor_id(motor_id):
        raise ValueError("Invalid motor ID")
    
    with get_db_connection() as conn:
        # Use parameterized query to prevent SQL injection
        feature_cols_str = ', '.join(f'"{col}"' for col in feature_cols)  # Quote column names
        query = f"""
            SELECT {feature_cols_str} 
            FROM sensor_readings 
            WHERE motor_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(motor_id, REQUIRED_SEQUENCE_LENGTH))
        
        if df.empty:
            raise ValueError(f"No data found for motor {motor_id}")
        
        # Reverse to get chronological order
        return df.reindex(index=df.index[::-1]).reset_index(drop=True)

def get_multiple_sequences_from_db(motor_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """Get latest sequences for multiple motors with validation."""
    if not motor_ids or not all(validate_motor_id(mid) for mid in motor_ids):
        raise ValueError("Invalid motor IDs")
    
    with get_db_connection() as conn:
        placeholders = ','.join(['?' for _ in motor_ids])
        feature_cols_str = ', '.join(f'"{col}"' for col in feature_cols)
        
        query = f"""
        WITH ranked_readings AS (
            SELECT motor_id, {feature_cols_str}, timestamp,
                   ROW_NUMBER() OVER (PARTITION BY motor_id ORDER BY timestamp DESC) as rn
            FROM sensor_readings
            WHERE motor_id IN ({placeholders})
        )
        SELECT motor_id, {feature_cols_str}
        FROM ranked_readings
        WHERE rn <= ?
        ORDER BY motor_id, timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=motor_ids + [REQUIRED_SEQUENCE_LENGTH])
    
    # Group by motor_id and return dictionary of DataFrames
    motor_sequences = {}
    for motor_id, group in df.groupby('motor_id'):
        motor_sequences[motor_id] = group[feature_cols].reset_index(drop=True)
    
    return motor_sequences

def get_all_active_motors() -> List[str]:
    """Get list of all motors from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Try motors table first
            cursor.execute("SELECT DISTINCT motor_id FROM motors WHERE motor_id IS NOT NULL")
            motor_ids = [row[0] for row in cursor.fetchall()]
            
            # If no motors found, try sensor_readings
            if not motor_ids:
                cursor.execute("SELECT DISTINCT motor_id FROM sensor_readings WHERE motor_id IS NOT NULL")
                motor_ids = [row[0] for row in cursor.fetchall()]
                
        except sqlite3.OperationalError:
            # If motors table doesn't exist, use sensor_readings
            cursor.execute("SELECT DISTINCT motor_id FROM sensor_readings WHERE motor_id IS NOT NULL")
            motor_ids = [row[0] for row in cursor.fetchall()]
    
    return [mid for mid in motor_ids if validate_motor_id(mid)]

# Motor status functions
def get_motor_status(motor_id: str) -> str:
    """Get current status of a motor."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT latest_status FROM motors WHERE motor_id = ?", (motor_id,))
        result = cursor.fetchone()
        return result[0] if result and result[0] else 'Optimal'

def get_multiple_motor_statuses(motor_ids: List[str]) -> Dict[str, str]:
    """Get current status of multiple motors efficiently."""
    if not motor_ids:
        return {}
    
    with get_db_connection() as conn:
        placeholders = ','.join(['?' for _ in motor_ids])
        query = f"SELECT motor_id, latest_status FROM motors WHERE motor_id IN ({placeholders})"
        cursor = conn.cursor()
        cursor.execute(query, motor_ids)
        results = cursor.fetchall()
    
    # Return as dictionary with default 'Optimal' for missing motors
    status_dict = {motor_id: 'Optimal' for motor_id in motor_ids}
    for motor_id, status in results:
        if status:  # Don't validate here since we're reading from DB
            status_dict[motor_id] = status
    
    return status_dict

def update_motor_status(motor_id: str, new_status: str) -> bool:
    """Update status of a motor."""
    # Allow all statuses for motor status updates (Optimal, Degrading, Critical)
    if new_status not in ['Optimal', 'Degrading', 'Critical']:
        raise ValueError(f"Invalid motor status: {new_status}")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE motors SET latest_status = ? WHERE motor_id = ?", (new_status, motor_id))
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Error updating motor status for {motor_id}: {e}")
        return False

def batch_update_motor_statuses(motor_status_updates: List[Tuple[str, str]]) -> bool:
    """Batch update motor statuses efficiently."""
    if not motor_status_updates:
        return True
    
    # Validate all updates
    for motor_id, status in motor_status_updates:
        if not validate_motor_id(motor_id) or status not in ['Optimal', 'Degrading', 'Critical']:
            raise ValueError(f"Invalid motor update: {motor_id}, {status}")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE motors SET latest_status = ? WHERE motor_id = ?",
                [(status, motor_id) for motor_id, status in motor_status_updates]
            )
            conn.commit()
            return True
    except sqlite3.Error as e:
        logger.error(f"Error batch updating motor statuses: {e}")
        return False


# Alert functions
def create_alert(motor_id: str, severity: str, message: str) -> bool:
    """Create a new alert - FIXED to use valid severities."""
    if not validate_motor_id(motor_id):
        raise ValueError("Invalid motor ID")
    
    # Convert severity to valid database severity
    db_severity = get_alert_severity_for_status(severity)
    if not validate_severity(db_severity):
        raise ValueError(f"Invalid severity: {severity}")
    
    if not message or len(message.strip()) == 0:
        raise ValueError("Alert message cannot be empty")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO alerts (motor_id, timestamp, severity, message) VALUES (?, ?, ?, ?)",
                (motor_id, timestamp, db_severity, message.strip())
            )
            conn.commit()
        
        logger.info(f"🚨 Alert created for {motor_id}: {message}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error creating alert for {motor_id}: {e}")
        return False

def batch_create_alerts(alerts_data: List[Dict]) -> bool:
    """Batch create alerts efficiently - FIXED to use valid severities."""
    if not alerts_data:
        return True
    
    # Validate and convert all alerts
    processed_alerts = []
    for alert in alerts_data:
        if not all(key in alert for key in ['motor_id', 'severity', 'message']):
            raise ValueError("Invalid alert data structure")
        
        if not validate_motor_id(alert['motor_id']):
            raise ValueError(f"Invalid motor ID: {alert['motor_id']}")
        
        # Convert severity to valid database severity
        db_severity = get_alert_severity_for_status(alert['severity'])
        if not validate_severity(db_severity):
            raise ValueError(f"Invalid alert severity: {alert['severity']}")
        
        processed_alerts.append({
            'motor_id': alert['motor_id'],
            'severity': db_severity,
            'message': alert['message']
        })
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            alerts_to_insert = [
                (alert['motor_id'], timestamp, alert['severity'], alert['message'].strip())
                for alert in processed_alerts
            ]
            
            cursor.executemany(
                "INSERT INTO alerts (motor_id, timestamp, severity, message) VALUES (?, ?, ?, ?)",
                alerts_to_insert
            )
            conn.commit()
        
        for alert in processed_alerts:
            logger.info(f"🚨 Alert created for {alert['motor_id']}: {alert['message']}")
        
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error batch creating alerts: {e}")
        return False

# Prediction functions
def predict_single_motor(motor_id: str, sequence_df: pd.DataFrame, previous_status: str = None) -> Dict:
    """Predict for a single motor with comprehensive error handling."""
    try:
        if len(sequence_df) < REQUIRED_SEQUENCE_LENGTH:
            return {
                'motor_id': motor_id,
                'error': f'Not enough data to make a prediction. Need {REQUIRED_SEQUENCE_LENGTH} readings, got {len(sequence_df)}.',
                'success': False
            }
        
        # Validate sequence data
        if sequence_df.isnull().any().any():
            logger.warning(f"Motor {motor_id} has null values in sequence data")
            sequence_df = sequence_df.fillna(sequence_df.mean())  # Fill with column means
        
        with model_lock:  # Thread safety for model predictions
            scaled_sequence = feature_scaler.transform(sequence_df)
            reshaped_sequence = np.expand_dims(scaled_sequence, axis=0)
            
            # Make prediction with error handling
            predictions = model.predict(reshaped_sequence, verbose=0)
            
            if len(predictions) != 2:
                raise ValueError("Model should return both classification and regression predictions")
        
        class_prediction, reg_prediction = predictions
        predicted_class_index = np.argmax(class_prediction, axis=1)[0]
        predicted_status = status_map.get(predicted_class_index, "Unknown")
        
        # Validate RUL prediction
        scaled_rul_prediction = reg_prediction[0]
        predicted_rul = rul_scaler.inverse_transform([scaled_rul_prediction])[0][0]
        
        # Ensure RUL is non-negative
        predicted_rul = max(0, float(predicted_rul))
        
        # Determine if alert should be created - FIXED logic
        alert_needed = False
        if previous_status and predicted_status != 'Optimal':  # Only create alerts for non-optimal statuses
            status_priority = {'Optimal': 0, 'Degrading': 1, 'Critical': 2}
            if status_priority.get(predicted_status, -1) > status_priority.get(previous_status, -1):
                alert_needed = True
        
        return {
            'motor_id': motor_id,
            'predicted_status': predicted_status,
            'predicted_rul': round(predicted_rul, 2),
            'probabilities': [float(p) for p in class_prediction[0]],
            'previous_status': previous_status,
            'alert_needed': alert_needed,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Prediction error for motor {motor_id}: {e}")
        return {
            'motor_id': motor_id,
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check database connectivity
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            db_healthy = True
    except Exception:
        db_healthy = False
    
    return jsonify({
        'status': 'healthy' if (model is not None and db_healthy) else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'explainers_loaded': classification_explainer is not None and regression_explainer is not None,
        'database_accessible': db_healthy,
        'feature_count': len(feature_cols) if feature_cols else 0
    })

@app.route('/predict/<string:motor_id>', methods=['GET'])
@api_key_required
def predict(motor_id):
    """Predict motor status and RUL with comprehensive validation."""
    try:
        if not validate_motor_id(motor_id):
            return jsonify({'error': 'Invalid motor ID format'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        sequence_df = get_latest_sequence_from_db(motor_id)
        previous_status = get_motor_status(motor_id)
        result = predict_single_motor(motor_id, sequence_df, previous_status)
        
        if not result['success']:
            return jsonify({'error': result['error']}), 404
        
        # Create alert if needed and status is not optimal
        if result['alert_needed'] and result['predicted_status'] != 'Optimal':
            message = f"Status changed from {result['previous_status']} to {result['predicted_status']}. Predicted RUL: {result['predicted_rul']:.0f}"
            create_alert(motor_id, result['predicted_status'], message)
        
        # Update motor status
        update_motor_status(motor_id, result['predicted_status'])
        
        return jsonify({
            'motor_id': result['motor_id'],
            'predicted_status': result['predicted_status'],
            'predicted_rul': result['predicted_rul'],
            'probabilities': result['probabilities'],
            'timestamp': datetime.now().isoformat()
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error for motor {motor_id}: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
@api_key_required
def batch_predict():
    """Batch predict motor status and RUL for multiple motors."""
    start_time = time.time()
    try:
        # Get motor IDs from request body
        data = request.get_json()
        if not data or 'motor_ids' not in data:
            return jsonify({'error': 'Request must contain motor_ids array'}), 400
        
        motor_ids = data['motor_ids']
        max_motors = data.get('max_motors', MAX_BATCH_SIZE)
        
        if not isinstance(motor_ids, list) or len(motor_ids) == 0:
            return jsonify({'error': 'motor_ids must be a non-empty array'}), 400
        
        if len(motor_ids) > max_motors:
            return jsonify({'error': f'Too many motors requested. Maximum: {max_motors}'}), 400
        
        # Validate all motor IDs
        if not all(validate_motor_id(mid) for mid in motor_ids):
            return jsonify({'error': 'One or more invalid motor IDs'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get data for all motors efficiently
        motor_sequences = get_multiple_sequences_from_db(motor_ids)
        previous_statuses = get_multiple_motor_statuses(motor_ids)
        
        # Filter out motors without sufficient data
        valid_motors = {mid: seq for mid, seq in motor_sequences.items() if len(seq) >= REQUIRED_SEQUENCE_LENGTH}
        invalid_motors = [mid for mid in motor_ids if mid not in valid_motors]
        
        if not valid_motors:
            return jsonify({
                'error': 'No motors have sufficient data for prediction',
                'invalid_motors': invalid_motors
            }), 404
        
        # Prepare batch prediction
        motor_list = list(valid_motors.keys())
        
        # Handle null values in batch
        processed_sequences = []
        for motor_id in motor_list:
            seq_df = valid_motors[motor_id]
            if seq_df.isnull().any().any():
                logger.warning(f"Motor {motor_id} has null values, filling with column means")
                seq_df = seq_df.fillna(seq_df.mean())
            processed_sequences.append(feature_scaler.transform(seq_df))
        
        sequences_array = np.array(processed_sequences)
        
        # Batch prediction with thread safety
        with model_lock:
            predictions = model.predict(sequences_array, verbose=0)
            if len(predictions) != 2:
                raise ValueError("Model should return both classification and regression predictions")
        
        class_predictions, reg_predictions = predictions
        
        # Process results
        results = []
        alerts_to_create = []
        status_updates = []
        
        for i, motor_id in enumerate(motor_list):
            try:
                predicted_class_index = np.argmax(class_predictions[i])
                predicted_status = status_map.get(predicted_class_index, "Unknown")
                predicted_rul = max(0, float(rul_scaler.inverse_transform([reg_predictions[i]])[0][0]))
                previous_status = previous_statuses.get(motor_id, 'Optimal')
                
                # Check if alert is needed - FIXED logic
                status_priority = {'Optimal': 0, 'Degrading': 1, 'Critical': 2}
                if (predicted_status != 'Optimal' and 
                    status_priority.get(predicted_status, -1) > status_priority.get(previous_status, -1)):
                    alerts_to_create.append({
                        'motor_id': motor_id,
                        'severity': predicted_status,
                        'message': f"Status changed from {previous_status} to {predicted_status}. Predicted RUL: {predicted_rul:.0f}"
                    })
                
                status_updates.append((motor_id, predicted_status))
                
                results.append({
                    'motor_id': motor_id,
                    'predicted_status': predicted_status,
                    'predicted_rul': round(predicted_rul, 2),
                    'probabilities': [float(p) for p in class_predictions[i]],
                    'previous_status': previous_status
                })
                
            except Exception as e:
                logger.error(f"Error processing motor {motor_id}: {e}")
                results.append({
                    'motor_id': motor_id,
                    'error': f'Processing failed: {str(e)}',
                    'success': False
                })
        
        # Batch database operations
        batch_update_motor_statuses(status_updates)
        batch_create_alerts(alerts_to_create)
        
        # Add failed motors to results
        failed_results = [{'motor_id': mid, 'error': f'Not enough data. Need {REQUIRED_SEQUENCE_LENGTH} readings.', 'success': False}
                         for mid in invalid_motors]
        
        processing_time = round(time.time() - start_time, 3)
        
        return jsonify({
            'success': True,
            'total_requested': len(motor_ids),
            'successful_predictions': len([r for r in results if r.get('success', True)]),
            'failed_predictions': len(failed_results) + len([r for r in results if not r.get('success', True)]),
            'alerts_created': len(alerts_to_create),
            'processing_time_seconds': processing_time,
            'predictions': results,
            'failed_motors': failed_results,
            'timestamp': datetime.now().isoformat()
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/predict/all', methods=['POST'])
@api_key_required
def predict_all_active():
    """Predict for all motors in the system."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Get all motors
        motor_ids = get_all_active_motors()
        if not motor_ids:
            return jsonify({'error': 'No motors found in the system'}), 404
        
        logger.info(f"Starting prediction for all {len(motor_ids)} motors")
        
        # Use the batch prediction logic
        start_time = time.time()
        
        # Get motor sequences and previous statuses
        motor_sequences = get_multiple_sequences_from_db(motor_ids)
        previous_statuses = get_multiple_motor_statuses(motor_ids)
        
        # Filter out motors without sufficient data
        valid_motors = {mid: seq for mid, seq in motor_sequences.items() if len(seq) >= REQUIRED_SEQUENCE_LENGTH}
        invalid_motors = [mid for mid in motor_ids if mid not in valid_motors]
        
        if not valid_motors:
            return jsonify({
                'error': 'No motors have sufficient data for prediction',
                'total_motors_found': len(motor_ids),
                'invalid_motors': invalid_motors
            }), 404
        
        # Process batch prediction
        motor_list = list(valid_motors.keys())
        
        # Handle null values
        processed_sequences = []
        for motor_id in motor_list:
            seq_df = valid_motors[motor_id]
            if seq_df.isnull().any().any():
                logger.warning(f"Motor {motor_id} has null values, filling with column means")
                seq_df = seq_df.fillna(seq_df.mean())
            processed_sequences.append(feature_scaler.transform(seq_df))
        
        sequences_array = np.array(processed_sequences)
        
        # Batch prediction with thread safety
        with model_lock:
            predictions = model.predict(sequences_array, verbose=0)
            if len(predictions) != 2:
                raise ValueError("Model should return both classification and regression predictions")
        
        class_predictions, reg_predictions = predictions
        
        # Process results
        results = []
        alerts_to_create = []
        status_updates = []
        
        for i, motor_id in enumerate(motor_list):
            try:
                predicted_class_index = np.argmax(class_predictions[i])
                predicted_status = status_map.get(predicted_class_index, "Unknown")
                predicted_rul = max(0, float(rul_scaler.inverse_transform([reg_predictions[i]])[0][0]))
                previous_status = previous_statuses.get(motor_id, 'Optimal')
                
                # Check if alert is needed - FIXED logic
                status_priority = {'Optimal': 0, 'Degrading': 1, 'Critical': 2}
                if (predicted_status != 'Optimal' and 
                    status_priority.get(predicted_status, -1) > status_priority.get(previous_status, -1)):
                    alerts_to_create.append({
                        'motor_id': motor_id,
                        'severity': predicted_status,
                        'message': f"Status changed from {previous_status} to {predicted_status}. Predicted RUL: {predicted_rul:.0f}"
                    })
                
                status_updates.append((motor_id, predicted_status))
                
                results.append({
                    'motor_id': motor_id,
                    'predicted_status': predicted_status,
                    'predicted_rul': round(predicted_rul, 2),
                    'probabilities': [float(p) for p in class_predictions[i]],
                    'previous_status': previous_status
                })
                
            except Exception as e:
                logger.error(f"Error processing motor {motor_id}: {e}")
                results.append({
                    'motor_id': motor_id,
                    'error': f'Processing failed: {str(e)}',
                    'success': False
                })
        
        # Batch database operations
        batch_update_motor_statuses(status_updates)
        batch_create_alerts(alerts_to_create)
        
        # Add failed motors to results
        failed_results = [{'motor_id': mid, 'error': f'Not enough data. Need {REQUIRED_SEQUENCE_LENGTH} readings.', 'success': False}
                         for mid in invalid_motors]
        
        processing_time = round(time.time() - start_time, 3)
        
        return jsonify({
            'success': True,
            'total_requested': len(motor_ids),
            'successful_predictions': len([r for r in results if r.get('success', True)]),
            'failed_predictions': len(failed_results) + len([r for r in results if not r.get('success', True)]),
            'alerts_created': len(alerts_to_create),
            'processing_time_seconds': processing_time,
            'predictions': results,
            'failed_motors': failed_results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Predict all motors error: {e}")
        return jsonify({'error': f'Predict all failed: {str(e)}'}), 500

@app.route('/alerts', methods=['GET'])
@api_key_required
def get_alerts():
    """Fetch alerts with optional filtering."""
    try:
        # Get query parameters
        motor_id = request.args.get('motor_id')
        severity = request.args.get('severity')
        acknowledged = request.args.get('acknowledged', 'false').lower()
        limit = request.args.get('limit', 100, type=int)
        
        # Validate parameters
        if motor_id and not validate_motor_id(motor_id):
            return jsonify({'error': 'Invalid motor ID'}), 400
        
        if severity and not validate_severity(severity):
            return jsonify({'error': 'Invalid severity'}), 400
        
        if limit > 1000:  # Prevent excessive data retrieval
            return jsonify({'error': 'Limit too high. Maximum: 1000'}), 400
        
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query dynamically
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if motor_id:
                query += " AND motor_id = ?"
                params.append(motor_id)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if acknowledged == 'false':
                query += " AND acknowledged = 0"
            elif acknowledged == 'true':
                query += " AND acknowledged = 1"
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            alerts = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        return jsonify({'error': f'Failed to fetch alerts: {str(e)}'}), 500

@app.route('/alerts/<int:alert_id>/ack', methods=['POST'])
@api_key_required
def acknowledge_alert(alert_id):
    """Mark a specific alert as acknowledged."""
    try:
        if alert_id <= 0:
            return jsonify({'error': 'Invalid alert ID'}), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE alerts SET acknowledged = 1 WHERE alert_id = ?", (alert_id,))
            conn.commit()
            
            if cursor.rowcount == 0:
                return jsonify({'error': 'Alert not found'}), 404
        
        logger.info(f"Alert {alert_id} acknowledged")
        return jsonify({
            'message': f'Alert {alert_id} acknowledged',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        return jsonify({'error': f'Failed to acknowledge alert: {str(e)}'}), 500

@app.route('/alerts/batch/ack', methods=['POST'])
@api_key_required
def batch_acknowledge_alerts():
    """Batch acknowledge multiple alerts."""
    try:
        data = request.get_json()
        if not data or 'alert_ids' not in data:
            return jsonify({'error': 'Request must contain alert_ids array'}), 400
        
        alert_ids = data['alert_ids']
        if not isinstance(alert_ids, list) or len(alert_ids) == 0:
            return jsonify({'error': 'alert_ids must be a non-empty array'}), 400
        
        # Validate alert IDs
        if not all(isinstance(aid, int) and aid > 0 for aid in alert_ids):
            return jsonify({'error': 'All alert IDs must be positive integers'}), 400
        
        if len(alert_ids) > 100:  # Reasonable limit
            return jsonify({'error': 'Too many alert IDs. Maximum: 100'}), 400
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in alert_ids])
            query = f"UPDATE alerts SET acknowledged = 1 WHERE alert_id IN ({placeholders})"
            cursor.execute(query, alert_ids)
            conn.commit()
            acknowledged_count = cursor.rowcount
        
        logger.info(f"Batch acknowledged {acknowledged_count} alerts")
        return jsonify({
            'message': f'{acknowledged_count} alerts acknowledged',
            'acknowledged_count': acknowledged_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error batch acknowledging alerts: {e}")
        return jsonify({'error': f'Failed to batch acknowledge alerts: {str(e)}'}), 500

@app.route('/explain/status/<string:motor_id>', methods=['GET'])
@api_key_required
def explain_status(motor_id):
    """Explain status predictions using SHAP."""
    try:
        if classification_explainer is None:
            return jsonify({'error': 'Classification explainer not available'}), 503
        
        if not validate_motor_id(motor_id):
            return jsonify({'error': 'Invalid motor ID'}), 400
        
        sequence_df = get_latest_sequence_from_db(motor_id)
        if len(sequence_df) < REQUIRED_SEQUENCE_LENGTH:
            return jsonify({'error': f'Not enough data to explain. Need {REQUIRED_SEQUENCE_LENGTH} readings.'}), 404
        
        # Handle null values
        if sequence_df.isnull().any().any():
            logger.warning(f"Motor {motor_id} has null values for explanation, filling with column means")
            sequence_df = sequence_df.fillna(sequence_df.mean())
        
        with model_lock:
            scaled_sequence = feature_scaler.transform(sequence_df)
            reshaped_sequence = np.expand_dims(scaled_sequence, axis=0)
            shap_values = classification_explainer.shap_values(reshaped_sequence)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_vals = shap_values
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 1))
        
        # Ensure mean_abs_shap is 1D
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        
        # Create feature importance pairs and sort them
        feature_importance_pairs = list(zip(feature_cols, mean_abs_shap))
        feature_importance = sorted(feature_importance_pairs, key=lambda x: float(x[1]), reverse=True)
        
        return jsonify({
            'motor_id': motor_id,
            'explanation_for': 'status',
            'feature_importance': {feature: float(importance) for feature, importance in feature_importance},
            'top_features': feature_importance[:10],  # Top 10 most important features
            'timestamp': datetime.now().isoformat()
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Status explanation error for motor {motor_id}: {e}")
        return jsonify({'error': f'Status explanation failed: {str(e)}'}), 500

@app.route('/explain/rul/<string:motor_id>', methods=['GET'])
@api_key_required
def explain_rul(motor_id):
    """Explain RUL predictions using SHAP."""
    try:
        if regression_explainer is None:
            return jsonify({'error': 'Regression explainer not available'}), 503
        
        if not validate_motor_id(motor_id):
            return jsonify({'error': 'Invalid motor ID'}), 400
        
        sequence_df = get_latest_sequence_from_db(motor_id)
        if len(sequence_df) < REQUIRED_SEQUENCE_LENGTH:
            return jsonify({'error': f'Not enough data to explain. Need {REQUIRED_SEQUENCE_LENGTH} readings.'}), 404
        
        # Handle null values
        if sequence_df.isnull().any().any():
            logger.warning(f"Motor {motor_id} has null values for explanation, filling with column means")
            sequence_df = sequence_df.fillna(sequence_df.mean())
        
        with model_lock:
            scaled_sequence = feature_scaler.transform(sequence_df)
            reshaped_sequence = np.expand_dims(scaled_sequence, axis=0)
            shap_values = regression_explainer.shap_values(reshaped_sequence)
        
        # For regression, shap_values should be a single array
        if isinstance(shap_values, list):
            shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
        else:
            shap_vals = shap_values
        
        # Calculate mean absolute SHAP values across batch and time dimensions
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=(0, 1))
        
        # Ensure mean_abs_shap is 1D
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        
        # Create feature importance pairs and sort them
        feature_importance_pairs = list(zip(feature_cols, mean_abs_shap))
        feature_importance = sorted(feature_importance_pairs, key=lambda x: float(x[1]), reverse=True)
        
        return jsonify({
            'motor_id': motor_id,
            'explanation_for': 'rul',
            'feature_importance': {feature: float(importance) for feature, importance in feature_importance},
            'top_features': feature_importance[:10],  # Top 10 most important features
            'timestamp': datetime.now().isoformat()
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"RUL explanation error for motor {motor_id}: {e}")
        return jsonify({'error': f'RUL explanation failed: {str(e)}'}), 500

@app.route('/motors', methods=['GET'])
@api_key_required
def get_motors():
    """Get list of all motors with their current status."""
    try:
        motor_ids = get_all_active_motors()
        if not motor_ids:
            return jsonify({
                'motors': [],
                'count': 0,
                'timestamp': datetime.now().isoformat()
            })
        
        motor_statuses = get_multiple_motor_statuses(motor_ids)
        
        motors_info = []
        for motor_id in motor_ids:
            motors_info.append({
                'motor_id': motor_id,
                'status': motor_statuses.get(motor_id, 'Optimal')
            })
        
        return jsonify({
            'motors': motors_info,
            'count': len(motors_info),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching motors: {e}")
        return jsonify({'error': f'Failed to fetch motors: {str(e)}'}), 500

# Endpoint to fetch the latest readings for all motors

@app.route('/motors/readings/latest', methods=['GET'])
@api_key_required
def get_latest_readings_for_all_motors():
    """
    Fetches the single most recent sensor reading for all active motors.
    Ideal for a live overview dashboard.
    """
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            # This SQL query efficiently gets the latest row for each motor
            query = """
            SELECT r.* FROM sensor_readings r
            INNER JOIN (
                SELECT motor_id, MAX(timestamp) as max_ts
                FROM sensor_readings
                GROUP BY motor_id
            ) latest ON r.motor_id = latest.motor_id AND r.timestamp = latest.max_ts
            """
            cursor = conn.cursor()
            cursor.execute(query)
            readings = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            'latest_readings': readings,
            'count': len(readings),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error fetching latest readings for all motors: {e}")
        return jsonify({'error': f'Failed to fetch latest readings: {str(e)}'}), 500

# Endpoint to fetch historical readings for a specific motor

@app.route('/motors/<string:motor_id>/readings', methods=['GET'])
@api_key_required
def get_motor_readings_history(motor_id):
    """
    Fetches historical raw sensor data for a specific motor.
    Ideal for populating charts on a detail screen.
    """
    try:
        if not validate_motor_id(motor_id):
            return jsonify({'error': 'Invalid motor ID format'}), 400
            
        # Get a query parameter to control how many readings to return
        limit = request.args.get('limit', default=200, type=int)
        if limit > 1000: # Prevent excessive data retrieval
            return jsonify({'error': 'Limit too high. Maximum: 1000'}), 400

        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM sensor_readings WHERE motor_id = ? ORDER BY timestamp DESC LIMIT ?"
            cursor = conn.cursor()
            cursor.execute(query, (motor_id, limit))
            readings = [dict(row) for row in cursor.fetchall()]
        
        # Return in chronological order for charting
        return jsonify({
            'motor_id': motor_id,
            'readings': list(reversed(readings)),
            'count': len(readings),
            'timestamp': datetime.now().isoformat()
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error fetching readings for {motor_id}: {e}")
        return jsonify({'error': f'Failed to fetch readings: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    
    # Load model assets
    if not load_model_assets():
        logger.error("Failed to load model assets. Exiting.")
        exit(1)
    
    # Initialize explainers (optional)
    initialize_explainers()
    
    logger.info("🚀 Starting server on port 5001...")
    app.run(host='0.0.0.0', use_reloader=False, port=5001, threaded=True)