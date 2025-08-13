import sqlite3
import random
import time
import math
from datetime import datetime
import sys
import logging
from typing import Dict, List, Tuple, Optional

# Configuration constants
DB_FILE = "motors.db"
SLEEP_INTERVAL = 5  # seconds between readings
BATCH_SIZE = 10  # number of readings to batch before commit
REPAIR_THRESHOLD = 50  # time steps before auto-repair
MAX_CONSECUTIVE_ERRORS = 10
HEALTH_THRESHOLDS = {'CRITICAL': 30, 'DEGRADING': 70}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('motor_simulator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MotorState:
    """Enum-like class for motor states."""
    HEALTHY = 'HEALTHY'
    DEGRADING = 'DEGRADING'
    CRITICAL = 'CRITICAL'

class MotorSimulator:
    """Manages the cyclical state and data generation for a single motor."""
    
    # Define sensor parameter order to ensure consistency
    SENSOR_PARAMS = [
        'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
        's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
        's18', 's19', 's20', 's21'
    ]
    
    def __init__(self, motor_id: str):
        self.motor_id = motor_id
        self.health = 100.0
        self.time_step = 0
        self.state = MotorState.HEALTHY
        self.time_in_critical = 0
        self.repair_count = 0
        self.last_state_change = datetime.now()
        
        # Initialize base readings
        self.base_readings = self._get_initial_state()
        
        logger.info(f"Initialized simulator for motor {motor_id}")

    def _get_initial_state(self) -> Dict[str, float]:
        """Returns a baseline healthy reading with proper parameter names."""
        return {
            'setting1': 0.45, 'setting2': 0.6, 'setting3': 0.5,
            's1': 0.2, 's2': 0.3, 's3': 0.4, 's4': 0.25, 's5': 0.1,
            's6': 0.15, 's7': 0.35, 's8': 0.5, 's9': 0.6, 's10': 0.1,
            's11': 0.2, 's12': 0.4, 's13': 0.55, 's14': 0.7, 's15': 0.3,
            's16': 0.1, 's17': 0.4, 's18': 0.1, 's19': 0.1, 's20': 0.3,
            's21': 0.3
        }

    def _update_state(self) -> bool:
        """Update the state based on current health. Returns True if state changed."""
        old_state = self.state
        
        if self.health < HEALTH_THRESHOLDS['CRITICAL']:
            if self.state != MotorState.CRITICAL:
                self.time_in_critical = 0  # Reset timer when entering critical
                self.last_state_change = datetime.now()
            self.state = MotorState.CRITICAL
        elif self.health < HEALTH_THRESHOLDS['DEGRADING']:
            if self.state != MotorState.DEGRADING:
                self.last_state_change = datetime.now()
            self.state = MotorState.DEGRADING
            self.time_in_critical = 0
        else:
            if self.state != MotorState.HEALTHY:
                self.last_state_change = datetime.now()
            self.state = MotorState.HEALTHY
            self.time_in_critical = 0
        
        state_changed = old_state != self.state
        if state_changed:
            logger.info(f"Motor {self.motor_id} state changed: {old_state} → {self.state} (Health: {self.health:.1f}%)")
        
        return state_changed

    def _simulate_repair(self) -> None:
        """Simulate motor repair and reset to healthy state."""
        logger.warning(f"🔧 Simulating repair for {self.motor_id} (repair #{self.repair_count + 1})")
        self.health = random.uniform(85.0, 100.0)  # Don't always repair to 100%
        self.time_in_critical = 0
        self.repair_count += 1
        self.last_state_change = datetime.now()
        self._update_state()

    def advance_time_step(self) -> None:
        """Advance the simulation by one time step."""
        self.time_step += 1
        
        # Handle critical state repairs
        if self.state == MotorState.CRITICAL:
            self.time_in_critical += 1
            if self.time_in_critical > REPAIR_THRESHOLD:
                self._simulate_repair()
                return
        
        # Degrade health over time with some randomness
        if self.state == MotorState.HEALTHY:
            degradation = random.uniform(0.02, 0.08)
        elif self.state == MotorState.DEGRADING:
            degradation = random.uniform(0.08, 0.15)
        else:  # CRITICAL
            degradation = random.uniform(0.1, 0.2)
        
        # Add occasional random failures
        if random.random() < 0.001:  # 0.1% chance of sudden degradation
            degradation *= random.uniform(5, 10)
            logger.warning(f"Sudden degradation event for motor {self.motor_id}")
        
        self.health = max(0, self.health - degradation)
        self._update_state()

    def generate_reading(self) -> List[float]:
        """Generate a new sensor reading based on the current health and state."""
        # Create a fresh copy of base readings
        reading = self.base_readings.copy()
        health_factor = (100 - self.health) / 100.0

        # Determine noise and drift based on state
        if self.state == MotorState.HEALTHY:
            noise_level = 0.005
            drift_factor = 0.05
        elif self.state == MotorState.DEGRADING:
            noise_level = 0.02
            drift_factor = 0.2
        else:  # CRITICAL
            noise_level = 0.05
            drift_factor = 0.4

        # Apply cyclical patterns to key sensors
        cycle_time = self.time_step * 0.1
        reading['s4'] += health_factor * drift_factor + math.sin(cycle_time) * 0.01
        reading['s11'] += health_factor * drift_factor + math.cos(cycle_time * 0.7) * 0.008
        reading['s14'] += health_factor * drift_factor * 0.5 + math.sin(cycle_time * 1.3) * 0.005
        
        # Add temperature-like drift to some sensors
        if self.state != MotorState.HEALTHY:
            temp_drift = health_factor * 0.3
            reading['s7'] += temp_drift
            reading['s13'] += temp_drift * 0.8

        # Apply noise and ensure values stay within bounds [0, 1]
        for key in reading:
            jitter = random.uniform(-noise_level, noise_level)
            reading[key] = max(0.0, min(1.0, reading[key] + jitter))

        # Return values in consistent order
        return [reading[param] for param in self.SENSOR_PARAMS]

    def get_status_summary(self) -> Dict[str, any]:
        """Get a summary of the motor's current status."""
        return {
            'motor_id': self.motor_id,
            'health': round(self.health, 1),
            'state': self.state,
            'time_step': self.time_step,
            'time_in_critical': self.time_in_critical,
            'repair_count': self.repair_count,
            'last_state_change': self.last_state_change.isoformat()
        }

class DatabaseManager:
    """Handles database operations with batching and error recovery."""
    
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.pending_readings = []
        
    def get_motor_ids(self) -> List[str]:
        """Retrieve all motor IDs from the database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT motor_id FROM motors ORDER BY motor_id")
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving motor IDs: {e}")
            return []

    def add_reading(self, motor_id: str, reading: List[float]) -> None:
        """Add a reading to the pending batch."""
        timestamp = datetime.now().isoformat()
        values = (motor_id, timestamp) + tuple(reading)
        self.pending_readings.append(values)

    def commit_batch(self) -> bool:
        """Commit all pending readings to the database."""
        if not self.pending_readings:
            return True
            
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                sql = """
                    INSERT INTO sensor_readings
                    (motor_id, timestamp, setting1, setting2, setting3, s1, s2, s3, s4, s5, s6, s7, s8, s9,
                     s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.executemany(sql, self.pending_readings)
                conn.commit()
                
                readings_count = len(self.pending_readings)
                logger.debug(f"Successfully committed {readings_count} readings to database")
                self.pending_readings.clear()
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error committing batch to database: {e}")
            return False

    def update_motor_status(self, motor_id: str, status: str) -> bool:
        """Update the latest status of a motor."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE motors SET latest_status = ? WHERE motor_id = ?",
                    (status, motor_id)
                )
                conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error updating motor status: {e}")
            return False

def main():
    """Main loop to run the motor simulators and insert data."""
    try:
        # Initialize database manager
        db_manager = DatabaseManager(DB_FILE)
        motor_ids = db_manager.get_motor_ids()
        
        if not motor_ids:
            logger.error("No motors found. Please run database_setup.py first.")
            return

        # Initialize simulators
        simulators = {motor_id: MotorSimulator(motor_id) for motor_id in motor_ids}
        
        logger.info(f"Starting motor simulation with {len(motor_ids)} motors")
        logger.info(f"Batch size: {BATCH_SIZE}, Sleep interval: {SLEEP_INTERVAL}s")
        print(f"🚀 Monitoring {len(motor_ids)} motors. Press CTRL+C to stop.\n")
        
        consecutive_errors = 0
        cycle_count = 0
        
        while True:
            cycle_count += 1
            batch_success = True
            status_updates = []
            
            # Generate readings for all motors
            for motor_id, simulator in simulators.items():
                try:
                    # Advance simulation state
                    simulator.advance_time_step()
                    
                    # Generate and store reading
                    reading = simulator.generate_reading()
                    db_manager.add_reading(motor_id, reading)
                    
                    # Track status changes
                    status_updates.append((motor_id, simulator.state))
                    
                except Exception as e:
                    logger.error(f"Error processing motor {motor_id}: {e}")
                    batch_success = False
            
            # Commit batch to database
            if len(db_manager.pending_readings) >= BATCH_SIZE or cycle_count % 10 == 0:
                if not db_manager.commit_batch():
                    batch_success = False
            
            # Update motor statuses
            for motor_id, status in status_updates:
                db_manager.update_motor_status(motor_id, status)
            
            # Display current status
            if cycle_count % 5 == 0:  # Show status every 5 cycles
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cycle {cycle_count}")
                for motor_id, sim in simulators.items():
                    status_icon = {
                        MotorState.HEALTHY: "✅",
                        MotorState.DEGRADING: "⚠️",
                        MotorState.CRITICAL: "🚨"
                    }[sim.state]
                    print(f"  {status_icon} {motor_id}: {sim.health:.1f}% ({sim.state})")
            
            # Handle consecutive errors
            if not batch_success:
                consecutive_errors += 1
                logger.warning(f"Batch failed (consecutive errors: {consecutive_errors})")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}). Stopping generator.")
                    break
            else:
                consecutive_errors = 0
            
            time.sleep(SLEEP_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("🛑 Data generator stopped by user")
        print("\n🛑 Stopping simulator...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Final commit of any pending readings
        if 'db_manager' in locals():
            db_manager.commit_batch()
        logger.info("Motor simulator shutdown complete")

if __name__ == '__main__':
    main()