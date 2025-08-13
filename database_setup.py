import sqlite3
from datetime import datetime

DB_FILE = 'motors.db'

def create_database():
    """Create the motors database with proper tables and sample data."""
    conn = None
    try:
        # Connect to (or create) the database file
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        print("Creating database tables...")
        
        # --- Create the 'motors' table to store static info ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS motors (
                motor_id TEXT PRIMARY KEY,
                motor_type TEXT NOT NULL,
                installation_date TEXT NOT NULL,
                latest_status TEXT DEFAULT 'Optimal',
                active INTEGER DEFAULT 1 
            )
        ''')
        
        # --- Create the 'sensor_readings' table for the live feed ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                motor_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                setting1 REAL, setting2 REAL, setting3 REAL,
                s1 REAL, s2 REAL, s3 REAL, s4 REAL, s5 REAL, s6 REAL,
                s7 REAL, s8 REAL, s9 REAL, s10 REAL, s11 REAL, s12 REAL,
                s13 REAL, s14 REAL, s15 REAL, s16 REAL, s17 REAL, s18 REAL,
                s19 REAL, s20 REAL, s21 REAL,
                FOREIGN KEY (motor_id) REFERENCES motors (motor_id) ON DELETE CASCADE
            )
        ''')
        
        # --- Create the 'alerts' table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                motor_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('Optimal','Degrading', 'Critical', 'Warning')),
                message TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0 CHECK (acknowledged IN (0, 1)),
                FOREIGN KEY (motor_id) REFERENCES motors (motor_id) ON DELETE CASCADE
            )
        ''')
        
        # Handle existing databases that might not have the latest_status column
        try:
            cursor.execute("SELECT latest_status FROM motors LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            cursor.execute("ALTER TABLE motors ADD COLUMN latest_status TEXT DEFAULT 'Optimal'")
            print("Added 'latest_status' column to motors table.")
            
        # --- Add sample motors to the database ---
        sample_motors = [
            ('Motor-A-01', 'AC Induction', '2024-01-15'),
            ('Motor-B-02', 'DC Brushless', '2024-03-20'),
            ('Motor-C-03', 'Servo Motor', '2024-02-10'),
            ('Motor-D-04', 'Stepper Motor', '2024-04-05'),
        ]
        
        cursor.executemany(
            'INSERT OR IGNORE INTO motors (motor_id, motor_type, installation_date) VALUES (?, ?, ?)', 
            sample_motors
        )
        
        # Create indexes for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sensor_readings_motor_timestamp 
            ON sensor_readings (motor_id, timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_motor_timestamp 
            ON alerts (motor_id, timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alerts_severity 
            ON alerts (severity, acknowledged)
        ''')
        
        conn.commit()
        
        # Verification steps...
        cursor.execute("SELECT COUNT(*) FROM motors")
        motor_count = cursor.fetchone()[0]
        
        print(f"✅ Database '{DB_FILE}' updated/verified successfully!")
        print(f"✅ Tables ready: 'motors', 'sensor_readings', and 'alerts'")
        print(f"✅ {motor_count} sample motors present")
        print(f"✅ Foreign key constraints enabled")
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error creating/updating database: {e}")
        return False
    
    finally:
        if conn:
            conn.close()

def verify_database():
    """Verify database structure and compatibility with Flask app."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        print("\n🔍 Verifying database compatibility...")
        
        # Check table existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['motors', 'sensor_readings', 'alerts']
        all_tables_exist = True
        for table in required_tables:
            if table in tables:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' missing")
                all_tables_exist = False

        if not all_tables_exist: 
            return False
        
        # Check motors table structure
        cursor.execute("PRAGMA table_info(motors)")
        motor_columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        required_motor_columns = {
            'motor_id': 'TEXT',
            'motor_type': 'TEXT',
            'installation_date': 'TEXT',
            'latest_status': 'TEXT'
        }
        
        for col_name, col_type in required_motor_columns.items():
            if col_name in motor_columns:
                print(f"✅ Column '{col_name}' exists in motors table")
            else:
                print(f"❌ Column '{col_name}' missing from motors table")
                return False
        
        # Check alerts table structure
        cursor.execute("PRAGMA table_info(alerts)")
        alert_columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        required_alert_columns = {
            'alert_id': 'INTEGER',
            'motor_id': 'TEXT',
            'timestamp': 'TEXT',
            'severity': 'TEXT',
            'message': 'TEXT',
            'acknowledged': 'INTEGER'
        }
        
        for col_name, col_type in required_alert_columns.items():
            if col_name in alert_columns:
                print(f"✅ Column '{col_name}' exists in alerts table")
            else:
                print(f"❌ Column '{col_name}' missing from alerts table")
                return False
        
        # Check foreign key constraints are enabled
        cursor.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]
        if fk_enabled:
            print("✅ Foreign key constraints are enabled")
        else:
            print("⚠️  Foreign key constraints are disabled")

        print(f"\n✅ Database is compatible and ready for the alerting system!")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database verification error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during verification: {e}")
        return False
    
    finally:
        if conn:
            conn.close()

def clear_sensor_data():
    """Clear all sensor readings and alerts (keeping motors table intact)."""
    conn = None
    try:
        # Add confirmation prompt for safety
        confirmation = input("⚠️  This will delete ALL sensor readings and alerts. Continue? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return False
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON")
        
        cursor.execute("DELETE FROM sensor_readings")
        deleted_readings = cursor.rowcount
        
        cursor.execute("DELETE FROM alerts")
        deleted_alerts = cursor.rowcount
        
        # Reset motor status to Optimal
        cursor.execute("UPDATE motors SET latest_status = 'Optimal'")
        updated_motors = cursor.rowcount
        
        conn.commit()
        
        print(f"✅ Cleared {deleted_readings} sensor readings and {deleted_alerts} alerts.")
        print(f"✅ Reset status for {updated_motors} motors to 'Optimal'.")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error clearing data: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error clearing data: {e}")
        return False
    
    finally:
        if conn:
            conn.close()

def backup_database(backup_path=None):
    """Create a backup of the database."""
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"motors_backup_{timestamp}.db"
    
    conn = None
    backup_conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        backup_conn = sqlite3.connect(backup_path)
        
        conn.backup(backup_conn)
        print(f"✅ Database backed up to: {backup_path}")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Backup error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during backup: {e}")
        return False
    
    finally:
        if conn:
            conn.close()
        if backup_conn:
            backup_conn.close()

def main():
    """Main function to set up the database."""
    print("🚀 Setting up motors database...")
    
    if create_database():
        verify_database()
        print(f"\n🎉 Setup complete!")
    else:
        print("❌ Database setup failed.")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clear':
            clear_sensor_data()
        elif sys.argv[1] == '--verify':
            verify_database()
        elif sys.argv[1] == '--backup':
            backup_path = sys.argv[2] if len(sys.argv) > 2 else None
            backup_database(backup_path)
        else:
            print("Usage: python database_setup.py [--clear|--verify|--backup [path]]")
    else:
        main()