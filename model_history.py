import sqlite3
import json
from datetime import datetime
import os
import yaml
import logging

class ModelHistory:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.db_path = config['model']['history_db']
        self.config_path = config_path
        
        # Ensure checkpoint directory exists first
        os.makedirs('checkpoints', exist_ok=True)  # Create base directory
        
        # Then ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        # Setup logging
        self.logger = logging.getLogger('ModelHistory')
        self.logger.setLevel(logging.INFO)
        
        # Initialize database
        self.setup_database()
        
        # Log initial state
        self.logger.info(f"Initialized ModelHistory with database at {self.db_path}")
        self._log_database_state()
        
    def _log_database_state(self):
        """Log the current state of the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM model_experiments')
                count = cursor.fetchone()[0]
                self.logger.info(f"Current database has {count} experiments")
                
                if count > 0:
                    cursor.execute('''
                        SELECT metric_name, metric_value, timestamp 
                        FROM model_experiments 
                        ORDER BY metric_value DESC 
                        LIMIT 1
                    ''')
                    best = cursor.fetchone()
                    self.logger.info(f"Best result so far: {best[0]}={best[1]:.4f} from {best[2]}")
        except Exception as e:
            self.logger.error(f"Error checking database state: {e}")

    def setup_database(self):
        """Create the database and tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    architecture TEXT,
                    hyperparameters TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    config_path TEXT
                )
            ''')
            conn.commit()

    def save_experiment(self, config_path, metric_value, metric_name):
        """Save experiment results to database."""
        self.logger.info(f"Saving new experiment with {metric_name}={metric_value:.4f}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            architecture = {
                'input_size': config['model']['input_size'],
                'hidden_layers': config['best_model']['hidden_layers'],
                'num_classes': config['model']['num_classes']
            }

            hyperparameters = {
                'dropout_rate': config['best_model']['dropout_rate'],
                'learning_rate': config['best_model']['learning_rate'],
                'use_batch_norm': config['best_model']['use_batch_norm'],
                'weight_decay': config['best_model']['weight_decay']
            }

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_experiments
                    (timestamp, architecture, hyperparameters, metric_name, metric_value, config_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    json.dumps(architecture),
                    json.dumps(hyperparameters),
                    metric_name,
                    metric_value,
                    config_path
                ))
                conn.commit()
                
            self._log_database_state()  # Log updated state
            self.logger.info("Successfully saved experiment")
            
        except Exception as e:
            self.logger.error(f"Error saving experiment: {e}")
            raise

    def get_best_architecture(self, metric_name='f1_score', n_best=5):
        """Get the best performing architecture based on historical data."""
        self.logger.info(f"Fetching best architecture for {metric_name}")
        with sqlite3.connect(self.db_path) as conn:
            # First get the best metric value
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MAX(metric_value) 
                FROM model_experiments 
                WHERE metric_name = ?
            ''', (metric_name,))
            best_value = cursor.fetchone()[0]
            
            # Then get the most recent architectures with that value
            cursor.execute('''
                SELECT architecture, metric_value, timestamp
                FROM model_experiments
                WHERE metric_name = ?
                AND metric_value >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (metric_name, best_value * 0.95, n_best))  # Allow 5% margin from best
            
            results = cursor.fetchall()
            
            if not results:
                self.logger.warning("No historical data found")
                return None
                
            self.logger.info(f"Found {len(results)} previous results near best value of {best_value:.4f}")
            architectures = [json.loads(row[0]) for row in results]
            avg_hidden_layers = []
            
            # Calculate average layer width for each position
            max_layers = max(len(arch['hidden_layers']) for arch in architectures)
            for i in range(max_layers):
                widths = [arch['hidden_layers'][i] for arch in architectures 
                         if i < len(arch['hidden_layers'])]
                if widths:
                    avg_hidden_layers.append(round(sum(widths) / len(widths)))
            
            return {
                'input_size': architectures[0]['input_size'],
                'hidden_layers': avg_hidden_layers,
                'num_classes': architectures[0]['num_classes']
            }

    def get_best_hyperparameters(self, metric_name='f1_score', n_best=5):
        """Get the best performing hyperparameters based on historical data."""
        with sqlite3.connect(self.db_path) as conn:
            # Get best metric value first
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MAX(metric_value) 
                FROM model_experiments 
                WHERE metric_name = ?
            ''', (metric_name,))
            best_value = cursor.fetchone()[0]
            
            # Get most recent hyperparameters near the best value
            cursor.execute('''
                SELECT hyperparameters, metric_value, timestamp
                FROM model_experiments
                WHERE metric_name = ?
                AND metric_value >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (metric_name, best_value * 0.95, n_best))
            
            results = cursor.fetchall()
            
            if not results:
                return None
            
            self.logger.info(f"Using {len(results)} best hyperparameter sets near value {best_value:.4f}")    
            hyperparams_list = [json.loads(row[0]) for row in results]
            # ...rest of existing averaging code...

            if not results:
                return None
                
            # Average the hyperparameters from top performers
            hyperparams_list = [json.loads(row[0]) for row in results]
            avg_hyperparams = {}
            
            numerical_params = ['dropout_rate', 'learning_rate', 'weight_decay']
            bool_params = ['use_batch_norm']
            
            for param in numerical_params:
                values = [h[param] for h in hyperparams_list]
                avg_hyperparams[param] = sum(values) / len(values)
            
            for param in bool_params:
                values = [h[param] for h in hyperparams_list]
                avg_hyperparams[param] = max(set(values), key=values.count)
            
            return avg_hyperparams

    def clear_database(self):
        """Clear all records from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM model_experiments')
                conn.commit()
            self.logger.info("Database cleared successfully")
            self._log_database_state()
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise

def update_default_config(config_path):
    """Update the default configuration with historical best performers."""
    logger = logging.getLogger('ModelHistory')
    logger.info(f"Updating default configuration from {config_path}")
    
    try:
        history = ModelHistory(config_path)
        best_arch = history.get_best_architecture()
        best_params = history.get_best_hyperparameters()
        
        if not best_arch or not best_params:
            logger.warning("No historical data available for updating defaults")
            return
            
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create default_model section if it doesn't exist
        if 'default_model' not in config:
            config['default_model'] = {}
        
        # Update with best parameters
        config['default_model'].update({
            'hidden_layers': best_arch['hidden_layers'],
            'dropout_rate': best_params['dropout_rate'],
            'learning_rate': best_params['learning_rate'],
            'use_batch_norm': best_params['use_batch_norm'],
            'weight_decay': best_params['weight_decay']
        })
        
        logger.info(f"Updated default configuration with: {config['default_model']}")
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise
