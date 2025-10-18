"""
Data Import Script for AutoATC
Import validation data and prepare datasets
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class DataImporter:
    """Import and prepare data for AutoATC validation."""
    
    def __init__(self, db_path: str = "atc.db"):
        """
        Initialize data importer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        
    def import_validation_data(self, csv_file: str, validator_id: str = "validator_001"):
        """
        Import validation data from CSV file.
        
        Args:
            csv_file: Path to CSV file with validation data
            validator_id: ID of the validator
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_columns = ['animal_id', 'manual_atc_score']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Import each validation record
            imported_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Get analysis record
                    analysis_query = """
                    SELECT id FROM animal_analyses 
                    WHERE animal_id = ?
                    """
                    analysis_result = conn.execute(analysis_query, (row['animal_id'],)).fetchone()
                    
                    if not analysis_result:
                        logger.warning(f"Analysis not found for animal {row['animal_id']}")
                        continue
                    
                    analysis_id = analysis_result[0]
                    
                    # Prepare validation data
                    validation_data = {
                        'analysis_id': analysis_id,
                        'manual_atc_score': row['manual_atc_score'],
                        'manual_breed': row.get('manual_breed'),
                        'manual_measurements': json.dumps(row.get('manual_measurements', {})) if 'manual_measurements' in row else None,
                        'validator_id': validator_id,
                        'validator_notes': row.get('validator_notes', ''),
                        'validation_date': datetime.now().isoformat(),
                        'is_approved': True
                    }
                    
                    # Calculate accuracy metrics
                    ai_data = self._get_ai_data(conn, analysis_id)
                    if ai_data:
                        validation_data.update(self._calculate_metrics(ai_data, validation_data))
                    
                    # Insert validation record
                    insert_query = """
                    INSERT INTO validation_records 
                    (analysis_id, manual_atc_score, manual_breed, manual_measurements,
                     validator_id, validator_notes, validation_date, is_approved,
                     atc_score_difference, breed_match, measurement_accuracy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    conn.execute(insert_query, (
                        validation_data['analysis_id'],
                        validation_data['manual_atc_score'],
                        validation_data['manual_breed'],
                        validation_data['manual_measurements'],
                        validation_data['validator_id'],
                        validation_data['validator_notes'],
                        validation_data['validation_date'],
                        validation_data['is_approved'],
                        validation_data.get('atc_score_difference'),
                        validation_data.get('breed_match'),
                        validation_data.get('measurement_accuracy')
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error importing validation for animal {row['animal_id']}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully imported {imported_count} validation records")
            
        except Exception as e:
            logger.error(f"Error importing validation data: {e}")
            raise
    
    def import_breed_data(self, breed_file: str):
        """
        Import breed classification data.
        
        Args:
            breed_file: Path to breed data file (JSON or CSV)
        """
        try:
            if breed_file.endswith('.json'):
                with open(breed_file, 'r') as f:
                    breed_data = json.load(f)
            elif breed_file.endswith('.csv'):
                df = pd.read_csv(breed_file)
                breed_data = df.to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
            
            # Process breed data
            conn = sqlite3.connect(self.db_path)
            
            for breed_info in breed_data:
                # Update analysis record with breed information
                update_query = """
                UPDATE animal_analyses 
                SET breed_predicted = ?, breed_confidence = ?
                WHERE animal_id = ?
                """
                
                conn.execute(update_query, (
                    breed_info.get('breed'),
                    breed_info.get('confidence', 0.0),
                    breed_info.get('animal_id')
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully imported breed data for {len(breed_data)} animals")
            
        except Exception as e:
            logger.error(f"Error importing breed data: {e}")
            raise
    
    def import_measurement_data(self, measurement_file: str):
        """
        Import measurement data.
        
        Args:
            measurement_file: Path to measurement data file
        """
        try:
            if measurement_file.endswith('.json'):
                with open(measurement_file, 'r') as f:
                    measurement_data = json.load(f)
            elif measurement_file.endswith('.csv'):
                df = pd.read_csv(measurement_file)
                measurement_data = df.to_dict('records')
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
            
            # Process measurement data
            conn = sqlite3.connect(self.db_path)
            
            for measurement_info in measurement_data:
                # Update analysis record with measurements
                update_query = """
                UPDATE animal_analyses 
                SET measurements = ?
                WHERE animal_id = ?
                """
                
                measurements_json = json.dumps(measurement_info.get('measurements', {}))
                conn.execute(update_query, (measurements_json, measurement_info.get('animal_id')))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully imported measurement data for {len(measurement_data)} animals")
            
        except Exception as e:
            logger.error(f"Error importing measurement data: {e}")
            raise
    
    def _get_ai_data(self, conn: sqlite3.Connection, analysis_id: int) -> Optional[Dict]:
        """Get AI analysis data for an analysis ID."""
        try:
            query = """
            SELECT animal_id, atc_score, atc_grade, breed_predicted, 
                   measurements, confidence, diseases_detected
            FROM animal_analyses 
            WHERE id = ?
            """
            
            result = conn.execute(query, (analysis_id,)).fetchone()
            
            if result:
                return {
                    'animal_id': result[0],
                    'atc_score': result[1],
                    'atc_grade': result[2],
                    'breed': result[3],
                    'measurements': json.loads(result[4]) if result[4] else {},
                    'confidence': result[5],
                    'diseases': json.loads(result[6]) if result[6] else []
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting AI data: {e}")
            return None
    
    def _calculate_metrics(self, ai_data: Dict, validation_data: Dict) -> Dict:
        """Calculate accuracy metrics between AI and manual data."""
        try:
            metrics = {}
            
            # ATC Score difference
            if ai_data.get('atc_score') and validation_data.get('manual_atc_score'):
                ai_score = ai_data['atc_score']
                manual_score = validation_data['manual_atc_score']
                metrics['atc_score_difference'] = abs(ai_score - manual_score)
            
            # Breed match
            if ai_data.get('breed') and validation_data.get('manual_breed'):
                ai_breed = ai_data['breed'].lower() if ai_data['breed'] else ''
                manual_breed = validation_data['manual_breed'].lower() if validation_data['manual_breed'] else ''
                metrics['breed_match'] = ai_breed == manual_breed
            
            # Measurement accuracy
            if ai_data.get('measurements') and validation_data.get('manual_measurements'):
                ai_measurements = ai_data['measurements']
                manual_measurements = json.loads(validation_data['manual_measurements']) if isinstance(validation_data['manual_measurements'], str) else validation_data['manual_measurements']
                
                if manual_measurements:
                    accuracies = []
                    for key, manual_value in manual_measurements.items():
                        if key in ai_measurements and manual_value > 0:
                            ai_value = ai_measurements[key]
                            if ai_value > 0:
                                accuracy = 1 - abs(ai_value - manual_value) / manual_value
                                accuracies.append(max(0, accuracy))
                    
                    if accuracies:
                        metrics['measurement_accuracy'] = np.mean(accuracies)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def create_sample_data(self, output_file: str = "sample_validation_data.csv"):
        """
        Create sample validation data for testing.
        
        Args:
            output_file: Output CSV file path
        """
        try:
            # Generate sample data
            np.random.seed(42)
            n_samples = 100
            
            sample_data = []
            
            for i in range(n_samples):
                animal_id = f"ATC_{i+1:03d}"
                manual_atc_score = np.random.normal(75, 15)
                manual_atc_score = max(0, min(100, manual_atc_score))  # Clamp to 0-100
                
                # Generate manual breed
                breeds = ['Gir', 'Sahiwal', 'Murrah', 'Nili-Ravi', 'Jafrabadi', 'Surti']
                manual_breed = np.random.choice(breeds)
                
                # Generate manual measurements
                manual_measurements = {
                    'height': np.random.normal(120, 20),
                    'length': np.random.normal(150, 25),
                    'girth': np.random.normal(180, 30),
                    'width': np.random.normal(60, 10)
                }
                
                # Generate validator notes
                notes_options = [
                    "Good quality image, clear analysis",
                    "Slightly blurry image, but analysis seems accurate",
                    "Excellent image quality, very confident in manual scores",
                    "Difficult to assess due to poor lighting",
                    "Animal not in ideal position for analysis"
                ]
                validator_notes = np.random.choice(notes_options)
                
                sample_data.append({
                    'animal_id': animal_id,
                    'manual_atc_score': round(manual_atc_score, 1),
                    'manual_breed': manual_breed,
                    'manual_measurements': json.dumps(manual_measurements),
                    'validator_notes': validator_notes
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(sample_data)
            df.to_csv(output_file, index=False)
            
            logger.info(f"Sample validation data created: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            raise

def main():
    """Main function for data import."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoATC Data Import')
    parser.add_argument('--action', choices=['import', 'create-sample'], required=True, help='Action to perform')
    parser.add_argument('--file', type=str, help='Input file path')
    parser.add_argument('--type', choices=['validation', 'breed', 'measurement'], help='Data type')
    parser.add_argument('--validator', type=str, default='validator_001', help='Validator ID')
    parser.add_argument('--output', type=str, default='sample_validation_data.csv', help='Output file for sample data')
    parser.add_argument('--db', type=str, default='atc.db', help='Database path')
    
    args = parser.parse_args()
    
    # Initialize importer
    importer = DataImporter(args.db)
    
    if args.action == 'create-sample':
        importer.create_sample_data(args.output)
        print(f"Sample data created: {args.output}")
        
    elif args.action == 'import':
        if not args.file or not args.type:
            print("Error: --file and --type are required for import action")
            return
        
        if args.type == 'validation':
            importer.import_validation_data(args.file, args.validator)
        elif args.type == 'breed':
            importer.import_breed_data(args.file)
        elif args.type == 'measurement':
            importer.import_measurement_data(args.file)
        
        print(f"Successfully imported {args.type} data from {args.file}")

if __name__ == "__main__":
    main()
