"""
Validation Scripts for AutoATC
Compare AI-generated scores with manual ATC scores and generate accuracy reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationAnalyzer:
    """Analyzes validation data and generates accuracy reports."""
    
    def __init__(self, db_path: str = "atc.db"):
        """
        Initialize validation analyzer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.results = {}
        
    def load_validation_data(self, days: int = 30) -> pd.DataFrame:
        """
        Load validation data from database.
        
        Args:
            days: Number of days to load data for
            
        Returns:
            DataFrame with validation data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                va.animal_id,
                va.atc_score as ai_atc_score,
                va.atc_grade as ai_atc_grade,
                va.breed_predicted as ai_breed,
                va.measurements as ai_measurements,
                va.confidence as ai_confidence,
                vr.manual_atc_score,
                vr.manual_breed,
                vr.manual_measurements,
                vr.atc_score_difference,
                vr.breed_match,
                vr.measurement_accuracy,
                vr.validation_date,
                vr.validator_id
            FROM animal_analyses va
            LEFT JOIN validation_records vr ON va.id = vr.analysis_id
            WHERE vr.validation_date >= date('now', '-{} days')
            AND vr.manual_atc_score IS NOT NULL
            """.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} validation records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading validation data: {e}")
            return pd.DataFrame()
    
    def calculate_accuracy_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate accuracy metrics from validation data.
        
        Args:
            df: Validation data DataFrame
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            metrics = {}
            
            # ATC Score accuracy
            if 'ai_atc_score' in df.columns and 'manual_atc_score' in df.columns:
                ai_scores = df['ai_atc_score'].dropna()
                manual_scores = df['manual_atc_score'].dropna()
                
                if len(ai_scores) > 0 and len(manual_scores) > 0:
                    # Align data
                    common_idx = ai_scores.index.intersection(manual_scores.index)
                    ai_scores = ai_scores.loc[common_idx]
                    manual_scores = manual_scores.loc[common_idx]
                    
                    if len(ai_scores) > 0:
                        metrics['atc_mae'] = mean_absolute_error(manual_scores, ai_scores)
                        metrics['atc_mse'] = mean_squared_error(manual_scores, ai_scores)
                        metrics['atc_rmse'] = np.sqrt(metrics['atc_mse'])
                        metrics['atc_r2'] = r2_score(manual_scores, ai_scores)
                        metrics['atc_correlation'] = pearsonr(manual_scores, ai_scores)[0]
                        metrics['atc_accuracy'] = 1 - (metrics['atc_mae'] / manual_scores.mean())
            
            # Breed classification accuracy
            if 'ai_breed' in df.columns and 'manual_breed' in df.columns:
                breed_matches = df['breed_match'].dropna()
                if len(breed_matches) > 0:
                    metrics['breed_accuracy'] = breed_matches.mean()
            
            # Measurement accuracy
            if 'measurement_accuracy' in df.columns:
                measurement_acc = df['measurement_accuracy'].dropna()
                if len(measurement_acc) > 0:
                    metrics['measurement_accuracy'] = measurement_acc.mean()
            
            # Overall confidence
            if 'ai_confidence' in df.columns:
                confidences = df['ai_confidence'].dropna()
                if len(confidences) > 0:
                    metrics['average_confidence'] = confidences.mean()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            return {}
    
    def generate_accuracy_report(self, days: int = 30) -> Dict:
        """
        Generate comprehensive accuracy report.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Accuracy report dictionary
        """
        try:
            # Load data
            df = self.load_validation_data(days)
            
            if df.empty:
                return {
                    'error': 'No validation data found',
                    'total_validations': 0
                }
            
            # Calculate metrics
            metrics = self.calculate_accuracy_metrics(df)
            
            # Generate report
            report = {
                'total_validations': len(df),
                'analysis_period': {
                    'start': df['validation_date'].min(),
                    'end': df['validation_date'].max()
                },
                'accuracy_metrics': metrics,
                'recommendations': self._generate_recommendations(metrics),
                'detailed_analysis': self._detailed_analysis(df)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating accuracy report: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on accuracy metrics."""
        recommendations = []
        
        # ATC Score recommendations
        if 'atc_accuracy' in metrics:
            atc_acc = metrics['atc_accuracy']
            if atc_acc < 0.7:
                recommendations.append("ATC scoring accuracy is below 70%. Consider retraining the model.")
            elif atc_acc < 0.8:
                recommendations.append("ATC scoring accuracy could be improved. Fine-tune scoring parameters.")
        
        # Breed classification recommendations
        if 'breed_accuracy' in metrics:
            breed_acc = metrics['breed_accuracy']
            if breed_acc < 0.6:
                recommendations.append("Breed classification accuracy is below 60%. Collect more training data.")
            elif breed_acc < 0.8:
                recommendations.append("Breed classification could be improved. Consider data augmentation.")
        
        # Measurement recommendations
        if 'measurement_accuracy' in metrics:
            meas_acc = metrics['measurement_accuracy']
            if meas_acc < 0.8:
                recommendations.append("Measurement accuracy is below 80%. Improve keypoint detection.")
        
        # Correlation recommendations
        if 'atc_correlation' in metrics:
            corr = metrics['atc_correlation']
            if corr < 0.7:
                recommendations.append("Low correlation between AI and manual ATC scores. Review scoring criteria.")
        
        if not recommendations:
            recommendations.append("System performance is satisfactory across all metrics.")
        
        return recommendations
    
    def _detailed_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform detailed analysis of validation data."""
        try:
            analysis = {}
            
            # ATC Score distribution
            if 'ai_atc_score' in df.columns and 'manual_atc_score' in df.columns:
                analysis['atc_score_distribution'] = {
                    'ai_mean': df['ai_atc_score'].mean(),
                    'ai_std': df['ai_atc_score'].std(),
                    'manual_mean': df['manual_atc_score'].mean(),
                    'manual_std': df['manual_atc_score'].std()
                }
            
            # Breed analysis
            if 'ai_breed' in df.columns and 'manual_breed' in df.columns:
                breed_confusion = pd.crosstab(df['manual_breed'], df['ai_breed'], margins=True)
                analysis['breed_confusion_matrix'] = breed_confusion.to_dict()
            
            # Confidence analysis
            if 'ai_confidence' in df.columns:
                analysis['confidence_distribution'] = {
                    'mean': df['ai_confidence'].mean(),
                    'std': df['ai_confidence'].std(),
                    'min': df['ai_confidence'].min(),
                    'max': df['ai_confidence'].max()
                }
            
            # Temporal analysis
            if 'validation_date' in df.columns:
                df['validation_date'] = pd.to_datetime(df['validation_date'])
                daily_validations = df.groupby(df['validation_date'].dt.date).size()
                analysis['daily_validations'] = daily_validations.to_dict()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in detailed analysis: {e}")
            return {}
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "validation_plots"):
        """
        Create visualization plots for validation data.
        
        Args:
            df: Validation data DataFrame
            output_dir: Output directory for plots
        """
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            # ATC Score comparison
            if 'ai_atc_score' in df.columns and 'manual_atc_score' in df.columns:
                plt.figure(figsize=(10, 8))
                
                # Scatter plot
                plt.subplot(2, 2, 1)
                plt.scatter(df['manual_atc_score'], df['ai_atc_score'], alpha=0.6)
                plt.plot([0, 100], [0, 100], 'r--', label='Perfect correlation')
                plt.xlabel('Manual ATC Score')
                plt.ylabel('AI ATC Score')
                plt.title('AI vs Manual ATC Scores')
                plt.legend()
                
                # Residuals plot
                plt.subplot(2, 2, 2)
                residuals = df['ai_atc_score'] - df['manual_atc_score']
                plt.scatter(df['manual_atc_score'], residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Manual ATC Score')
                plt.ylabel('Residuals (AI - Manual)')
                plt.title('Residuals Plot')
                
                # Distribution comparison
                plt.subplot(2, 2, 3)
                plt.hist(df['manual_atc_score'], alpha=0.5, label='Manual', bins=20)
                plt.hist(df['ai_atc_score'], alpha=0.5, label='AI', bins=20)
                plt.xlabel('ATC Score')
                plt.ylabel('Frequency')
                plt.title('Score Distribution')
                plt.legend()
                
                # Error distribution
                plt.subplot(2, 2, 4)
                plt.hist(residuals, bins=20, alpha=0.7)
                plt.xlabel('Residuals')
                plt.ylabel('Frequency')
                plt.title('Error Distribution')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/atc_score_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Breed classification confusion matrix
            if 'ai_breed' in df.columns and 'manual_breed' in df.columns:
                plt.figure(figsize=(12, 8))
                breed_confusion = pd.crosstab(df['manual_breed'], df['ai_breed'])
                sns.heatmap(breed_confusion, annot=True, fmt='d', cmap='Blues')
                plt.title('Breed Classification Confusion Matrix')
                plt.xlabel('AI Prediction')
                plt.ylabel('Manual Label')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/breed_confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Confidence analysis
            if 'ai_confidence' in df.columns:
                plt.figure(figsize=(10, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(df['ai_confidence'], bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('AI Confidence')
                plt.ylabel('Frequency')
                plt.title('AI Confidence Distribution')
                
                plt.subplot(1, 2, 2)
                if 'manual_atc_score' in df.columns and 'ai_atc_score' in df.columns:
                    errors = np.abs(df['ai_atc_score'] - df['manual_atc_score'])
                    plt.scatter(df['ai_confidence'], errors, alpha=0.6)
                    plt.xlabel('AI Confidence')
                    plt.ylabel('Absolute Error')
                    plt.title('Confidence vs Error')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/confidence_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def export_report(self, report: Dict, output_file: str = "accuracy_report.json"):
        """
        Export accuracy report to file.
        
        Args:
            report: Accuracy report dictionary
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Accuracy report exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")

def main():
    """Main function for validation analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoATC Validation Analysis')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--db', type=str, default='atc.db', help='Database path')
    parser.add_argument('--output', type=str, default='validation_report.json', help='Output file')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--plot-dir', type=str, default='validation_plots', help='Plot output directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ValidationAnalyzer(args.db)
    
    # Generate report
    logger.info(f"Generating accuracy report for last {args.days} days...")
    report = analyzer.generate_accuracy_report(args.days)
    
    if 'error' in report:
        logger.error(f"Error generating report: {report['error']}")
        return
    
    # Export report
    analyzer.export_report(report, args.output)
    
    # Generate plots if requested
    if args.plots:
        logger.info("Generating visualization plots...")
        df = analyzer.load_validation_data(args.days)
        if not df.empty:
            analyzer.create_visualizations(df, args.plot_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("AUTOATC VALIDATION REPORT")
    print("="*50)
    print(f"Total Validations: {report['total_validations']}")
    print(f"Analysis Period: {report['analysis_period']['start']} to {report['analysis_period']['end']}")
    
    metrics = report['accuracy_metrics']
    if 'atc_accuracy' in metrics:
        print(f"ATC Score Accuracy: {metrics['atc_accuracy']:.2%}")
    if 'breed_accuracy' in metrics:
        print(f"Breed Classification Accuracy: {metrics['breed_accuracy']:.2%}")
    if 'measurement_accuracy' in metrics:
        print(f"Measurement Accuracy: {metrics['measurement_accuracy']:.2%}")
    if 'average_confidence' in metrics:
        print(f"Average AI Confidence: {metrics['average_confidence']:.2%}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"• {rec}")
    
    print(f"\nDetailed report saved to: {args.output}")
    if args.plots:
        print(f"Visualization plots saved to: {args.plot_dir}")

if __name__ == "__main__":
    main()
