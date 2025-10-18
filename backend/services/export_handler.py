"""
Export Handler Service
Handles data export to BPA and other external systems
"""

import httpx
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from sqlalchemy.orm import Session

from db.database import SessionLocal
from db.models import AnimalAnalysis, BPAExport
from models.schemas import BPAExportData, BPAAimalData, ExportStatus
from config.settings import get_settings

logger = logging.getLogger(__name__)

class ExportHandler:
    """Handles data export to external systems like BPA."""
    
    def __init__(self):
        """Initialize the export handler."""
        self.settings = get_settings()
        self.bpa_api_url = self.settings.BPA_API_URL
        self.bpa_api_key = self.settings.BPA_API_KEY
        self.bpa_timeout = self.settings.BPA_TIMEOUT
        
        logger.info("Export handler initialized")
    
    async def export_to_bpa_async(self, export_id: int, bpa_data: BPAExportData):
        """
        Export data to BPA asynchronously.
        
        Args:
            export_id: ID of the export record
            bpa_data: Data to export to BPA
        """
        try:
            db = SessionLocal()
            try:
                # Get export record
                export_record = db.query(BPAExport).filter(BPAExport.id == export_id).first()
                if not export_record:
                    logger.error(f"Export record {export_id} not found")
                    return
                
                # Update status to processing
                export_record.status = ExportStatus.PENDING
                db.commit()
                
                # Prepare BPA API request
                api_data = self._prepare_bpa_api_data(bpa_data)
                
                # Send to BPA API
                success, response_data, error_message = await self._send_to_bpa_api(api_data)
                
                if success:
                    # Update export record with success
                    export_record.status = ExportStatus.SUCCESS
                    export_record.bpa_animal_id = response_data.get('animal_id')
                    export_record.exported_data = response_data
                    export_record.error_message = None
                    
                    # Update analysis record
                    analysis = db.query(AnimalAnalysis).filter(
                        AnimalAnalysis.id == export_record.analysis_id
                    ).first()
                    if analysis:
                        analysis.is_exported_to_bpa = True
                    
                    logger.info(f"Successfully exported to BPA: {export_id}")
                else:
                    # Update export record with failure
                    export_record.status = ExportStatus.FAILED
                    export_record.error_message = error_message
                    export_record.retry_count += 1
                    export_record.last_retry = datetime.now()
                    
                    logger.error(f"Failed to export to BPA: {export_id}, Error: {error_message}")
                
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in BPA export: {e}")
            # Update export record with error
            try:
                db = SessionLocal()
                export_record = db.query(BPAExport).filter(BPAExport.id == export_id).first()
                if export_record:
                    export_record.status = ExportStatus.FAILED
                    export_record.error_message = str(e)
                    export_record.retry_count += 1
                    export_record.last_retry = datetime.now()
                    db.commit()
                db.close()
            except:
                pass
    
    async def export_batch_to_bpa_async(self, export_ids: List[int], batch_data: BPAExportData):
        """
        Export batch data to BPA asynchronously.
        
        Args:
            export_ids: List of export record IDs
            batch_data: Batch data to export to BPA
        """
        try:
            db = SessionLocal()
            try:
                # Get export records
                export_records = db.query(BPAExport).filter(BPAExport.id.in_(export_ids)).all()
                
                # Update all records to processing
                for record in export_records:
                    record.status = ExportStatus.PENDING
                db.commit()
                
                # Prepare batch API request
                api_data = self._prepare_bpa_batch_api_data(batch_data)
                
                # Send to BPA API
                success, response_data, error_message = await self._send_to_bpa_batch_api(api_data)
                
                if success:
                    # Update all export records with success
                    for i, record in enumerate(export_records):
                        record.status = ExportStatus.SUCCESS
                        record.bpa_animal_id = response_data.get('animal_ids', [{}])[i].get('animal_id')
                        record.exported_data = response_data
                        record.error_message = None
                        
                        # Update analysis record
                        analysis = db.query(AnimalAnalysis).filter(
                            AnimalAnalysis.id == record.analysis_id
                        ).first()
                        if analysis:
                            analysis.is_exported_to_bpa = True
                    
                    logger.info(f"Successfully exported batch to BPA: {len(export_ids)} records")
                else:
                    # Update all export records with failure
                    for record in export_records:
                        record.status = ExportStatus.FAILED
                        record.error_message = error_message
                        record.retry_count += 1
                        record.last_retry = datetime.now()
                    
                    logger.error(f"Failed to export batch to BPA: {error_message}")
                
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error in batch BPA export: {e}")
            # Update export records with error
            try:
                db = SessionLocal()
                export_records = db.query(BPAExport).filter(BPAExport.id.in_(export_ids)).all()
                for record in export_records:
                    record.status = ExportStatus.FAILED
                    record.error_message = str(e)
                    record.retry_count += 1
                    record.last_retry = datetime.now()
                db.commit()
                db.close()
            except:
                pass
    
    async def retry_export_async(self, export_id: int):
        """
        Retry a failed export.
        
        Args:
            export_id: ID of the export record to retry
        """
        try:
            db = SessionLocal()
            try:
                export_record = db.query(BPAExport).filter(BPAExport.id == export_id).first()
                if not export_record:
                    logger.error(f"Export record {export_id} not found")
                    return
                
                # Check retry limit
                max_retries = 3
                if export_record.retry_count >= max_retries:
                    logger.warning(f"Export {export_id} has exceeded retry limit")
                    return
                
                # Get original data
                bpa_data = BPAExportData(**export_record.exported_data)
                
                # Retry export
                await self.export_to_bpa_async(export_id, bpa_data)
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error retrying export {export_id}: {e}")
    
    def _prepare_bpa_api_data(self, bpa_data: BPAExportData) -> Dict[str, Any]:
        """Prepare data for BPA API."""
        try:
            api_data = {
                'batch_id': bpa_data.export_batch_id,
                'export_date': bpa_data.export_date.isoformat(),
                'total_animals': bpa_data.total_animals,
                'animals': []
            }
            
            for animal in bpa_data.animals:
                animal_data = {
                    'animal_id': animal.animal_id,
                    'breed': animal.breed,
                    'atc_score': animal.atc_score,
                    'measurements': animal.measurements,
                    'analysis_date': animal.analysis_date.isoformat(),
                    'image_url': animal.image_url,
                    'metadata': animal.metadata
                }
                api_data['animals'].append(animal_data)
            
            return api_data
            
        except Exception as e:
            logger.error(f"Error preparing BPA API data: {e}")
            return {}
    
    def _prepare_bpa_batch_api_data(self, batch_data: BPAExportData) -> Dict[str, Any]:
        """Prepare batch data for BPA API."""
        try:
            api_data = {
                'batch_id': batch_data.export_batch_id,
                'export_date': batch_data.export_date.isoformat(),
                'total_animals': batch_data.total_animals,
                'animals': []
            }
            
            for animal in batch_data.animals:
                animal_data = {
                    'animal_id': animal.animal_id,
                    'breed': animal.breed,
                    'atc_score': animal.atc_score,
                    'measurements': animal.measurements,
                    'analysis_date': animal.analysis_date.isoformat(),
                    'image_url': animal.image_url,
                    'metadata': animal.metadata
                }
                api_data['animals'].append(animal_data)
            
            return api_data
            
        except Exception as e:
            logger.error(f"Error preparing BPA batch API data: {e}")
            return {}
    
    async def _send_to_bpa_api(self, api_data: Dict[str, Any]) -> Tuple[bool, Dict, str]:
        """
        Send data to BPA API.
        
        Args:
            api_data: Data to send to BPA API
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.bpa_api_key}' if self.bpa_api_key else ''
            }
            
            async with httpx.AsyncClient(timeout=self.bpa_timeout) as client:
                response = await client.post(
                    f"{self.bpa_api_url}/animals/import",
                    json=api_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return True, response_data, ""
                else:
                    error_message = f"BPA API error: {response.status_code} - {response.text}"
                    return False, {}, error_message
                    
        except httpx.TimeoutException:
            error_message = "BPA API timeout"
            return False, {}, error_message
        except httpx.RequestError as e:
            error_message = f"BPA API request error: {str(e)}"
            return False, {}, error_message
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            return False, {}, error_message
    
    async def _send_to_bpa_batch_api(self, api_data: Dict[str, Any]) -> Tuple[bool, Dict, str]:
        """
        Send batch data to BPA API.
        
        Args:
            api_data: Batch data to send to BPA API
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.bpa_api_key}' if self.bpa_api_key else ''
            }
            
            async with httpx.AsyncClient(timeout=self.bpa_timeout) as client:
                response = await client.post(
                    f"{self.bpa_api_url}/animals/batch-import",
                    json=api_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return True, response_data, ""
                else:
                    error_message = f"BPA API error: {response.status_code} - {response.text}"
                    return False, {}, error_message
                    
        except httpx.TimeoutException:
            error_message = "BPA API timeout"
            return False, {}, error_message
        except httpx.RequestError as e:
            error_message = f"BPA API request error: {str(e)}"
            return False, {}, error_message
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            return False, {}, error_message
    
    def validate_export_data(self, bpa_data: BPAExportData) -> Dict[str, Any]:
        """
        Validate data before export.
        
        Args:
            bpa_data: Data to validate
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'missing_fields': []
            }
            
            # Check required fields
            required_fields = ['export_batch_id', 'export_date', 'total_animals', 'animals']
            for field in required_fields:
                if not hasattr(bpa_data, field) or getattr(bpa_data, field) is None:
                    validation['missing_fields'].append(field)
                    validation['valid'] = False
            
            # Validate animals data
            if hasattr(bpa_data, 'animals') and bpa_data.animals:
                for i, animal in enumerate(bpa_data.animals):
                    animal_required_fields = ['animal_id', 'breed', 'atc_score', 'measurements']
                    for field in animal_required_fields:
                        if not hasattr(animal, field) or getattr(animal, field) is None:
                            validation['errors'].append(f"Animal {i}: Missing {field}")
                            validation['valid'] = False
                    
                    # Validate ATC score
                    if hasattr(animal, 'atc_score') and animal.atc_score is not None:
                        if not (0 <= animal.atc_score <= 100):
                            validation['warnings'].append(f"Animal {i}: ATC score out of range")
            
            # Check data consistency
            if hasattr(bpa_data, 'total_animals') and hasattr(bpa_data, 'animals'):
                if bpa_data.total_animals != len(bpa_data.animals):
                    validation['warnings'].append("Total animals count doesn't match actual animals")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating export data: {e}")
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}'],
                'warnings': [],
                'missing_fields': []
            }
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        try:
            db = SessionLocal()
            try:
                # Get export counts by status
                total_exports = db.query(BPAExport).count()
                successful_exports = db.query(BPAExport).filter(BPAExport.status == ExportStatus.SUCCESS).count()
                failed_exports = db.query(BPAExport).filter(BPAExport.status == ExportStatus.FAILED).count()
                pending_exports = db.query(BPAExport).filter(BPAExport.status == ExportStatus.PENDING).count()
                
                # Get retry statistics
                retry_counts = db.query(BPAExport.retry_count).all()
                avg_retries = sum(r[0] for r in retry_counts) / len(retry_counts) if retry_counts else 0
                
                return {
                    'total_exports': total_exports,
                    'successful_exports': successful_exports,
                    'failed_exports': failed_exports,
                    'pending_exports': pending_exports,
                    'success_rate': (successful_exports / total_exports * 100) if total_exports > 0 else 0,
                    'average_retries': round(avg_retries, 2)
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting export statistics: {e}")
            return {
                'total_exports': 0,
                'successful_exports': 0,
                'failed_exports': 0,
                'pending_exports': 0,
                'success_rate': 0,
                'average_retries': 0
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get export handler service status."""
        return {
            'service': 'export_handler',
            'status': 'operational',
            'bpa_api_url': self.bpa_api_url,
            'bpa_api_configured': bool(self.bpa_api_key),
            'timeout': self.bpa_timeout,
            'capabilities': {
                'single_export': True,
                'batch_export': True,
                'retry_failed': True,
                'data_validation': True
            }
        }
