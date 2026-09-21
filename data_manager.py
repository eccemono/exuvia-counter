"""
Data Manager: Handle Excel logging, statistics, and data export
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DataManager:
    """Manage exuvia count data and Excel export"""
    
    SCHEMA = [
        "Timestamp", "Batch", "Tray_ID", "Zoom", "Model",
        "Count", "Est_Range_Low", "Est_Range_High",
        "Mean_Confidence", "Confidence_Thresh", "IoU_Thresh",
        "Tile_Size", "Overlap_px", "Image_Path", "Notes",
    ]
    
    def __init__(self, log_file="data/exuvia_log.xlsx"):
        """
        Initialize data manager.
        
        Args:
            log_file: Path to Excel log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.df = self._load_or_create()
    
    def _load_or_create(self):
        """Load existing Excel or create new one"""
        try:
            if self.log_file.exists():
                df = pd.read_excel(self.log_file)
                # Back-fill any columns added after the log was first created
                for col in self.SCHEMA:
                    if col not in df.columns:
                        df[col] = ""
                df = df.reindex(columns=self.SCHEMA)
                logger.info(f"Loaded existing log: {self.log_file}")
                return df
        except Exception as e:
            logger.warning(f"Error loading existing log: {e}. Creating new.")
        
        df = pd.DataFrame(columns=self.SCHEMA)
        logger.info(f"Created new log file ready at {self.log_file}")
        return df
    
    def add_detection(self, tray_id, zoom, model, count, mean_confidence,
                     image_path, notes="", batch=1,
                     est_range_low=None, est_range_high=None,
                     confidence_thresh=None, iou_thresh=None,
                     tile_size=None, overlap=None):
        """
        Add a detection record.

        Args:
            tray_id: Tray identifier (e.g., "tray_001")
            zoom: Zoom level (1, 2, or 3)
            model: Model name
            count: Number of exuvia detected
            mean_confidence: Average confidence score
            image_path: Path to image file
            notes: Optional notes
            batch: Batch number (integer)
            est_range_low/high: Heuristic count range bounds
            confidence_thresh: Confidence threshold used during detection
            iou_thresh: IoU threshold used during detection
            tile_size: Tile size in pixels
            overlap: Tile overlap in pixels

        Returns:
            Updated dataframe
        """
        new_row = pd.DataFrame([{
            "Timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "Batch": int(batch),
            "Tray_ID": tray_id,
            "Zoom": f"{zoom}x",
            "Model": model,
            "Count": int(count),
            "Est_Range_Low": int(est_range_low) if est_range_low is not None else "",
            "Est_Range_High": int(est_range_high) if est_range_high is not None else "",
            "Mean_Confidence": f"{mean_confidence:.2f}",
            "Confidence_Thresh": f"{confidence_thresh:.2f}" if confidence_thresh is not None else "",
            "IoU_Thresh": f"{iou_thresh:.2f}" if iou_thresh is not None else "",
            "Tile_Size": int(tile_size) if tile_size is not None else "",
            "Overlap_px": int(overlap) if overlap is not None else "",
            "Image_Path": str(image_path),
            "Notes": notes,
        }])
        
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._save()
        
        logger.info(f"Added detection: {tray_id}, {count} exuvia")
        return self.df
    
    def _save(self):
        """Save dataframe to Excel"""
        try:
            self.df.to_excel(self.log_file, index=False)
            logger.info(f"Data saved to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def clear_all(self):
        """Delete all historical log records and overwrite the Excel file."""
        self.df = pd.DataFrame(columns=self.SCHEMA)
        self._save()
        logger.info("Full history log cleared.")

    def get_summary_stats(self):
        """
        Calculate summary statistics.
        
        Returns:
            dict with stats
        """
        if len(self.df) == 0:
            return {
                "total_records": 0,
                "total_exuvia": 0,
                "average_count": 0,
                "std_dev": 0,
                "min_count": 0,
                "max_count": 0,
                "unique_trays": 0
            }
        
        counts = pd.to_numeric(self.df["Count"], errors="coerce").dropna()
        
        return {
            "total_records": len(self.df),
            "total_exuvia": int(counts.sum()),
            "average_count": float(counts.mean()),
            "std_dev": float(counts.std()),
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "unique_trays": int(self.df["Tray_ID"].nunique())
        }
    
    def get_tray_summary(self, tray_id):
        """
        Get summary for specific tray.
        
        Args:
            tray_id: Tray identifier
        
        Returns:
            dict with tray stats
        """
        tray_df = self.df[self.df["Tray_ID"] == tray_id]
        
        if len(tray_df) == 0:
            return {"count": 0, "average": 0, "records": 0}
        
        counts = pd.to_numeric(tray_df["Count"], errors="coerce").dropna()
        
        return {
            "count": len(tray_df),
            "total_exuvia": int(counts.sum()),
            "average": float(counts.mean()),
            "std_dev": float(counts.std()),
            "min": int(counts.min()),
            "max": int(counts.max())
        }
    
    def detect_outliers(self, threshold=2.0):
        """
        Detect outlier counts using z-score.
        
        Args:
            threshold: Z-score threshold (default 2.0 = 95% confidence)
        
        Returns:
            DataFrame of outlier records
        """
        if len(self.df) < 2:
            return pd.DataFrame()
        
        counts = pd.to_numeric(self.df["Count"], errors="coerce")
        z_scores = np.abs(stats.zscore(counts.dropna()))
        
        outlier_indices = np.where(z_scores > threshold)[0]
        return self.df.iloc[outlier_indices]
    
    def export_by_tray(self, output_dir="data/exports"):
        """
        Export separate Excel files for each tray.
        
        Args:
            output_dir: Directory to save exports
        
        Returns:
            list of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = []
        
        try:
            for tray_id in self.df["Tray_ID"].unique():
                tray_df = self.df[self.df["Tray_ID"] == tray_id]
                filename = f"{tray_id}_export.xlsx"
                filepath = output_dir / filename
                
                tray_df.to_excel(filepath, index=False)
                exported.append(filepath)
                logger.info(f"Exported: {filepath}")
            
            return exported
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return []
    
    def get_dataframe(self):
        """Get current dataframe"""
        return self.df.copy()
    
    def delete_record(self, record_index):
        """Delete a record by index"""
        try:
            self.df.drop(record_index, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self._save()
            logger.info(f"Deleted record {record_index}")
        except Exception as e:
            logger.error(f"Error deleting record: {e}")


# Singleton for Streamlit
_manager_instance = None

def get_data_manager(log_file="data/exuvia_log.xlsx"):
    """Get or create data manager"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DataManager(log_file)
    return _manager_instance
