"""
Tiler module: Split images into tiles for training data
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ImageTiler:
    """Split images into overlapping tiles"""
    
    @staticmethod
    def create_tiles(image_source, tile_size=256, overlap=50):
        """
        Split image into tiles.
        
        Args:
            image_source: Path to image or numpy array
            tile_size: Size of each tile (square)
            overlap: Overlap pixels between adjacent tiles
        
        Returns:
            list of numpy arrays (tiles)
        """
        if isinstance(image_source, str) or isinstance(image_source, Path):
            img = cv2.imread(str(image_source))
            if img is None:
                logger.error(f"Cannot read image: {image_source}")
                return []
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_source
        
        try:
            h, w = img.shape[:2]
            tiles = []
            coords = []
            
            stride = tile_size - overlap
            
            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    tile = img[y:y+tile_size, x:x+tile_size]
                    if tile.shape == (tile_size, tile_size, 3):
                        tiles.append(tile)
                        coords.append((x, y))
            
            # Handle edge cases: tiles that don't fit exactly
            # Right edge
            if w % stride != 0:
                x = w - tile_size
                for y in range(0, h - tile_size + 1, stride):
                    tile = img[y:y+tile_size, x:w]
                    if tile.shape[1] == tile_size and tile.shape[0] == tile_size:
                        tiles.append(tile)
                        coords.append((x, y))
            
            # Bottom edge
            if h % stride != 0:
                y = h - tile_size
                for x in range(0, w - tile_size + 1, stride):
                    tile = img[y:h, x:x+tile_size]
                    if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                        tiles.append(tile)
                        coords.append((x, y))
            
            # Bottom-right corner
            if w % stride != 0 and h % stride != 0:
                x, y = w - tile_size, h - tile_size
                tile = img[y:h, x:w]
                if tile.shape == (tile_size, tile_size, 3):
                    tiles.append(tile)
                    coords.append((x, y))
            
            logger.info(f"Created {len(tiles)} tiles from image")
            return tiles, coords
        
        except Exception as e:
            logger.error(f"Error creating tiles: {e}")
            return [], []
    
    @staticmethod
    def save_tiles(tiles, tile_coords, output_dir, tray_id, image_id=None):
        """
        Save tiles to disk.
        
        Args:
            tiles: List of tile arrays
            tile_coords: List of (x, y) coordinates for each tile
            output_dir: Base output directory
            tray_id: Tray identifier
            image_id: Image identifier (timestamp)
        
        Returns:
            list of paths to saved tiles
        """
        output_dir = Path(output_dir) / tray_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if image_id is None:
            image_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_paths = []
        
        try:
            for i, (tile, (x, y)) in enumerate(zip(tiles, tile_coords)):
                filename = f"{image_id}_tile_{i:03d}_x{x}_y{y}.jpg"
                filepath = output_dir / filename
                
                tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(filepath), tile_bgr)
                
                if success:
                    saved_paths.append(filepath)
            
            logger.info(f"Saved {len(saved_paths)} tiles to {output_dir}")
            return saved_paths
        
        except Exception as e:
            logger.error(f"Error saving tiles: {e}")
            return []
    
    @staticmethod
    def tiles_from_file(image_path, tile_size=256, overlap=50, 
                        output_dir="data/tiles", tray_id="tray_001"):
        """
        Convenience function: load image, tile it, save tiles.
        
        Args:
            image_path: Path to input image
            tile_size: Tile size
            overlap: Overlap pixels
            output_dir: Base output directory
            tray_id: Tray identifier
        
        Returns:
            list of saved tile paths
        """
        tiles, coords = ImageTiler.create_tiles(image_path, tile_size=tile_size, overlap=overlap)
        image_id = Path(image_path).stem
        return ImageTiler.save_tiles(tiles, coords, output_dir, tray_id, image_id)


def load_tile_library(base_dir="data/tiles"):
    """
    Load all tiles organized by tray.
    
    Returns:
        dict: {tray_id: [list of tile paths]}
    """
    base_dir = Path(base_dir)
    library = {}
    
    if not base_dir.exists():
        return library
    
    try:
        for tray_dir in base_dir.iterdir():
            if tray_dir.is_dir():
                tray_id = tray_dir.name
                tiles = list(tray_dir.glob("*.jpg"))
                if tiles:
                    library[tray_id] = sorted(tiles)
        
        logger.info(f"Loaded tile library: {len(library)} trays")
        return library
    
    except Exception as e:
        logger.error(f"Error loading tile library: {e}")
        return {}
