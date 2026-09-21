"""
Example usage of Exuvia Counter modules
Shows how to use the system programmatically outside of Streamlit
"""

# ===== EXAMPLE 1: Capture an image and detect =====
def example_capture_and_detect():
    """Take one photo and run detection"""
    from camera import get_camera
    from detector import get_detector
    
    # Initialize camera
    cam = get_camera(use_pi_camera=True)
    
    # Capture image
    img_path = cam.capture_image(
        tray_id="tray_001",
        save_dir="data/captures",
        zoom=1
    )
    print(f"Captured: {img_path}")
    
    # Run detection
    detector = get_detector("yolov8n.pt")
    results = detector.detect(str(img_path), confidence=0.5)
    
    print(f"Found {results['count']} exuvia")
    print(f"Confidence: {results['mean_confidence']:.2f}")
    
    cam.close()


# ===== EXAMPLE 2: Batch process images =====
def example_batch_process():
    """Process all images in a folder"""
    from pathlib import Path
    from detector import get_detector
    from data_manager import get_data_manager
    
    detector = get_detector("yolov8n.pt")
    data_mgr = get_data_manager()
    
    # Process all JPGs in captures folder
    image_folder = Path("data/captures")
    for img_file in image_folder.glob("*.jpg"):
        results = detector.detect(str(img_file), confidence=0.5)
        
        if results:
            # Log to database
            data_mgr.add_detection(
                tray_id="tray_001",
                zoom=1,
                model="YOLOv8n",
                count=results['count'],
                mean_confidence=results['mean_confidence'],
                image_path=img_file
            )
            
            print(f"{img_file.name}: {results['count']} exuvia")


# ===== EXAMPLE 3: Create tiles from image =====
def example_create_tiles():
    """Split an image into training tiles"""
    from tiler import ImageTiler
    
    image_path = "data/captures/sample.jpg"
    
    tiles, coords = ImageTiler.create_tiles(
        image_path,
        tile_size=256,
        overlap=50
    )
    
    print(f"Created {len(tiles)} tiles")
    
    # Save them
    saved = ImageTiler.save_tiles(
        tiles,
        coords,
        output_dir="data/tiles",
        tray_id="tray_001",
        image_id="sample"
    )
    
    print(f"Saved {len(saved)} tiles")


# ===== EXAMPLE 4: Get statistics =====
def example_get_statistics():
    """Show all statistics"""
    from data_manager import get_data_manager
    
    data_mgr = get_data_manager()
    
    # Overall stats
    stats = data_mgr.get_summary_stats()
    print("Overall Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Total exuvia: {stats['total_exuvia']}")
    print(f"  Average: {stats['average_count']:.1f}")
    print(f"  Std Dev: {stats['std_dev']:.2f}")
    print()
    
    # Per-tray stats
    for tray in data_mgr.df["Tray_ID"].unique():
        tray_stats = data_mgr.get_tray_summary(tray)
        print(f"{tray}: {tray_stats['total_exuvia']} total, {tray_stats['average']:.1f} avg")
    
    # Find outliers
    outliers = data_mgr.detect_outliers(threshold=2.0)
    print(f"\nFound {len(outliers)} outliers")


# ===== EXAMPLE 5: Export data =====
def example_export_data():
    """Export data to Excel"""
    from data_manager import get_data_manager
    
    data_mgr = get_data_manager()
    
    # Export all data
    data_mgr._save()  # Auto-saves to data/exuvia_log.xlsx
    
    # Export by tray
    files = data_mgr.export_by_tray("data/exports")
    print(f"Exported {len(files)} files")


# ===== EXAMPLE 6: Manual detection on local image =====
def example_detect_local_image():
    """Detect objects in a local image file"""
    from detector import get_detector
    import cv2
    
    detector = get_detector("yolov8n.pt")
    
    # Path to any image on your system
    image_path = "/path/to/image.jpg"
    
    results = detector.detect_and_save(
        image_path,
        output_dir="data/detections",
        confidence=0.5
    )
    
    print(f"Count: {results['count']}")
    print(f"Saved annotated image to: {results.get('saved_path')}")


# ===== EXAMPLE 7: Use different camera =====
def example_usb_camera():
    """Use USB webcam instead of Pi camera"""
    from camera import get_camera
    
    # Force USB camera (no Pi camera)
    cam = get_camera(use_pi_camera=False)
    
    # Get frame
    frame = cam.get_frame(zoom=1)
    
    if frame is not None:
        print(f"Got frame: {frame.shape}")
    
    cam.close()


# ===== EXAMPLE 8: Train a simple loop =====
def example_training_loop():
    """Simulate continuous capture and detection"""
    from camera import get_camera
    from detector import get_detector
    from data_manager import get_data_manager
    import time
    
    cam = get_camera(use_pi_camera=True)
    detector = get_detector("yolov8n.pt")
    data_mgr = get_data_manager()
    
    # Capture 5 images
    for i in range(5):
        print(f"Capture {i+1}/5...")
        
        # Capture
        img_path = cam.capture_image(
            tray_id="tray_001",
            zoom=1
        )
        
        # Detect
        results = detector.detect(str(img_path), confidence=0.5)
        
        # Log
        data_mgr.add_detection(
            tray_id="tray_001",
            zoom=1,
            model="YOLOv8n",
            count=results['count'],
            mean_confidence=results['mean_confidence'],
            image_path=img_path
        )
        
        print(f"  Found {results['count']} exuvia")
        
        time.sleep(2)  # Wait 2 seconds between captures
    
    cam.close()
    print("Done! Check data/exuvia_log.xlsx")


# ===== RUN EXAMPLES =====
if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Capture and detect", example_capture_and_detect),
        "2": ("Batch process images", example_batch_process),
        "3": ("Create tiles", example_create_tiles),
        "4": ("Get statistics", example_get_statistics),
        "5": ("Export data", example_export_data),
        "6": ("Detect local image", example_detect_local_image),
        "7": ("Use USB camera", example_usb_camera),
        "8": ("Training loop", example_training_loop),
    }
    
    print("Exuvia Counter Examples")
    print("=====================")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    print()
    
    choice = input("Select example (1-8): ").strip()
    
    if choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
