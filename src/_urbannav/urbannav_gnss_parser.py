"""
UrbanNav GNSS Data Parser
Parses RINEX observation files to extract uncorrected GNSS positions
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import os


class UrbanNav_GNSSParser:
    """
    Parser for UrbanNav RINEX GNSS observation files.
    
    This class processes RINEX .obs files to extract uncorrected GNSS positions
    which can be used as measurements in the sensor fusion algorithm.
    The reference.csv contains RTK-corrected positions for ground truth.
    
    Methods:
    - parse_rinex_to_csv: Convert RINEX to CSV using external tools
    - get_gnss_positions: Extract latitude, longitude, altitude from RINEX
    """
    
    def __init__(self, obs_file_path: str, nav_file_path: str):
        """
        Initialize GNSS parser.
        
        Args:
            obs_file_path: Path to rover RINEX observation file (.obs)
            nav_file_path: Path to navigation file (.nav)
        """
        self.obs_file = Path(obs_file_path)
        self.nav_file = Path(nav_file_path)
        
        if not self.obs_file.exists():
            raise FileNotFoundError(f"Observation file not found: {obs_file_path}")
        if not self.nav_file.exists():
            raise FileNotFoundError(f"Navigation file not found: {nav_file_path}")
    
    def parse_using_georinex(self, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Parse RINEX files using georinex library (Python-based).
        
        This requires: pip install georinex
        
        Args:
            output_csv: Optional path to save parsed data as CSV
            
        Returns:
            DataFrame with GPS TOW, Latitude, Longitude, Altitude
        """
        try:
            import georinex as gr
        except ImportError:
            raise ImportError(
                "georinex library not installed. Install with: pip install georinex"
            )
        
        # Read observation and navigation files
        obs = gr.load(str(self.obs_file))
        nav = gr.load(str(self.nav_file))
        
        # TODO: Implement position computation from pseudoranges
        # This is a simplified placeholder - full implementation requires:
        # 1. Extract pseudorange measurements from obs
        # 2. Compute satellite positions from nav ephemeris
        # 3. Solve navigation equations for receiver position
        
        print("Warning: georinex parsing requires additional implementation")
        print("Consider using RTKLIB for automated position computation")
        
        return pd.DataFrame()
    
    def parse_using_rtklib(self, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Parse RINEX files using RTKLIB (external tool).
        
        RTKLIB provides rnx2rtkp command to compute positions from RINEX files.
        
        Installation:
        - macOS: brew install rtklib (if available) or download from GitHub
        - Manual: https://github.com/tomojitakasu/RTKLIB
        
        Args:
            output_dir: Directory to save output files
            
        Returns:
            DataFrame with GPS TOW, Latitude, Longitude, Altitude
        """
        # Check if rnx2rtkp is available
        try:
            result = subprocess.run(
                ['which', 'rnx2rtkp'], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                raise FileNotFoundError(
                    "rnx2rtkp not found. Please install RTKLIB:\n"
                    "GitHub: https://github.com/tomojitakasu/RTKLIB"
                )
        except Exception as e:
            print(f"Warning: Cannot check for rnx2rtkp: {e}")
            print("Install RTKLIB for automated GNSS processing")
            return pd.DataFrame()
        
        # Set output directory
        if output_dir is None:
            output_dir = self.obs_file.parent
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Output position file
        pos_file = output_path / f"{self.obs_file.stem}_gnss_pos.csv"
        
        # Run rnx2rtkp in single point positioning (SPP) mode
        cmd = [
            'rnx2rtkp',
            '-k', str(self.nav_file),  # Navigation file
            '-o', str(pos_file),        # Output file
            '-p', '0',                   # Mode: 0=Single, 1=DGPS, 2=Kinematic, etc.
            str(self.obs_file)          # Observation file
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running rnx2rtkp:\n{result.stderr}")
            return pd.DataFrame()
        
        # Parse RTKLIB position output (format: lat lon height)
        # RTKLIB output format varies, may need adjustment
        df = self._parse_rtklib_output(pos_file)
        
        return df
    
    def _parse_rtklib_output(self, pos_file: Path) -> pd.DataFrame:
        """Parse RTKLIB position output file."""
        # RTKLIB .pos file format (header starts with %)
        # Format: GPST latitude(deg) longitude(deg) height(m) Q ns sdn sde sdu
        
        data = []
        with open(pos_file, 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue  # Skip header
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                # Parse fields
                # Format: YYYY/MM/DD HH:MM:SS.SSS lat lon height ...
                date_time = f"{parts[0]} {parts[1]}"
                lat = float(parts[2])
                lon = float(parts[3])
                alt = float(parts[4])
                
                data.append({
                    'DateTime': date_time,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Altitude': alt
                })
        
        return pd.DataFrame(data)
    
    def get_recommendation(self) -> str:
        """Get recommendation for GNSS data usage."""
        return """
GNSS Data Usage Recommendation:

1. UNCORRECTED GNSS (for sensor fusion input):
   - Parse rover_ublox.obs or rover_trimble.obs files
   - Use RTKLIB (rnx2rtkp in SPP mode) or georinex
   - Typical accuracy: 2-10 meters
   - Frequency: 5 Hz (ublox) or 10 Hz (trimble)

2. RTK-CORRECTED GNSS (ground truth):
   - Use reference.csv directly
   - Contains Latitude, Longitude, Ellipsoid Height
   - High accuracy: cm-level (from Applanix POS LV620)
   - Frequency: 10 Hz

3. Recommended Workflow:
   a) Use reference.csv for ground truth comparison
   b) For sensor fusion input, either:
      - Process RINEX files to get uncorrected positions
      - OR use reference.csv but add artificial noise to simulate
        uncorrected GNSS measurements

4. Data Files:
   - rover_ublox.obs: Lower cost receiver (5 Hz)
   - rover_trimble.obs: Higher grade receiver (10 Hz)
   - base_trimble.obs + base.nav: Base station for RTK
   - reference.csv: RTK-corrected positions (ground truth)
"""


def create_gnss_csv_from_rinex(
    rover_obs: str,
    nav_file: str,
    output_csv: str,
    method: str = 'rtklib'
) -> bool:
    """
    Convenience function to create GNSS CSV from RINEX files.
    
    Args:
        rover_obs: Path to rover observation file
        nav_file: Path to navigation file
        output_csv: Path to output CSV file
        method: 'rtklib' or 'georinex'
        
    Returns:
        True if successful, False otherwise
    """
    parser = UrbanNav_GNSSParser(rover_obs, nav_file)
    
    if method == 'rtklib':
        df = parser.parse_using_rtklib(output_dir=Path(output_csv).parent)
    elif method == 'georinex':
        df = parser.parse_using_georinex(output_csv=output_csv)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rtklib' or 'georinex'")
    
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"GNSS positions saved to: {output_csv}")
        return True
    else:
        print("Failed to parse RINEX files")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        obs_file = sys.argv[1]
        nav_file = sys.argv[2] if len(sys.argv) > 2 else "base.nav"
        
        # parser = UrbanNav_GNSSParser(obs_file, nav_file)
        # print(parser.get_recommendation())
        create_gnss_csv_from_rinex(
            rover_obs=obs_file,
            nav_file=nav_file,
            output_csv="gnss_positions.csv",
            method='rtklib'
        )
    else:
        print("Usage: python urbannav_gnss_parser.py <rover.obs> <base.nav>")
        print("\nExample:")
        print("python urbannav_gnss_parser.py rover_ublox.obs base.nav")
