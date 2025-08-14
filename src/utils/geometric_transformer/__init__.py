import logging
import numpy as np
from .base_geometric_transformer import TransformationField
from .kitti_geometric_transformer import KITTI_GeometricTransformer
from .euroc_geometric_transformer import EuRoC_GeometricTransformer
from ...common import HardwareConfig, DatasetType, SensorDataField


class GeometryTransformer:
    
    def __init__(self, hardware_config: HardwareConfig):
        self.config = hardware_config
        
        self._transformer = self._get_transformer()
    
    def _get_transformer(self):
        dataset_type = DatasetType.get_type_from_str(self.config.type)
        kwargs = {
            'hardware_config': self.config,
        }
        match (dataset_type):
            case DatasetType.KITTI:
                logging.info(f"Setting KITTI geometric transformer.")
                return KITTI_GeometricTransformer(**kwargs)
            case DatasetType.EUROC:
                logging.info(f"Setting EuRoC geometric transformer.")
                return EuRoC_GeometricTransformer(**kwargs)
            case _:
                logging.error(f"Unsupported dataset type: {dataset_type}. Using KITTI transformer as default.")
                return KITTI_GeometricTransformer(**kwargs)
            
    def transform(self, fields: TransformationField) -> np.ndarray:
        if fields.coord_from == fields.coord_to:
            logging.warning("Same transformation coordinate frame. The data is not transformed.")
            return np.array(fields.value).reshape(-1, 1)
        
        return self._transformer.transform_data(fields)

    def transfrom_imu(self, data: SensorDataField) -> np.ndarray:
        """Transform IMU data into virtual IMU coordinate frame."""

        return self._transformer._transform_virtual_imu_data(data).reshape(-1, 1)
