import logging
import numpy as np
from typing import Tuple
from queue import PriorityQueue

from .kalman_filters import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
    EnsembleKalmanFilter,
    CubatureKalmanFilter,
)
from .common import (
    FilterConfig,
    HardwareConfig,
    SensorType,
    DatasetType,
    CoordinateFrame,
    Pose, State,
    SensorDataField,
    TimeUpdateField, MeasurementUpdateField,
    FusionResponse,
    InitialState
)
from .utils.noise_manager import NoiseManager
from .utils.time_reporter import time_reporter
from .utils.signal_processor import SignalProcessor
from .utils.geometric_transformer import GeometryTransformer, TransformationField



class SensorFusion:
    def __init__(
        self,
        filter_config: FilterConfig,
        hardware_config: HardwareConfig,
    ):
        
        self.filter_config = filter_config
        self.hardware_config = hardware_config

        # NOTE: Initialize Kalman Filter
        logging.debug("Configuring Kalman Filter")
        self.kalman_filter = self._get_kalman_filter()
    
        # NOTE: Initialize Sensor Noise Manager
        logging.debug("Configuring Noise Manager")
        self.noise_manager = NoiseManager(filter_config=filter_config, hardware_config=hardware_config)

        # NOTE: Initialize Geometry Transformer
        logging.debug("Configuring Geometry Transformer")
        self.geo_transformer = GeometryTransformer(
            hardware_config=hardware_config
        )

        # NOTE: Initialize IMU Signal processor
        logging.debug("Configuring IMU Signal Processor")
        self.preprocessor = SignalProcessor(
            filter_config=filter_config,
            hardware_config=hardware_config
        )

        self.filter_buffer_queue = PriorityQueue()
        
        self.last_vo_pose = None
        self.last_vo_timestamp = None
        self.initial_timestamp = None

    def _get_kalman_filter(self):
        filter_type = str(self.filter_config.type).lower()
        coordinate_system = DatasetType.get_coordinate_system(self.hardware_config.type)
        initial_state = State.get_initial_state_from_config(filter_config=self.filter_config)
        
        kwargs = {
            'config': self.filter_config,
            'hardware_config': self.hardware_config,
            'x': initial_state,
            'P': np.eye(initial_state.get_vector_size()) * 0.1,
            'coordinate_system': coordinate_system,
        }
        
        match (filter_type):
            case "ekf":
                logging.info(f"Configuring Extended Kalman Filter.")
                return ExtendedKalmanFilter(**kwargs)
            case "ukf":
                logging.info(f"Configuring Unscented Kalman Filter.")
                return UnscentedKalmanFilter(**kwargs)
            case "pf":
                logging.info(f"Configuring Particle Filter.")
                return ParticleFilter(**kwargs)
            case "enkf":
                logging.info(f"Configuring Ensemble Kalman Filter.")
                return EnsembleKalmanFilter(**kwargs)
            case "ckf":  
                logging.info(f"Configuring Cubature Kalman Filter.")
                return CubatureKalmanFilter(**kwargs)
            case _:
                # NOTE: Set EKF as a default filter
                logging.warning(f"dataset: {filter_type} is not found. EKF is used instead.")
                return ExtendedKalmanFilter(**kwargs)


    def set_initial_state(self, initial_state: InitialState):
        """Set initial state for the Kalman Filter"""
        self.kalman_filter.x = initial_state.x
        self.kalman_filter.P = initial_state.P
        self.preprocessor.set_initial_angle(q=initial_state.x.q.flatten())

        self.last_vo_pose = Pose(
            R=initial_state.x.get_rotation_matrix(),
            t=initial_state.x.p.flatten()
        )

    def _prepare_response(self, sensor_data: SensorDataField, visualizing_data: np.ndarray) -> FusionResponse:
        response = FusionResponse()
        
        if SensorType.is_vo_data(sensor_data.type):
            response.vo_data = visualizing_data.flatten()[:3]
        elif SensorType.is_gps_data(sensor_data.type):
            response.gps_data = visualizing_data.flatten()[:3]
            
        return response

    def _get_measurement_update_data(self, sensor_data: SensorDataField) -> Tuple[MeasurementUpdateField, FusionResponse]:
        """Construct a measurement update data for the Kalman Filter"""

        visualizing_data = None

        if SensorType.is_vo_data(sensor_data.type):
            # NOTE: Construct a VO estimated position from the relative pose between image frames at t1 and t2
            relative_pose_in_camera_coord = sensor_data.data.relative_pose
            if relative_pose_in_camera_coord.shape != (3, 4):
                relative_pose_in_camera_coord = relative_pose_in_camera_coord.reshape(3, 4)

            relative_pose_inertial = self.geo_transformer.transform(fields=TransformationField(
                state=self.kalman_filter.x,
                value=relative_pose_in_camera_coord,
                coord_from=sensor_data.coordinate_frame,
                coord_to=CoordinateFrame.INERTIAL))
            
            relative_pose = Pose(R=relative_pose_inertial[:3, :3], t=relative_pose_inertial[:3, 3])
            self.last_vo_pose = self.last_vo_pose * relative_pose  # update vo pose estimation in inertial frame

            # TODO: Enable to select which data in vo estimate to fuse
            # Velocity to fuse
            dt = sensor_data.data.dt if sensor_data.data.dt > 1e-6 else 0.1
            z = relative_pose.t.reshape(-1, 1) / dt # velocity

            visualizing_data = self.last_vo_pose.t.flatten()
        else:
            z = self.geo_transformer.transform(fields=TransformationField(
                state=self.kalman_filter.x,
                value=sensor_data.data.z,
                coord_from=sensor_data.coordinate_frame,
                coord_to=CoordinateFrame.INERTIAL))
            visualizing_data = z

        R = self.noise_manager.get_measurement_noise(sensor_data=sensor_data)
        R = R[:z.shape[0], :z.shape[0]]

        response = self._prepare_response(sensor_data=sensor_data, visualizing_data=visualizing_data)

        return MeasurementUpdateField(z=z, R=R, sensor_type=sensor_data.type), response
    
    def _store_current_vo_estimate(self, sensor_data: SensorDataField):
        """After fusing VO estimate, store current state estimate in camera coordinate"""
        if not SensorType.is_vo_data(sensor_data.type):
            return

        state_in_inertial = self.kalman_filter.x.get_state_vector().flatten()
        state_in_camera = self.geo_transformer.transform(
            fields=TransformationField(
                state=self.kalman_filter.x,
                value=np.hstack([
                    state_in_inertial[:3], state_in_inertial[6:10]
                ]),  # position, rotation
                coord_from=CoordinateFrame.INERTIAL,
                coord_to=CoordinateFrame.STEREO_LEFT)).flatten()
        t = state_in_camera[:3]
        R = State.get_rotation_matrix_from_quaternion_vector(
            state_in_camera[3:])
        self.last_vo_pose = Pose(R=R, t=t)
        
    @time_reporter
    def run_time_update(self, sensor_data: SensorDataField) -> FusionResponse:
        """Run time update step of the Kalman Filter"""
        logging.debug(
            f"Running time update for sensor: {sensor_data.type.name}")
        # NOTE: Apply IMU preprocessing if needed
        value = self.preprocessor.get_control_input(sensor_data=sensor_data)
        # NOTE: Transform Control input if needed
        u = self.geo_transformer.transform(fields=TransformationField(
                state=self.kalman_filter.x,
                value=value,
                coord_from=sensor_data.coordinate_frame,
                coord_to=CoordinateFrame.INERTIAL)).flatten()
        
        # NOTE: Get process noise covariance matrix Q
        Q = self.noise_manager.get_process_noise(sensor_data=sensor_data)

        data = TimeUpdateField(u=u, dt=sensor_data.data.dt, Q=Q)

        self.kalman_filter.time_update(data)
        
        return FusionResponse(
                # pose=self.kalman_filter.get_current_estimate().matrix(), 
                timestamp=sensor_data.timestamp,
                imu_acceleration=u[:3].flatten(),
                imu_angular_velocity=u[3:].flatten(),
                # estimated_angle=self.kalman_filter.x.get_euler_angle_from_quaternion().flatten(),
                # estimated_linear_velocity=self.kalman_filter.x.v.flatten(),
            )
    
    @time_reporter
    def run_measurement_update(self, sensor_data: SensorDataField) -> FusionResponse:
        """Run measurement update step of the Kalman Filter"""
        logging.debug(
            f"Running measurement update for sensor: {sensor_data.type.name}")
        
        data, response = self._get_measurement_update_data(sensor_data=sensor_data)

        self.kalman_filter.measurement_update(data)
        
        # # NOTE: Store current state corrected by VO
        # self._store_current_vo_estimate(sensor_data)

        response.pose = self.kalman_filter.get_current_estimate().matrix()
        response.estimated_angle = self.kalman_filter.x.get_euler_angle_from_quaternion().flatten()
        response.estimated_linear_velocity = self.kalman_filter.x.v.flatten()
        response.timestamp = sensor_data.timestamp
        return response

    def compute(self, sensor_data: SensorDataField) -> FusionResponse:
        """Invoke kalaman filter step based on given sensor type"""
        # NOTE: Construct sensor data field based on the message
        # sensor_data = self.data_adapter.convert(sensor_data)
        # if sensor_data is None:
        #     return None

        # NOTE: Invoke corresponding step of the Kalman Filter
        if SensorType.is_time_update(sensor_data.type):
            response, _ = self.run_time_update(sensor_data)
            
            return response
        elif SensorType.is_measurement_update(sensor_data.type):
            response, _ = self.run_measurement_update(sensor_data)
            
            return response