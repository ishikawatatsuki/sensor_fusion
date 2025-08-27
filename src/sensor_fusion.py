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
    FusionData,
    FilterConfig,
    HardwareConfig,
    SensorType,
    DatasetType,
    CoordinateFrame,
    Pose, State,
    SensorDataField,
    TimeUpdateField, MeasurementUpdateField,
    FusionResponse,
    InitialState,
    SensorData,
    VisualOdometryData,
    ControlInput
)
from .utils.noise_manager import NoiseManager
from .utils.time_reporter import time_reporter
from .utils.signal_processor import SignalProcessor
from .utils.data_logger import DataLogger, LoggingMessage
from .utils.geometric_transformer import GeometryTransformer, TransformationField



class SensorFusion:
    def __init__(
        self,
        filter_config: FilterConfig,
        hardware_config: HardwareConfig,
        data_logger: DataLogger = None,
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
        self.data_logger = data_logger

        self.filter_buffer_queue = PriorityQueue()
        
        self.last_vo_pose = None
        self.independent_vo_pose = None
        self.vo_relative_pose_inertial = None
        self.is_independent_vo_estimation = False

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

        if False:
            # Transform the state from the camera coordinate to the inertial coordinate system
            T_in_camera = np.eye(4)
            T_in_camera[:3, 3] = initial_state.x.p.flatten()
            T_in_camera[:3, :3] = State.get_rotation_matrix_from_quaternion_vector(initial_state.x.q.flatten())

            position_inertial = self.geo_transformer.transform(fields=TransformationField(
                state=self.kalman_filter.x,
                value=T_in_camera[:3, :],
                coord_from=CoordinateFrame.STEREO_LEFT,
                coord_to=CoordinateFrame.INERTIAL
            ))
            T_in_camera = np.eye(4)
            T_in_camera[:3, 3] = initial_state.x.v.flatten()
            velocity_inertial = self.geo_transformer.transform(fields=TransformationField(
                state=self.kalman_filter.x,
                value=T_in_camera[:3, :],
                coord_from=CoordinateFrame.STEREO_LEFT,
                coord_to=CoordinateFrame.INERTIAL
            ))
            q = State.get_quaternion_from_rotation_matrix(velocity_inertial[:3, :3])
            new_state = State(
                p=position_inertial[:3, 3].reshape(-1, 1),
                v=velocity_inertial[:3, 3].reshape(-1, 1),
                q=q,
                b_w=initial_state.x.b_w.reshape(-1, 1),
                b_a=initial_state.x.b_a.reshape(-1, 1)
            )

            self.kalman_filter.x = new_state
            self.kalman_filter.P = initial_state.P
            self.preprocessor.set_initial_angle(q=initial_state.x.q.flatten())
            vo_pose = Pose(
                R=new_state.get_rotation_matrix(),
                t=new_state.p.flatten()
            )
        else:
            self.kalman_filter.x = initial_state.x
            self.kalman_filter.P = initial_state.P
            self.preprocessor.set_initial_angle(q=initial_state.x.q.flatten())
            vo_pose = Pose(
                R=initial_state.x.get_rotation_matrix(),
                t=initial_state.x.p.flatten()
            )

        self.last_vo_pose = vo_pose
        self.independent_vo_pose = vo_pose

    def _log_data(self, sensor_type: SensorType, data: np.ndarray, timestamp: float):
        if self.data_logger is None:
            return
        
        self.data_logger.log(
            message=LoggingMessage(
                sensor_type=sensor_type,
                timestamp=timestamp,
                data=data
            ),
            is_raw=False
        )

    def _prepare_response(self, sensor_data: SensorDataField, visualizing_data: np.ndarray) -> FusionResponse:
        response = FusionResponse()
        
        if SensorType.is_vo_data(sensor_data.type):
            response.vo_data = visualizing_data.flatten()[:3]
        elif SensorType.is_gps_data(sensor_data.type):
            response.gps_data = visualizing_data.flatten()[:3]
            
        return response
    
    def _get_vo_measurement_data(self, sensor_data: SensorDataField) -> Tuple[MeasurementUpdateField, FusionResponse]:
        """Construct a measurement update data for the Kalman Filter"""
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
        self.vo_relative_pose_inertial = relative_pose
        self.independent_vo_pose = self.independent_vo_pose * relative_pose  # update vo pose estimation in inertial frame
        last_pose = self.last_vo_pose * relative_pose

        # Velocity to fuse
        dt = sensor_data.data.dt if sensor_data.data.dt > 1e-6 else 0.1

        # position = self.independent_vo_pose.t.reshape(-1, 1)  # NOTE: VO position independent on the fusion system
        position = last_pose.t.reshape(-1, 1)  # NOTE: Fused VO position
        velocity = relative_pose.t.reshape(-1, 1) / dt # velocity
        velocity *= np.array([1., 0., 0.]).reshape(-1, 1)  # only x velocity is used for KITTI dataset
        quaternion = State.get_quaternion_from_rotation_matrix(last_pose.R).reshape(-1, 1)

        _logging_data = VisualOdometryData(
            dt=dt,
            relative_pose=relative_pose_inertial[:3, :],
            timestamp=sensor_data.timestamp,
        )

        return position, velocity, quaternion, _logging_data


    def _get_measurement_update_data(self, sensor_data: SensorDataField) -> Tuple[MeasurementUpdateField, FusionResponse]:
        """Construct a measurement update data for the Kalman Filter"""

        visualizing_data = None
        fusion_fields = self.filter_config.sensors.get(sensor_data.type, [])
        if len(fusion_fields) == 0:
            logging.warning(f"Fusion field is not set for sensor: {sensor_data.type.name}")

        if SensorType.is_vo_data(sensor_data.type):
            position, velocity, quaternion, _logging_data = self._get_vo_measurement_data(sensor_data=sensor_data)

            z = np.empty((0, 1))
            if FusionData.POSITION in fusion_fields:
                z = np.vstack([z, position])
            if FusionData.LINEAR_VELOCITY in fusion_fields:
                z = np.vstack([z, velocity])
            if FusionData.ORIENTATION in fusion_fields:
                z = np.vstack([z, quaternion])
                
            visualizing_data = self.independent_vo_pose.t.flatten()
            
        elif SensorType.is_constraint_data(sensor_data.type):
            # lateral and vertical velocity constraints
            z = sensor_data.data.z.reshape(-1, 1)
            visualizing_data = z

            _logging_data = SensorData(z=z.flatten())
        
        else: # Add more data handling
            z = self.geo_transformer.transform(fields=TransformationField(
                state=self.kalman_filter.x,
                value=sensor_data.data.z,
                coord_from=sensor_data.coordinate_frame,
                coord_to=CoordinateFrame.INERTIAL))
            visualizing_data = z

            _logging_data = SensorData(z=z.flatten())

        R = self.noise_manager.get_measurement_noise(sensor_data=sensor_data)
        R = R[:z.shape[0], :z.shape[0]]

        response = self._prepare_response(sensor_data=sensor_data, visualizing_data=visualizing_data)

        self._log_data(sensor_type=sensor_data.type, data=_logging_data, timestamp=sensor_data.timestamp)

        return MeasurementUpdateField(z=z, R=R, sensor_type=sensor_data.type), response
    
    def _store_current_vo_estimate(self, sensor_data: SensorDataField):
        """After fusing VO estimate, store current state estimate in camera coordinate"""
        if not SensorType.is_vo_data(sensor_data.type):
            return

        t = self.kalman_filter.x.p.flatten()
        R = State.get_rotation_matrix_from_quaternion_vector(
            self.kalman_filter.x.q.flatten())
        pose_in_inertial = Pose(R=R, t=t)
        # state_in_camera = self.geo_transformer.transform(
        #     fields=TransformationField(
        #         state=self.kalman_filter.x,
        #         value=np.hstack([
        #             state_in_inertial[:3], state_in_inertial[6:10]
        #         ]),  # position, rotation
        #         coord_from=CoordinateFrame.INERTIAL,
        #         coord_to=CoordinateFrame.STEREO_LEFT)).flatten()
        # t = state_in_camera[:3]
        # R = State.get_rotation_matrix_from_quaternion_vector(
        #     state_in_camera[3:])
        self.last_vo_pose = pose_in_inertial * self.vo_relative_pose_inertial
        return
        
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


        _logging_data = ControlInput(
            dt=data.dt,
            u=data.u.flatten()
        )
        self._log_data(sensor_type=sensor_data.type, data=_logging_data, timestamp=sensor_data.timestamp)
        
        return FusionResponse(
                timestamp=sensor_data.timestamp,
                imu_acceleration=u[:3].flatten(),
                imu_angular_velocity=u[3:].flatten(),
            )
    
    @time_reporter
    def run_measurement_update(self, sensor_data: SensorDataField) -> FusionResponse:
        """Run measurement update step of the Kalman Filter"""
        logging.debug(
            f"Running measurement update for sensor: {sensor_data.type.name}")
        
        data, response = self._get_measurement_update_data(sensor_data=sensor_data)

        logging.info(f"Fusing data from: {sensor_data.type.name}, z: {data.z.flatten()} at time: {sensor_data.timestamp}")
        self.kalman_filter.measurement_update(data)
        
        # # NOTE: Store current state corrected by VO
        self._store_current_vo_estimate(sensor_data)

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
        
    def get_current_estimate(self, timestamp: float) -> FusionResponse:
        response = FusionResponse()
        response.pose = self.kalman_filter.get_current_estimate().matrix()
        response.estimated_angle = self.kalman_filter.x.get_euler_angle_from_quaternion().flatten()
        response.estimated_linear_velocity = self.kalman_filter.x.v.flatten()
        response.timestamp = timestamp
        return response