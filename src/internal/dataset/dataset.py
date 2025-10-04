import os
import sys
import yaml
import time
import logging
import numpy as np
from queue import PriorityQueue
from collections import namedtuple
from dataclasses import dataclass
from threading import Thread
from typing import List

from .kitti import (
    OXTS_GPSDataReader,
    OXTS_IMUDataReader,
    KITTI_StereoFrameReader,
    KITTI_GroundTruthDataReader,
    KITTI_UpwardLeftwardVelocityDataReader
)
from .uav import (
    PX4_IMUDataReader,
    PX4_GPSDataReader,
    PX4_VisualOdometryDataReader,
    PX4_MagnetometerDataReader,
    PX4_ActuatorMotorDataReader,
    PX4_ActuatorOutputDataReader,
    PX4_CustomIMUDataReader,
    VOXL_TrackingCameraDataReader,
    VOXL_IMUDataReader,
    VOXL_StereoFrameReader,
    VOXL_QVIOOverlayDataReader
)
from .euroc import (
    EuRoC_IMUDataReader,
    EuRoC_LeiCaDataReader,
    EuRoC_StereoFrameReader,
    EuRoC_GroundTruthDataReader
)
from ..extended_common import (
    DatasetType,
    FilterConfig,
    DatasetConfig,
    SensorConfig,
    GyroSpecification,
    AccelSpecification,
    SensorType, 
    KITTI_SensorType,
    UAV_SensorType,
    EuRoC_SensorType,
    State, Pose,
    CoordinateFrame,
    ExtendedSensorConfig,
    StereoField,
    CoordinateFrame,
    SensorDataField,
    SensorData,
    ControlInput,
    VisualOdometryData,
    InitialState,
    
    IMU_FREQUENCY_MAP,
    EUROC_SEQUENCE_MAPS,
    UAV_SEQUENCE_MAPS,
    KITTI_SEQUENCE_TO_DATE,
    KITTI_SEQUENCE_TO_DRIVE,
    MAX_CONSECUTIVE_DROPOUT_RATIO
)


class Sensor:
    def __init__(
        self, 
        dataset, 
        type: SensorType, 
        output_queue: PriorityQueue,
        dropout_ratio: float=0.0):
        
        self.type = type
        self.dataset = dataset
        self.dataset_starttime = dataset.starttime
        self.starttime = None
        self.started = False
        self.stopped = False
        self.output_queue = output_queue
        
        self.field = namedtuple('sensor', ['type', 'data'])
        
        assert 0.0 <= dropout_ratio and dropout_ratio <= 1.0, "Please set dropout ratio 0. <= ratio <= 1."
        self.dropout_ratio = dropout_ratio
        
        publisher = self.publish_consecutive_drp if self.type is SensorType.KITTI_STEREO and\
                                                  self.dropout_ratio > 0.0 else self.publish
                                                  
        self.publish_thread = Thread(target=publisher)
        
    def start(self, starttime):
        self.started = True
        self.starttime = starttime
        self.publish_thread.start()
        
    def stop(self):
        self.stopped = True
        if self.started:
            self.publish_thread.join()
      
    def publish(self):
        dataset = iter(self.dataset)
        while not self.stopped:
            try:
                data = next(dataset)
            except StopIteration:
                return

            data_dropped = np.random.uniform() < self.dropout_ratio
            if not data_dropped:
                self.output_queue.put((data.timestamp, self.field(type=self.type, data=data)))
    
    def publish_consecutive_drp(self):
        """This is a publisher only used in experiment to obtain a data that is dropped consecutively.
        Assuming that 0. <= dropout ratio < 0.5
        Maximum sequence of dropout is 30% of total length:
          if 0.5 is set for dropout ratio, it is divided into 0.3 and 0.2
          
        """
        dataset = list(iter(self.dataset))
        iter_length = len(dataset)
        np.random.seed(777)
        
        dp_list = [MAX_CONSECUTIVE_DROPOUT_RATIO for i in range(int(self.dropout_ratio // MAX_CONSECUTIVE_DROPOUT_RATIO))]
      
        if self.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO != 0.0 and 0.001 < self.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO < MAX_CONSECUTIVE_DROPOUT_RATIO:
            dp_list.append(round(self.dropout_ratio % MAX_CONSECUTIVE_DROPOUT_RATIO, 3))
          
        logging.info("-"*30)
        logging.info(f"Stereo camera dropout ratio list: {dp_list}")
        logging.info("-"*30)
        
        dp_list.reverse()
        dropout_length = int(iter_length * dp_list.pop())
        start_dropping_at = np.random.randint(int(iter_length * 0.2))
        end_dropping_at = start_dropping_at + dropout_length
        
        i = 0
        while not self.stopped and i < iter_length:
            try:
                data = dataset[i]
                i += 1
            except StopIteration:
                return
            # FIXME: recover the comment out
            # data_dropped = start_dropping_at < i <= end_dropping_at
            data_dropped = False
            if 100 < i <= 200:
                data_dropped = True
            elif 500 < i <= 600:
                data_dropped = True
              
            if not data_dropped:
                self.output_queue.put((data.timestamp, self.field(type=self.type, data=data)))
        
            # if end_dropping_at < i and len(dp_list):
            #   dropout_length = int(iter_length * dp_list.pop())
            #   rest_iter = iter_length - end_dropping_at
            #   start_dropping_at = end_dropping_at + np.random.randint(int(rest_iter * 0.2))
            #   end_dropping_at = start_dropping_at + dropout_length

class BaseDataset:
    
    def __init__(
        self,
        config: DatasetConfig
        ):
        self.config = config
        
        self.type = DatasetType.get_type_from_str(self.config.type)
        
        self.output_queue = PriorityQueue()
        
        self.ground_truth_sensor_config = ExtendedSensorConfig(
            sensor_type=SensorType.GROUND_TRUTH, 
            name="ground_truth", 
            dropout_ratio=0., 
            window_size=1
        )
        
        self.sensor_list = self._get_sensor_list(type=self.config.type, sensors=self.config.sensors)

        assert len(self.sensor_list) > 0,\
                "Please select sensors"
        
        self.imu_frequency = self._set_imu_frequency()
        
        self.last_timestamp = None
        self.ground_truth_dataset = None

    def get_imu_frequency(self):
        logging.info(f"Selected IMU's frequency is: {self.imu_frequency}Hz.")
        return self.imu_frequency
    
    def _set_imu_frequency(self):
        
        imu = [sensor for sensor in self.sensor_list if SensorType.is_imu_data(sensor.sensor_type)]
        if len(imu) == 0:
            return 100.
        imu = imu.pop()
        
        return IMU_FREQUENCY_MAP.get(imu.sensor_type.name)        
    
    def _get_sensor_list(self, type: str, sensors: List[SensorConfig]) -> List[ExtendedSensorConfig]:
        sensor_list: List[ExtendedSensorConfig] = []
        get_sensor_from_str = SensorType.get_sensor_from_str_func(type)
        
        logging.debug("Configured sensors are: ")
        for s in sensors:
            sensor_type = get_sensor_from_str(s.name)
            if sensor_type is None:
                logging.warning(f"The sensor: {s} does not exist in {type} dataset.")
                continue
            sensor_list.append(ExtendedSensorConfig(
                name=s.name,
                dropout_ratio=s.dropout_ratio,
                window_size=s.window_size,
                sensor_type=sensor_type,
                args=s.args if s.args is not None else dict()
            ))
            logging.debug(sensor_type)
          
        return sensor_list
    
    def is_queue_empty(self) -> bool:
        return self.output_queue.empty()
    
class KITTIDataset(BaseDataset): 

    sensor_threads = None
    
    def __init__(
        self, 
        **kwargs,
        ):
        super().__init__(**kwargs)
        
        # assert date is not None, "Please provide proper kitti drive variant."
        
        self._populate_sensor_to_thread()

    def _populate_sensor_to_thread(self) -> List[Sensor]:
        
        def _get_dataset(sensor: ExtendedSensorConfig):
            
            date = KITTI_SEQUENCE_TO_DATE.get(self.config.variant)
            drive = KITTI_SEQUENCE_TO_DRIVE.get(self.config.variant)
            kwargs = {
                'root_path': self.config.root_path,
                'date': date,
                'drive': drive
            }
            match (sensor.sensor_type):
                case KITTI_SensorType.OXTS_IMU:
                    return OXTS_IMUDataReader(**kwargs)
                case KITTI_SensorType.OXTS_GPS:
                    return OXTS_GPSDataReader(**kwargs)
                case KITTI_SensorType.KITTI_STEREO:
                    return KITTI_StereoFrameReader(**kwargs)
                # case KITTI_SensorType.KITTI_VO:
                    # NOTE: This is experimental
                    # kwargs["simulate_delay"] = sensor.args.get("simulate_delay", False)
                    # kwargs["min_confidence"] = sensor.args.get("min_confidence", 0.8)
                    # kwargs["max_confidence"] = sensor.args.get("max_confidence", 1.0)
                    # kwargs["avg_delay"] = sensor.args.get("avg_delay", 0.5)
                    # kwargs["estimation_type"] = sensor.args.get("estimation_type", "pnp")
                    # return KITTI_VisualOdometry(**kwargs)
                case KITTI_SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY:
                    return KITTI_UpwardLeftwardVelocityDataReader(**kwargs)
                case SensorType.GROUND_TRUTH:
                    return KITTI_GroundTruthDataReader(**kwargs)
                case _:
                    return None
        
        sensor_threads = []
        
        for sensor in self.sensor_list:
            dataset = _get_dataset(sensor)

            if dataset is not None:
                s = Sensor(
                    type=sensor.sensor_type, 
                    dataset=dataset, 
                    output_queue=self.output_queue, 
                    dropout_ratio=sensor.dropout_ratio
                )
                sensor_threads.append(s)
        
        # ADD ground truth
        self.ground_truth_dataset = _get_dataset(self.ground_truth_sensor_config)
        s = Sensor(type=SensorType.GROUND_TRUTH, dataset=self.ground_truth_dataset, output_queue=self.output_queue)
        sensor_threads.append(s)
        
        self.sensor_threads = sensor_threads
    
    def start(self):
        now = time.time()
        
        for sensor_thread in self.sensor_threads:
            sensor_thread.start(starttime=now)
        
        time.sleep(0.5)
        last_timestamp, _ = self.output_queue.get()
        self.last_timestamp = last_timestamp
        
    def stop(self):
        for sensor_thread in self.sensor_threads:
            sensor_thread.stop()
    
    
    def get_sensor_data(self) -> SensorDataField:
        
        timestamp, sensor_data = self.output_queue.get()

        if SensorType.is_time_update(sensor_data.type):
            last_timestamp = self.last_timestamp
            self.last_timestamp = timestamp
            dt = timestamp - last_timestamp
        
        match(sensor_data.type):
            case KITTI_SensorType.OXTS_IMU:
                u = np.hstack([sensor_data.data.a, sensor_data.data.w])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=ControlInput(u=u, dt=dt),
                    coordinate_frame=CoordinateFrame.IMU)

            case KITTI_SensorType.OXTS_GPS:
                z = np.array([sensor_data.data.lon, sensor_data.data.lat, sensor_data.data.alt])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.GPS)
                
            case KITTI_SensorType.KITTI_STEREO:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.left_frame_id,
                        right_frame_id=sensor_data.data.right_frame_id),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)
            
            # case KITTI_SensorType.KITTI_VO:
                # data = VisualOdometryData(
                #     last_pose=sensor_data.data.last_pose,
                #     relative_pose=sensor_data.data.relative_pose,
                #     confidence=sensor_data.data.confidence,
                #     timestamp=timestamp,
                #     received_timestamp=timestamp,
                #     processed_timestamp=sensor_data.data.processed_timestamp,
                # )
                # data = VisualOdometryData(
                #     image_timestamp=timestamp,
                #     estimate_timestamp=timestamp,
                #     relative_pose=sensor_data.data.relative_pose,
                #     dt=sensor_data.data.dt
                # )
                # return SensorDataField(
                #     type=sensor_data.type, 
                #     timestamp=timestamp, 
                #     data=data,
                #     coordinate_frame=CoordinateFrame.STEREO_LEFT)
            
            case KITTI_SensorType.KITTI_COLOR_IMAGE:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.image_path, 
                        right_frame_id=sensor_data.data.image_path),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)

            case KITTI_SensorType.KITTI_UPWARD_LEFTWARD_VELOCITY:
                z = np.array([sensor_data.data.vu, sensor_data.data.vl])
                return SensorDataField(
                    type=sensor_data.type,
                    timestamp=timestamp,
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.INERTIAL)

            case SensorType.GROUND_TRUTH:
                pose = Pose(R=sensor_data.data.R, t=sensor_data.data.t)
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=pose.matrix(pose_only=True)),
                    coordinate_frame=CoordinateFrame.INERTIAL
                    )
            
            case _:
                return SensorDataField(
                        type=sensor_data.type, 
                        timestamp=timestamp, 
                        data=SensorData(z=np.zeros(3)),
                        coordinate_frame=CoordinateFrame.INERTIAL)

class UAVDataset(BaseDataset):

    sensor_threads: List[Sensor] = []
    imu_noise_vector = None
    
    def __init__(
        self, 
        **kwargs,
        ):
        
        super().__init__(**kwargs)

        sequence = UAV_SEQUENCE_MAPS.get(self.config.variant, "log0001")

        self.root_path = os.path.join(self.config.root_path, sequence)

        imu_config_path = self.config.imu_config_path
        if imu_config_path is None:
            imu_config_path = os.path.join(self.config.root_path, "configs/imu_config.yaml")
        
        sensor_config_path = self.config.sensor_config_path
        if sensor_config_path is None:
            sensor_config_path = os.path.join(self.config.root_path, "configs/sensor_path.yaml")

        filepath = None
        with open(sensor_config_path, "r") as f:
            filepath = yaml.safe_load(f)
            f.close()
            
        self.uav_sensor = namedtuple('uav_sensor', ['type', 'px4', 'voxl'])(**filepath[sequence])
        
        self.px4_path = namedtuple('px4', [
            'imu0_gyro', 'imu0_acc', 'imu1_gyro', 'imu1_acc',
            'gps', 'visual_odometry', 'actuator_motors', 'actuator_outputs', 'mag'
            ])(**self.uav_sensor.px4)
        
        self.voxl_path = namedtuple('voxl', [
            'imu0', 'imu1', 'stereo', 'qvio_overlay', 'tracking_camera'
            ])(**self.uav_sensor.voxl)
        
        self._populate_sensor_to_thread()

    def _populate_sensor_to_thread(self) -> List[Sensor]:
        
        px4_path = os.path.join(self.root_path, "px4")
        voxl_path = os.path.join(self.root_path, "run/mpa")
        
        def _get_dataset(sensor: ExtendedSensorConfig):
            match (sensor.sensor_type):
                case UAV_SensorType.VOXL_IMU0:
                    data_reader = VOXL_IMUDataReader(
                        path=os.path.join(voxl_path, self.voxl_path.imu0),
                        window_size=sensor.window_size,
                        )
                    return data_reader
                case UAV_SensorType.VOXL_IMU1:
                    data_reader = VOXL_IMUDataReader(
                        path=os.path.join(voxl_path, self.voxl_path.imu1),
                        window_size=sensor.window_size
                        )
                    return data_reader
                case UAV_SensorType.VOXL_STEREO:
                    return VOXL_StereoFrameReader(
                        path=os.path.join(voxl_path, self.voxl_path.stereo),
                        image_root_path=os.path.join(voxl_path, self.voxl_path.stereo.split("/")[0])
                    )
                case UAV_SensorType.VOXL_QVIO_OVERLAY:
                    return VOXL_QVIOOverlayDataReader(
                        path=os.path.join(voxl_path, self.voxl_path.qvio_overlay)
                    )
                case UAV_SensorType.VOXL_TRACKING_CAMERA:
                    return VOXL_TrackingCameraDataReader(
                        path=os.path.join(voxl_path, self.voxl_path.tracking_camera)
                    )
                case UAV_SensorType.PX4_IMU0:
                    data_reader = PX4_IMUDataReader(
                        gyro_path=os.path.join(px4_path, self.px4_path.imu0_gyro),
                        acc_path=os.path.join(px4_path, self.px4_path.imu0_acc),
                        window_size=sensor.window_size,
                        )
                    return data_reader
                case UAV_SensorType.PX4_IMU1:
                    data_reader = PX4_IMUDataReader(
                        gyro_path=os.path.join(px4_path, self.px4_path.imu1_gyro),
                        acc_path=os.path.join(px4_path, self.px4_path.imu1_acc),
                        window_size=sensor.window_size
                        )
                    return data_reader
                case UAV_SensorType.PX4_GPS:
                    return PX4_GPSDataReader(
                        path=os.path.join(px4_path, self.px4_path.gps),
                        )
                case UAV_SensorType.PX4_VO:
                    simulate_delay = sensor.args.get("simulate_delay", False)
                    min_confidence = sensor.args.get("min_confidence", 0.8)
                    max_confidence = sensor.args.get("max_confidence", 1.0)
                    avg_delay = sensor.args.get("avg_delay", 0.5)
                    return PX4_VisualOdometryDataReader(
                        path=os.path.join(px4_path, self.px4_path.visual_odometry),
                        simulate_delay=simulate_delay,
                        min_confidence=min_confidence,
                        max_confidence=max_confidence,
                        avg_delay=avg_delay
                    )
                case UAV_SensorType.PX4_MAG:
                    return PX4_MagnetometerDataReader(
                        path=os.path.join(px4_path, self.px4_path.mag),
                    )
                case SensorType.GROUND_TRUTH:
                    # NOTE: Currently, GPS data is set as a ground truth in UAV dataset
                    return PX4_GPSDataReader(
                        path=os.path.join(px4_path, self.px4_path.gps),
                    )
                case UAV_SensorType.UAV_VO:
                    return None
                    # return UAVCustomVisualOdometryDataReader(
                    #     path=os.path.join(px4_path, self.px4_path.visual_odometry),
                    # )
                case UAV_SensorType.PX4_ACTUATOR_MOTORS:
                    return PX4_ActuatorMotorDataReader(
                        path=os.path.join(px4_path, self.px4_path.actuator_motors),
                        model_path=os.path.join(self.config.root_path, "models/poly_model_rpm.npy"),
                        window_size=sensor.window_size,
                        starttime=603445530648,
                    )
                case UAV_SensorType.PX4_ACTUATOR_OUTPUTS:
                    return PX4_ActuatorOutputDataReader(
                        path=os.path.join(px4_path, self.px4_path.actuator_outputs),
                        window_size=sensor.window_size,
                        # starttime=604311034000,
                    )
                case UAV_SensorType.PX4_CUSTOM_IMU:
                    return PX4_CustomIMUDataReader(
                        path=px4_path,
                    )
                case _:
                    return None

        sensor_threads = []
        
        for sensor in self.sensor_list:
            dataset = _get_dataset(sensor)
            if dataset is not None:
                s = Sensor(
                    type=sensor.sensor_type, 
                    dataset=dataset, 
                    output_queue=self.output_queue, 
                    dropout_ratio=sensor.dropout_ratio
                )
                sensor_threads.append(s)
            
        
        # ADD ground truth
        self.ground_truth_dataset = _get_dataset(self.ground_truth_sensor_config)
        s = Sensor(type=SensorType.GROUND_TRUTH, dataset=self.ground_truth_dataset, output_queue=self.output_queue)
        sensor_threads.append(s)
        
        self.sensor_threads = sensor_threads
    
    def start(self):
        now = time.time()
        
        for sensor_thread in self.sensor_threads:
            sensor_thread.start(starttime=now)
        
        time.sleep(0.5)
        last_timestamp, _ = self.output_queue.get()
        self.last_timestamp = last_timestamp
        
    def stop(self):
        for sensor_thread in self.sensor_threads:
            sensor_thread.stop()

    def get_sensor_data(self) -> SensorDataField:
        
        timestamp, sensor_data = self.output_queue.get()

        if SensorType.is_time_update(sensor_data.type):
            last_timestamp = self.last_timestamp
            self.last_timestamp = timestamp
            dt = (timestamp - last_timestamp) / 1e9
        
        match(sensor_data.type):
            case UAV_SensorType.VOXL_IMU0 | UAV_SensorType.VOXL_IMU1\
                | UAV_SensorType.PX4_IMU0 | UAV_SensorType.PX4_IMU1:

                u = np.hstack([sensor_data.data.a, sensor_data.data.w])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=ControlInput(u=u, dt=dt),
                    coordinate_frame=CoordinateFrame.IMU)
            
            case UAV_SensorType.PX4_VO:
                data = VisualOdometryData(
                    image_timestamp=timestamp,
                    estimate_timestamp=timestamp,
                    relative_pose=sensor_data.data.relative_pose,
                    dt=sensor_data.data.dt
                )
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=data,
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)
            
            case UAV_SensorType.UAV_VO:
                data = VisualOdometryData(
                    image_timestamp=timestamp,
                    estimate_timestamp=timestamp,
                    relative_pose=sensor_data.data.relative_pose,
                    dt=sensor_data.data.dt
                )
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=data,
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)
            
            case UAV_SensorType.PX4_GPS:
                z = np.array([sensor_data.data.lon, sensor_data.data.lat, sensor_data.data.alt])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.GPS)
                
            case UAV_SensorType.PX4_MAG:
                z = np.array([sensor_data.data.x, sensor_data.data.y, sensor_data.data.z])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.MAGNETOMETER)
                
            case UAV_SensorType.VOXL_STEREO:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.left_frame_id,
                        right_frame_id=sensor_data.data.right_frame_id,
                    ),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)

            case SensorType.GROUND_TRUTH:
                z = np.array([sensor_data.data.lon, sensor_data.data.lat, sensor_data.data.alt])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.GPS)
                
            case UAV_SensorType.VOXL_QVIO_OVERLAY:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.image_path, 
                        right_frame_id=sensor_data.data.image_path),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)

            case UAV_SensorType.VOXL_TRACKING_CAMERA:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.image_path, 
                        right_frame_id=sensor_data.data.image_path),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)
            
            case UAV_SensorType.PX4_ACTUATOR_MOTORS:
                u = np.array([
                    sensor_data.data.c0, 
                    sensor_data.data.c1, 
                    sensor_data.data.c2, 
                    sensor_data.data.c3
                    ])
                return SensorDataField(
                    type=sensor_data.type,
                    timestamp=timestamp,
                    data=ControlInput(u=u, dt=dt),
                    coordinate_frame=CoordinateFrame.INERTIAL)
            case UAV_SensorType.PX4_ACTUATOR_OUTPUTS:
                u = np.array([
                    sensor_data.data.c0, 
                    sensor_data.data.c1, 
                    sensor_data.data.c2, 
                    sensor_data.data.c3
                    ])
                return SensorDataField(
                    type=sensor_data.type,
                    timestamp=timestamp,
                    data=ControlInput(u=u, dt=dt),
                    coordinate_frame=CoordinateFrame.INERTIAL)
            case UAV_SensorType.PX4_CUSTOM_IMU:
                z = np.hstack([sensor_data.data.a, sensor_data.data.w, sensor_data.data.m])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.IMU)
            case _:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=np.zeros(3)),
                    coordinate_frame=CoordinateFrame.INERTIAL)


class EuRoCDataset(BaseDataset):

    sensor_threads = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.variant = EUROC_SEQUENCE_MAPS.get(self.config.variant)
        self.root_path = os.path.join(self.config.root_path, self.variant)

        self._populate_sensor_to_thread()

    def _populate_sensor_to_thread(self) -> List[Sensor]:
        
        def _get_dataset(sensor: ExtendedSensorConfig):
            kwargs = {
                'root_path': self.root_path,
            }
            match (sensor.sensor_type):
                case EuRoC_SensorType.EuRoC_IMU:
                    kwargs["window_size"] = sensor.window_size
                    return EuRoC_IMUDataReader(**kwargs)

                case EuRoC_SensorType.EuRoC_LEICA:
                    return EuRoC_LeiCaDataReader(**kwargs)

                case EuRoC_SensorType.EuRoC_STEREO:
                    return EuRoC_StereoFrameReader(**kwargs)

                case SensorType.GROUND_TRUTH:
                    return EuRoC_GroundTruthDataReader(**kwargs)

                case _:
                    return None

        sensor_threads = []
        
        for sensor in self.sensor_list:
            dataset = _get_dataset(sensor)
            
            if dataset is not None:
                s = Sensor(
                    type=sensor.sensor_type, 
                    dataset=dataset, 
                    output_queue=self.output_queue, 
                    dropout_ratio=sensor.dropout_ratio
                )
                sensor_threads.append(s)
        
        # ADD ground truth
        self.ground_truth_dataset = _get_dataset(self.ground_truth_sensor_config)
        s = Sensor(type=SensorType.GROUND_TRUTH, dataset=self.ground_truth_dataset, output_queue=self.output_queue)
        sensor_threads.append(s)
        
        self.sensor_threads = sensor_threads

    def start(self):
        now = time.time()
        
        for sensor_thread in self.sensor_threads:
            sensor_thread.start(starttime=now)

        time.sleep(0.5)
        last_timestamp, _ = self.output_queue.get()
        self.last_timestamp = last_timestamp
        
    def stop(self):
        for sensor_thread in self.sensor_threads:
            sensor_thread.stop()

    def get_sensor_data(self) -> SensorDataField:
        
        timestamp, sensor_data = self.output_queue.get()

        if SensorType.is_time_update(sensor_data.type):
            last_timestamp = self.last_timestamp
            self.last_timestamp = timestamp
            dt = timestamp - last_timestamp
        
        match(sensor_data.type):
            case EuRoC_SensorType.EuRoC_IMU:
                u = np.hstack([sensor_data.data.a, sensor_data.data.w])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=ControlInput(u=u, dt=dt),
                    coordinate_frame=CoordinateFrame.IMU)

            case EuRoC_SensorType.EuRoC_LEICA:
                z = np.array([sensor_data.data.p_x, sensor_data.data.p_y, sensor_data.data.p_z])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.LEICA)
            
            case EuRoC_SensorType.EuRoC_STEREO:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=StereoField(
                        left_frame_id=sensor_data.data.left_frame_id,
                        right_frame_id=sensor_data.data.right_frame_id),
                    coordinate_frame=CoordinateFrame.STEREO_LEFT)

            case SensorType.GROUND_TRUTH:
                z = np.hstack([sensor_data.data.p, sensor_data.data.v, sensor_data.data.q])
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=z),
                    coordinate_frame=CoordinateFrame.INERTIAL)

            case _:
                return SensorDataField(
                    type=sensor_data.type, 
                    timestamp=timestamp, 
                    data=SensorData(z=np.zeros(3)),
                    coordinate_frame=CoordinateFrame.INERTIAL)

class Dataset:
    def __init__(
        self, 
        config: DatasetConfig,
    ):
        self.config = config
        self.dataset = None
        
        dataset_type = DatasetType.get_type_from_str(self.config.type)
        if dataset_type == DatasetType.KITTI:
            logging.info("Loading KITTI dataset...")
            self.dataset = KITTIDataset(config=config)
        elif dataset_type == DatasetType.UAV:
            logging.info("Loading UAV dataset...")
            self.dataset = UAVDataset(
                config=config,
            )
        elif dataset_type == DatasetType.EUROC:
            logging.info("Loading EuRoC dataset...")
            self.dataset = EuRoCDataset(
                config=config,
            )
        else:
            raise NotImplementedError(f"Dataset {config.type} not implemented.")
        
    def start(self):
        self.dataset.start()
    
    def stop(self):
        self.dataset.stop()

    def get_queue_size(self) -> int:
        return self.dataset.output_queue.qsize()
    
    def get_sensor_data(self) -> SensorDataField:
        return self.dataset.get_sensor_data()

    def get_initial_state(self, filter_config: FilterConfig) -> InitialState:
        if self.dataset.ground_truth_dataset is None:
            logging.warning("Ground truth dataset is not available. Returning default initial state.")
            x = State.get_initial_state_from_config(filter_config)
            P = np.eye(x.get_vector_size()) * 0.01
            return InitialState(x=x, P=P)

        x = self.dataset.ground_truth_dataset.get_initial_state(filter_config)
        P = np.eye(x.get_vector_size()) * 0.01
        logging.info(f"Initial state: {x}")
        return InitialState(x=x, P=P)
    
    def is_queue_empty(self) -> bool:
        return self.dataset.is_queue_empty()