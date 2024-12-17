import os
import sys
import logging
import numpy as np
from collections import namedtuple
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../interfaces'))

from config import DatasetConfig
from custom_types import (
    KITTI_SensorType,
    UAV_SensorType,
    Viztrack_SensorType,
    SensorType
)
from constants import (
    KITTI_DATE_MAPS,
    DECLINATION_OFFSET_RADIAN_IN_ESTONIA
)
from interfaces import State

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s > %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)



class BaseGeometryTransformer:
    
    def __init__(self):
        
        self._a = 6378137.0
        self._f = 1. / 298.257223563
        self._b = (1. - self._f) * self._a
        self._e = np.sqrt(self._a ** 2. - self._b ** 2.) / self._a
        self._e_prime = np.sqrt(self._a ** 2. - self._b ** 2.) /self. _b

    def _lla_to_ecef(self, points_lla: np.ndarray) -> np.ndarray:
        """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
        N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
        """
        lon = np.radians(points_lla[0])  # [N,]
        lat = np.radians(points_lla[1])  # [N,]
        alt = points_lla[2]  # [N,]

        N = self._a / np.sqrt(1. - (self._e * np.sin(lat)) ** 2.)  # [N,]
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1. - self._e ** 2.) + alt) * np.sin(lat)

        points_ecef = np.stack([x, y, z], axis=0)  # [3, N]
        return points_ecef


    def _ecef_to_enu(self, points_ecef: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
        N x [x, y, z] coords measured in a local East-North-Up frame.
        """
        lon = np.radians(ref_lla[0])
        lat = np.radians(ref_lla[1])

        ref_ecef = self._lla_to_ecef(ref_lla)  # [3,]

        relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]
        # R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
        R = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
            [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
        ])
        points_enu = R @ relative  # [3, N]
        return points_enu

    def _ecef_to_ned(self, points_ecef: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
        N x [x, y, z] coords measured in a local North-East-Down frame.
        """
        lon = np.radians(ref_lla[0])
        lat = np.radians(ref_lla[1])

        ref_ecef = self._lla_to_ecef(ref_lla)  # [3,]

        relative = points_ecef - ref_ecef[:, np.newaxis]  # [3, N]

        # R = Rz(np.pi / 2.0) @ Ry(np.pi / 2.0 - lat) @ Rz(lon)  # [3, 3]
        R = np.array([
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [-np.sin(lon), np.cos(lon), 0],
            [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
        ])
        points_ned = R @ relative  # [3, N]
        return points_ned


    def lla_to_enu(self, points_lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
        N x [x, y, z] coords measured in a local East-North-Up frame.
        """
        points_ecef = self._lla_to_ecef(points_lla)
        points_enu = self._ecef_to_enu(points_ecef, ref_lla)
        return points_enu

    def lla_to_ned(self, points_lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [longitude(deg), latitude(deg), altitude(m)] coords into
        N x [x, y, z] coords measured in a local North-East-Down frame.
        """
        points_ecef = self._lla_to_ecef(points_lla)
        points_ned = self._ecef_to_ned(points_ecef, ref_lla)
        return points_ned

    def enu_to_ecef(self, points_enu: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in a local East-North-Up frame into
        N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame.
        """
        # inverse transformation of `ecef_to_enu`

        lon = np.radians(ref_lla[0])
        lat = np.radians(ref_lla[1])
        alt = ref_lla[2]

        ref_ecef = self._lla_to_ecef(ref_lla)  # [3,]

        R = BaseGeometryTransformer.Rz(np.pi / 2.0) @ BaseGeometryTransformer.Ry(np.pi / 2.0 - lat) @ BaseGeometryTransformer.Rz(lon)  # [3, 3]
        R = R.T  # inverse rotation
        relative = R @ points_enu  # [3, N]

        points_ecef = ref_ecef[:, np.newaxis] + relative  # [3, N]
        return points_ecef


    def ecef_to_lla(self, points_ecef: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in Earth-Centered-Earth-Fixed frame into
        N x [longitude(deg), latitude(deg), altitude(m)] coords.
        """
        # approximate inverse transformation of `lla_to_ecef`
        
        x = points_ecef[0]  # [N,]
        y = points_ecef[1]  # [N,]
        z = points_ecef[2]  # [N,]

        p = np.sqrt(x ** 2. + y ** 2.)  # [N,]
        theta = np.arctan(z * self._a / (p * self._b))  # [N,]

        lon = np.arctan(y / x)  # [N,]
        lat = np.arctan(
            (z + (self._e_prime ** 2.) * self._b * (np.sin(theta) ** 3.)) / \
            (p - (self._e ** 2.) * self._a * (np.cos(theta)) ** 3.)
        )  # [N,]
        N = self._a / np.sqrt(1. - (self._e * np.sin(lat)) ** 2.)  # [N,]
        alt = p / np.cos(lat) - N  # [N,]

        lon = np.degrees(lon)
        lat = np.degrees(lat)

        points_lla = np.stack([lon, lat, alt], axis=0)  # [3, N]
        return points_lla


    def enu_to_lla(self, points_enu: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
        """transform N x [x, y, z] coords measured in a local East-North-Up frame into
        N x [longitude(deg), latitude(deg), altitude(m)] coords.
        """
        points_ecef = self._enu_to_ecef(points_enu, ref_lla)
        points_lla = self._ecef_to_lla(points_ecef)
        return points_lla

    def _get_rigid_transformation(self, calib_path: str) -> np.ndarray:
        with open(calib_path, 'r') as f:
            calib = f.readlines()
        R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
        t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
        T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
        return T
    
    @staticmethod
    def Rx(theta):
        """rotation matrix around x-axis
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])


    @staticmethod
    def Ry(theta):
        """rotation matrix around y-axis
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])


    @staticmethod
    def Rz(theta):
        """rotation matrix around z-axis
        """
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])


class KITTI_GeometricTransformer(BaseGeometryTransformer):

    def __init__(
            self,
            dataset_config: DatasetConfig
        ):
        super().__init__()
        
        self.dataset_config = dataset_config
        
        date = KITTI_DATE_MAPS.get(self.dataset_config.variant)
        assert date is not None, "Please provide proper kitti drive variant."
    
        self.calibration_path = os.path.join(self.dataset_config.root_path, date)

        self.T_velo_ref0 = self._get_rigid_transformation(os.path.join(self.calibration_path, "calib_velo_to_cam.txt"))
        self.T_imu_velo = self._get_rigid_transformation(os.path.join(self.calibration_path, "calib_imu_to_velo.txt"))
        self.T_from_imu_to_cam = self.T_imu_velo @ self.T_velo_ref0
        self.T_from_cam_to_imu = np.linalg.inv(self.T_from_imu_to_cam)
        
        self.origin = None

    def transform_data(
            self, 
            sensor_type: SensorType, 
            value: np.ndarray,
            state: State,
            to: SensorType = None,
        ) -> np.ndarray:
        """By default, given data is transformed into the inertial frame coordinate.

        Args:
            sensor_type (SensorType): _description_
            value (np.ndarray): _description_
            state (State): _description_
            to (SensorType, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        
        if to is not None:
            return self._transform_data(sensor_type=sensor_type, value=value, state=state, to=to)
        
        match (sensor_type):
            case KITTI_SensorType.OXTS_GPS:
                return self._process_kitti_gps_data(value).reshape(-1, 1)
            case SensorType.KITTI_CUSTOM_VO:
                return self._transform_kitti_vo_data_into_imu_coord(value).reshape(-1, 1)
            case SensorType.GROUND_TRUTH:
                return self._process_kitti_ground_truth(value)
            case _:
                return value.reshape(-1, 1)
            
    def _transform_data(
        self,
        sensor_type: SensorType,
        value: np.ndarray,
        state: State, 
        to: SensorType
    ):
        """All combinations of transformation from a sensor to another.

        Args:
            sensor_type (SensorType): _description_
            value (np.ndarray): _description_
            state (State): _description_
            to (SensorType): _description_
        """
        match (sensor_type):
            case KITTI_SensorType.OXTS_IMU:
                return self._transform_imu_data(value=value, state=state, to=to)
            case _:
                return value
            
    def _transform_imu_data(self, value: np.ndarray, state: State, to: SensorType):
        match (to):
            case KITTI_SensorType.OXTS_GPS:
                return self._transform_kitti_imu_into_gps(value)
            case KITTI_SensorType.KITTI_CUSTOM_VO:
                return self.T_from_imu_to_cam @ value
            case _:
                return value
        
    def _transform_kitti_imu_into_gps(self, imu_data: np.ndarray) -> np.ndarray:
        values = np.array([imu_data[0], imu_data[1], imu_data[2], 1])
        transformed_values = self.T_from_imu_to_cam @ values
        return np.array([transformed_values[0], transformed_values[1], transformed_values[2]])
    
    def _process_kitti_ground_truth(self, pose_data: np.ndarray) -> np.ndarray:
        
        if pose_data.shape != np.eye(4).shape:
            logger.warning("Different shape of ground-truth value found")
            return np.array([0., 0., 0.])
        # NOTE: get translation vector
        t = pose_data[:3, 3]
        return self._process_kitti_gps_data(t)
        
    def _process_kitti_gps_data(self, gps_data: np.ndarray) -> np.ndarray:
        data = self._transform_kitti_gps_data_into_imu_coord(gps_data)
        # if self.origin is None:
        #     self.origin = data.copy()
            
        # data -= self.origin
        return data
    
    # def _process_kitti_gps_data(self, gps_data: np.ndarray) -> np.ndarray:
        
    #     if self.origin is None:
    #         self.origin = gps_data
    #         return np.array([0., 0., 0.])
        
    #     # NOTE: lla to enu coord
    #     gps_data = self.lla_to_enu(gps_data.reshape(-1, 1), self.origin).flatten()
    #     return gps_data
        
    def _transform_kitti_gps_data_into_imu_coord(self, gps_data):
        lla_values = np.array([gps_data[0], gps_data[1], gps_data[2], 1])
        transformed_lla_values = self.T_from_cam_to_imu @ lla_values
        return np.array([transformed_lla_values[0], transformed_lla_values[1], transformed_lla_values[2]])
        
    def _transform_kitti_vo_data_into_imu_coord(self, vo_data: np.ndarray) -> np.ndarray:
        t_velocity = self.T_from_cam_to_imu @ np.array([vo_data[0], vo_data[1], vo_data[2], 1])
        
        if vo_data.shape[0] == 6:
            t_position = t_velocity 
            t_velocity = self.T_from_cam_to_imu @ np.array([vo_data[3], vo_data[4], vo_data[5], 1])
            
            # if self.origin is not None:
            #     t_position[:3] -= self.origin
            
            return np.hstack([t_position[:3], t_velocity[:3]])

        # NOTE: Assuming velocity only
        return t_velocity[:3]


class UAV_GeometricTransformer(BaseGeometryTransformer):

    def __init__(
            self,
            dataset_config: DatasetConfig
        ):
        super().__init__()
        
        self.dataset_config = dataset_config
        
        self.origin = None
    

    def transform_data(
            self, 
            sensor_type: SensorType, 
            value: np.ndarray,
            state: State,
            to: SensorType = None
        ) -> np.ndarray:
        """By default, given data is transformed into the inertial frame coordinate.

        Args:
            sensor_type (SensorType): _description_
            value (np.ndarray): _description_
            state (State): _description_
            to (SensorType, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        
        if to is not None:
            return self._transform_data(sensor_type=sensor_type, value=value, state=state, to=to)
        
        match (sensor_type):
            case UAV_SensorType.PX4_GPS:
                return self._process_px4_gps_data(gps_data=value).reshape(-1, 1)
            case UAV_SensorType.PX4_VO:
                return self._process_px4_vo_data(vo_data=value).reshape(-1, 1)
            case UAV_SensorType.PX4_MAG:
                return self._process_mag_data(mag_data=value, state=state).reshape(-1, 1)
            case SensorType.UAV_CUSTOM_VO:
                # Same VO provided by PX4, but tries to estimate pose based on system's state
                return self._process_uav_vo_data(vo_data=value).reshape(-1, 1)
            case SensorType.GROUND_TRUTH:
                return self._process_px4_gps_data(gps_data=value)
            case _:
                return value
            
    def _transform_data(
        self,
        sensor_type: SensorType,
        value: np.ndarray,
        state: State, 
        to: SensorType
    ):
        """All combinations of transformation from a sensor to another.

        Args:
            sensor_type (SensorType): _description_
            value (np.ndarray): _description_
            state (State): _description_
            to (SensorType): _description_
        """
        return value
        
    def _process_uav_vo_data(self, vo_data: np.ndarray) -> np.ndarray:
        position = BaseGeometryTransformer.Rz(np.radians(90)) @ vo_data[:3]
        position = position.flatten()
        return position
        
    def _process_px4_vo_data(self, vo_data: np.ndarray) -> np.ndarray:
        position = BaseGeometryTransformer.Rz(np.radians(90)) @ vo_data[:3]
        velocity = BaseGeometryTransformer.Rz(np.radians(90)) @ vo_data[3:]
        position = position.flatten()
        velocity = velocity.flatten()
        return np.hstack([position, velocity])
    
    def _decimal_place_shift(self, values: np.ndarray) -> float:
        return np.array([float(value / 10**(len(str(value)) - 2)) for value in values])
        
    def _process_px4_gps_data(self, gps_data: np.ndarray) -> np.ndarray:
        if self.origin is None:
            self.origin = self._decimal_place_shift(gps_data)
            return np.array([0., 0., 0.])
        
        # NOTE: lla to ned coord
        gps_data = self.lla_to_ned(self._decimal_place_shift(gps_data).reshape(-1, 1), self.origin).flatten()
        return gps_data
        
    def _process_mag_data(self, mag_data: np.ndarray, state: State) -> np.ndarray:
        """Convert magnetometer data measured in Body frame into inertial frame using the rotation matrix obtained from the quaternion.

        Args:
            position (np.ndarray): A earth's magnetic field reading in body frame.

        Returns:
            np.ndarray: The Earth's magnetic field in inertial frame.
        """
        mag_data = mag_data.flatten()
        # TODO: convert lla to NED coordinate
        Rot = state.get_rotation_matrix()
        mag_data = (Rot @ mag_data).flatten()
        
        z_m = np.arctan2(mag_data[0], mag_data[1])
        z_m += DECLINATION_OFFSET_RADIAN_IN_ESTONIA
        
        return z_m


class Viztrack_GeometricTransformer(BaseGeometryTransformer):

    def __init__(
            self,
            dataset_config: DatasetConfig
        ):
        super().__init__()
        
        self.dataset_config = dataset_config
        
        self.origin = None

    def transform_data(
            self, 
            sensor_type: SensorType, 
            value: np.ndarray,
            state: State,
            to: SensorType = None,
        ) -> np.ndarray:
        """By default, given data is transformed into the inertial frame coordinate.

        Args:
            sensor_type (SensorType): _description_
            value (np.ndarray): _description_
            state (State): _description_
            to (SensorType, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        
        if to is not None:
            return self._transform_data(sensor_type=sensor_type, value=value, state=state, to=to)
        
        match (sensor_type):
            case Viztrack_SensorType.Viztrack_GPS:
                return self._process_gps_transform(value).reshape(-1, 1)
            case SensorType.GROUND_TRUTH:
                return self._process_gps_transform(value)
            case _:
                return value.reshape(-1, 1)
            
    def _process_gps_transform(self, gps_data: np.ndarray) -> np.ndarray:
        
        # NOTE: lla to enu coord
        gps_data = self.lla_to_enu(gps_data.reshape(-1, 1), self.origin).flatten()
        
        if self.origin is None:
            self.origin = gps_data
            return np.array([0., 0., 0.])
        
        return gps_data
    
    def _transform_data(self, sensor_type=SensorType, value=np.ndarray, state=State, to=SensorType):
        
        match (sensor_type):
            case KITTI_SensorType.OXTS_IMU:
                return self._transform_imu_data(value=value, state=state, to=to)
            case _:
                return value
            
    def _transform_imu_data(self, value=np.ndarray, state=State, to=SensorType):
        
        match (to):
            case Viztrack_SensorType.OAK_D_IMU:
                # Rotate IMU data (NUE) to ENU
                return self.Rx(-np.pi/2) @ value
            case Viztrack_SensorType.Viztrack_GPS:
                return value
            case _:
                return value

if __name__ == "__main__":
    import pykitti
    from custom_types import DatasetType
    from visualizer import Visualizer, BaseVisualizationField
    from config import VisualizationConfig, FilterConfig
    sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset'))
    from dataset import KITTI_GroundTruthDataReader, KITTI_CustomVisualOdometry, PX4_GPSDataReader
    
    def _visualize(dataset=DatasetType):
        
        if dataset is DatasetType.KITTI:
            data_config = DatasetConfig(
                type="kitti",
                mode="stream",
                variant="0067",
                root_path="../../data/KITTI/",
                sensors=[]
            )
            geo_transformer = KITTI_GeometricTransformer(
                dataset_config=data_config
            )
            gt = KITTI_GroundTruthDataReader(
                root_path="../../data/KITTI",
                date="2011_09_26",
                drive="0067"
            )
            vo = KITTI_CustomVisualOdometry(
                root_path="../../data/KITTI",
                date="2011_09_26",
                drive="0067"
            )
        
        else:
            data_config = DatasetConfig(
                type="uav",
                mode="stream",
                variant="log0001",
                root_path="../../data/UAV/",
                sensors=[]
            )
            geo_transformer = UAV_GeometricTransformer(
                dataset_config=data_config
            )

            gt = PX4_GPSDataReader(
                path="../../data/UAV/log0001/px4/09_00_22_sensor_gps_0.csv",
                divider=1
            )
        
        filter_config = FilterConfig(
            type="ekf",
            dimension=3,
            motion_model="kinematics",
            noise_type=False,
            params=None
        )
        vis_config = VisualizationConfig(
            realtime=False,
            output_filepath="./",
            save_trajectory=True,
            show_end_result=True,
            show_vio_frame=False,
            show_particles=False,
            set_lim_in_plot=False,
            save_frames=False,
            show_vo_trajectory=False,
            show_innovation_history=False,
            limits=[]
        )
            
        visualizer =  Visualizer(
            config=vis_config,
            filter_config=filter_config
        )
        
        visualizer.start()
        
        state = State(
            p=np.zeros((3, 1)),
            v=np.zeros((3, 1)),
            q=np.array([1., 0., 0., 0.])
        )
        

        gt_dataset = iter(gt)
        vo_dataset = iter(vo)
        while True:
            try:
                data = next(gt_dataset)
                _vo = next(vo_dataset)
                
                if dataset is DatasetType.KITTI:
                    z = np.hstack([data.R, data.t.reshape(-1, 1)])
                    value = np.vstack([z, np.array([0., 0., 0., 1.])])
                else:
                    value = np.array([data.lon, data.lat, data.alt])
                    
                gps = geo_transformer.transform_data(
                    sensor_type=SensorType.GROUND_TRUTH,
                    value=value,
                    state=state
                )
                
                _vo =  geo_transformer.transform_data(
                    sensor_type=SensorType.KITTI_CUSTOM_VO,
                    value=np.array([_vo.x, _vo.y, _vo.z, _vo.vx, _vo.vy, _vo.vz]),
                    state=state
                )
                _vo = _vo.flatten()
                
                data = BaseVisualizationField(
                    x=gps[0], 
                    y=gps[1],
                    z=gps[2],
                    lw=1,
                    color='black'
                )
                
                vo_data = BaseVisualizationField(
                    x=_vo[0], 
                    y=_vo[1],
                    z=_vo[2],
                    lw=1,
                    color='blue'
                )
                visualizer.show_realtime_estimation(data=[
                    data,
                    vo_data
                ])
            except StopIteration:
                break
            except Exception as e:
                print(e)
        
        # date = KITTI_DATE_MAPS.get(data_config.variant)
        # kitti_dataset = pykitti.raw(data_config.root_path, date, data_config.variant)
        # for i, oxts in enumerate(kitti_dataset.oxts):
        #     packet = oxts.packet
        #     g = np.array([
        #                     packet.lon,
        #                     packet.lat,
        #                     packet.alt
        #                 ])
        #     gps = geo_transformer.transform_data(
        #         sensor_type=KITTI_SensorType.OXTS_GPS,
        #         value=g,
        #         state=state
        #     )
        #     data = VisualizationField(
        #         x=gps[0], 
        #         y=gps[1],
        #         z=gps[2]
        #     )
        #     visualizer.show_realtime_estimation(data=[data])
        
        visualizer.show_final_estimation(labels=["GT", "VO"])
        visualizer.stop()
    
    def _rotation_check():
        base = BaseGeometryTransformer()
        value = np.array([0., -9.81, 1.])
        print(f"Before rotation: {value}")
        rotated = base.Rx(-np.pi/2) @ value
        print(f"After rotation: {rotated}")
        
    
    # _visualize(dataset=DatasetType.KITTI)
    
    # _rotation_check()