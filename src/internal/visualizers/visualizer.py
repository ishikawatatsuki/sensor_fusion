from .realtime_visualizer import RealtimeVisualizer
from .static_visualizer import StaticVisualizer

from ..extended_common.extended_config import VisualizationConfig, FilterConfig


class Visualizer:

    def __init__(
            self,
            visualization_config: VisualizationConfig,
            filter_config: FilterConfig
    ):
        self.config = visualization_config
        self.filter_config = filter_config

        if self.config.type == "realtime":
            self.visualizer = RealtimeVisualizer(visualization_config)
        else:
            self.visualizer = StaticVisualizer(visualization_config, filter_config)

    def run(self):
        self.visualizer.run()

    def stop(self):
        self.visualizer.stop()
