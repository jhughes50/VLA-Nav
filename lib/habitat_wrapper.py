"""
    CIS 6200 -- Deep Learning Final Project
    Wrap the Habitat Sim for training
    April 2024
"""

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
import quaternion

class HabitatWrapper:

    def __init__(self, file_path):

        self.sim_ = None
        self.prev_scene_id_ = None
        self.file_path_ = file_path
        self.camera_res_ = [384, 384]

    def __del__(self):
        if not isinstance(None, type(None)):
            self.sim_.close()

    def place_agent(self, rotation, pose):
        state = habitat_sim.AgentState()
        state.position = pose
        state.rotation = quaternion.from_rotation_matrix(rotation)
        self.sim_.agents[0].set_state(state)

        return self.sim_.agents[0].scene_node.transformation_matrix()
 
    def update_sim(self, scene_id):
        if scene_id != self.prev_scene_id_:
            print("[SIM-WRAPPER] Updating sim at %s" %scene_id)
            self.prev_scene_id_ = scene_id
            if self.sim_ != None:
                print("[SIM-WRAPPER] Closing old sim")
                self.sim_.close()
            self.sim_ = habitat_sim.Simulator(self.make_config(scene_id))
            print("[SIM-WRAPPER] Finished updating")
        else:
            self.prev_scene_id_ = scene_id
            print("[SIM WRAPPER] Same scene no need to update")

    def get_sensor_obs(self, uuid):
        return self.sim_.get_sensor_observations()[uuid]
   
    def close(self):
        self.sim_.close()

    def make_config(self, scene_id):
        scene_file = self.file_path_ + "../data/mp3d/%s/%s.glb" %(scene_id, scene_id)

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_file

        backend_cfg.scene_dataset_config_file = self.file_path_+"../data/mp3d/mp3d.scene_dataset_config.json"
        
        camera_resolution = [240, 320]
        sensors = {
            "rgba_camera": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],  # ::: fix y to be 0 later
            },
                "semantic_camera": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": camera_resolution,
                "position": [0.0, 0.0, 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_specs.append(sensor_spec)
        
        # agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        
        return habitat_sim.Configuration(backend_cfg, [agent_cfg])
