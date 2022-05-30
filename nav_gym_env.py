import gym
from gym import spaces
import numpy as np
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import math
from logger import logging
from stable_baselines3.common.env_checker import check_env
import cv2


class NavEnvironment(gym.Env):
    MAX_WIDTH = 8.08
    MAX_HEIGHT = 4.48
    MAX_TIME = 180
    MAX_DISTANCE = math.sqrt(8.08 ** 2 + 4.48 ** 2)

    def __init__(self, Cog_env: CogEnvDecoder, env_id=1):
        super(NavEnvironment, self).__init__()
        self._Cog_env = Cog_env
        self.env_id = env_id
        # Define action and observation space
        # They must be gym.spaces objects
        # self._action_spec = array_spec.BoundedArraySpec(  # [vx, vy, vw]
        #     shape=(3,), dtype=np.float32, minimum=[-2, -2, -math.pi], maximum=[2, 2, math.pi], name="action"
        # )
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # vector_state中包含全部信息。但是在导航任务中，仅取子集。
        # 其中self_info包含血量等对抗信息，不需考虑。
        # enemy相关信息仅考虑pos（把它作为一个障碍物）
        # goal只需要取一个，即当前正在前往的目标。
        #         vector_state = [self_pos, self_info, enemy_act, enemy_pos, enemy_info,
        #                         goal1, goal2, goal3, goal4, goal5, collision_info]
        #         state = {"color_image":img, "laser": laser, "vector": vector_state}
        # 考虑状态因素：
        # laser: 61,
        # self_pos: 3 (x, y, angle),
        # enemy_pos: 3 (x, y, angle)
        # collision: 1 (0 or 1 表示当前是否正在碰撞)
        # goal: 2 (x_pos, y_pos)
        # time_taken: 1 表示当前已经经过了多少个时间步。时间拖得越长分数越低，所以必须催使AI尽快完成导航任务。
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     # 61+3+3+2+1+1 = 71
        #     shape=(71,), dtype=np.float32, name="observation"
        # )
        self.observation_space = spaces.Dict(spaces={
            # "image": spaces.Box(low=0, high=1, shape=(3, 64, 64), dtype=np.float32),
            "laser": spaces.Box(low=0, high=1, shape=(1, 61), dtype=np.float32),
            "vec": spaces.Box(low=0, high=1, shape=(7, ), dtype=np.float32)
        })
        # curr_goal表示目前的导航任务正在寻找的目标点是哪个目标点。
        self.curr_goal = 1
        # time_taken表示当前花费的时间
        self.time_taken = 0
        self._obs_state = None
        self._dict_state = None

    def transform_state(self, state, judge_result):
        _, time_taken, _, flag_ach = judge_result
        self.time_taken = time_taken

        find_goal = False
        flag_ach = int(flag_ach)
        old_goal = self.curr_goal
        self.curr_goal = flag_ach + 1
        if self.curr_goal > old_goal:
            find_goal = True
            logging.info(f"env {self.env_id} find goal {old_goal}.")

        color_image = state["color_image"]
        color_image = cv2.resize(color_image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        color_image = np.transpose(color_image, (2,0,1))
        color_image = color_image.astype(dtype=np.float32)
        color_image /= 255.

        laser_list = state["laser"].tolist()
        assert len(laser_list) == 61
        laser_list = np.array(laser_list, dtype=np.float32) / self.MAX_DISTANCE
        laser_list = laser_list.reshape(1, 61)
        vector_state = state["vector"]
        def transform_pos(pos):
            position = pos[0] / self.MAX_WIDTH, pos[1] / self.MAX_HEIGHT
            angle = pos[2]
            if angle < 0:
                angle += 2*math.pi
            angle /= 2*math.pi
            return position[0], position[1], angle

        self_pos = vector_state[0]
        enemy_pos = vector_state[3]
        norm_self_pos = transform_pos(self_pos)
        norm_enemy_pos = transform_pos(enemy_pos)

        if 1 <= self.curr_goal <= 5:
            goal_info = vector_state[4+self.curr_goal]
            goal_x, goal_y, goal_reached = goal_info
            goal = [goal_x, goal_y]
            norm_goal = [goal_x/self.MAX_WIDTH, goal_y/self.MAX_HEIGHT]
        else:
            goal = [self_pos[0] + 1e6, self_pos[1] + 1e6]
            norm_goal = [norm_self_pos[0], norm_self_pos[1]]

        norm_collision = 0
        collision_info = vector_state[-1]
        _, _, distance = self.goal_distance(self_pos, goal)
        goal_angle = abs(self.goal_angle(self_pos, goal))
        distance_variation = 0
        goal_angle_variation = 0

        if self._dict_state is not None:
            if collision_info[1] - self._dict_state["collision_info"][1] > 0:
                norm_collision = 1
            distance_variation = distance - self._dict_state["distance"]
            goal_angle_variation = goal_angle - self._dict_state["goal_angle"]

        rel_pos = [goal[0] - self_pos[0], goal[1] - self_pos[1], abs(self.goal_angle(self_pos, goal))]
        norm_rel_pos = [rel_pos[0]/self.MAX_WIDTH, rel_pos[1]/self.MAX_HEIGHT, (rel_pos[2]/math.pi)]
        time_taken = self.time_taken / self.MAX_TIME
        dict_state = {
            "laser": laser_list,
            "self_pos": self_pos,
            "norm_self_pos": norm_self_pos,
            "enemy_pos": enemy_pos,
            "norm_enemy_pos": norm_enemy_pos,
            "norm_collision": norm_collision,
            "collision_info": collision_info,
            "time_taken": time_taken,
            "goal_pos": goal,
            "norm_goal_pos": norm_goal,
            "rel_pos": rel_pos,
            "norm_rel_pos": norm_rel_pos,
            "distance": distance,
            "distance_variation": distance_variation,
            "goal_angle": goal_angle,
            "goal_angle_variation": goal_angle_variation,
            "find_goal": find_goal,
            "color_image": color_image
        }
        obs_state = {
            "laser": laser_list,
            "vec": np.array(np.hstack([norm_self_pos, norm_rel_pos, norm_collision]), dtype=np.float32),
            "image": color_image
        }
        # obs_state = np.array(np.hstack([laser_list, norm_self_pos, norm_enemy_pos, norm_collision, norm_goal]), dtype=np.float32)
        return obs_state, dict_state

    def step(self, action, auto_reset=True):
        # action仅包含前三列（x, y, w），在这里手动加上第四列（fire始终为False）
        full_action = [action[0] * 2, action[1] * 2, action[2] * math.pi/4, 0]
        state, _, done, [info, judge_result] = self._Cog_env.step(full_action)
        # judge_result = [score, time_taken, dmg, flag_ach]
        self._obs_state, self._dict_state = self.transform_state(state, judge_result)
        reward = self.reward()
        # 由于转换为单个目标的寻路任务，所以当到达第一个目标后直接done
        if self.curr_goal > 5 or done:
            if self.curr_goal > 5:
                logging.info(f"env id: {self.env_id} find last goal.")
                reward = 1000
            else:
                reward = -1000
            done = True
            if auto_reset:
                self.reset()
        return self._obs_state, reward, done, {"info": info, "judge_result": judge_result}

    def reward(self):
        find_goal_reward = 0
        if self._dict_state["find_goal"]:
            find_goal_reward = 500

        # _, _, distance = self.goal_distance(self._dict_state["self_pos"], self._dict_state["goal_pos"])
        # norm_distance = distance / self.MAX_DISTANCE
        # distance_reward = -norm_distance
        distance_variation = self._dict_state["distance_variation"]
        distance_reward = 0
        if distance_variation < 0:
            distance_reward = 1
        elif distance_variation > 0:
            distance_reward = -1

        angle_variation = self._dict_state["goal_angle_variation"]
        angle_reward = 0
        if angle_variation < 0:
            angle_reward = 1
        elif angle_variation > 0:
            angle_reward = -1

        collision_penalty = 0
        if self._dict_state["norm_collision"] == 1:
            collision_penalty = -50
        collision_reward = collision_penalty

        # abs_goal_angle = abs(self.goal_angle(self._dict_state["self_pos"], self._dict_state["goal_pos"]))
        # max_angle = math.pi
        # norm_angle = abs_goal_angle / max_angle
        # angle_reward = -norm_angle

        # time_taken = self.time_taken
        # norm_time_taken = time_taken / self.MAX_TIME
        # time_taken_reward = -norm_time_taken

        reward = find_goal_reward + distance_reward + angle_reward + collision_reward
        return reward

    def goal_distance(self, self_pos=None, goal_pos=None):
        if self_pos is None and goal_pos is None:
            self_pos = self._dict_state["self_pos"]
            goal_pos = self._dict_state["goal_pos"]
        self_pos_x, self_pos_y, _ = self_pos
        goal_pos_x, goal_pos_y = goal_pos
        distance = np.sqrt((self_pos_x - goal_pos_x) ** 2 + (self_pos_y - goal_pos_y) ** 2)
        return (self_pos_x, self_pos_y), (goal_pos_x, goal_pos_y), distance

    # 计算当前车辆位置与目标位置形成的向量与车辆方向所形成的夹角。
    # 已当前车辆方向为极坐标轴，取[-pi, pi]的范围。
    def goal_angle(self, self_pos=None, goal_pos=None):
        if self_pos is None and goal_pos is None:
            self_pos = self._dict_state["self_pos"]
            goal_pos = self._dict_state["goal_pos"]
        # self_angle取值范围为[-pi/2, 3*pi/2]，在极坐标的0到3*pi/2是整数，3*pi/2到2pi为负数。
        self_pos_x, self_pos_y, self_angle = self_pos
        goal_pos_x, goal_pos_y = goal_pos
        self_vec = [np.cos(self_angle), np.sin(self_angle)]
        # 设自己为原点，再求出目标到当前位置的向量。
        goal_vec = [goal_pos_x - self_pos_x, goal_pos_y - self_pos_y]
        angle = NavEnvironment.angle(self_vec, goal_vec)
        # 此时计算的angle是属于0，pi的绝对角度。我们需要把它转化为从self_vec向哪个方向旋转后能够得到goal_vec的angle。
        # 需要确定angle的正负号。所以我们将self_vec旋转angle或-angle角度，角度较小的那个就是正确的angle
        pos_rot = self.rotate(self_vec, angle)
        pos_angle = self.angle(pos_rot, goal_vec)
        neg_rot = self.rotate(self_vec, -angle)
        neg_angle = self.angle(neg_rot, goal_vec)
        angle = -angle if neg_angle < pos_angle else angle
        return angle

    @staticmethod
    def angle(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        unit_vector_1 = vec1 / np.linalg.norm(vec1)
        unit_vector_2 = vec2 / np.linalg.norm(vec2)
        return np.arccos(np.clip(np.dot(unit_vector_1, unit_vector_2), -1.0, 1.0))

    @staticmethod
    def rotate(vec, angle):
        vec = np.array(vec)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rot_vec = np.dot(rot, vec).tolist()
        return rot_vec

    def reset(self):
        state = self._Cog_env.reset()
        while not np.any(state["color_image"]):
            state, _, _, _ = self._Cog_env.step([0, 0, 0, 0])

        state, _, _, [_, judge_result] = self._Cog_env.step([0, 0, 0, 0])
        self.curr_goal = 1
        self.time_taken = 0
        self._obs_state = None
        self._dict_state = None
        self._obs_state, self._dict_state = self.transform_state(state, judge_result)
        return self._obs_state

    def render(self, mode="human"):
        pass

    def close(self):
        self._Cog_env.close()


def main():
    cog_env = CogEnvDecoder(
        env_name="C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\win_v2.1\\cog_sim2real_env.exe",
        no_graphics=False, time_scale=1, worker_id=1)
    env = NavEnvironment(cog_env)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    for i in range(3000):
        env.step([-0.5, 0, 1])
        r = env.reward()
        goal_angle = env.goal_angle()
        goal_rel_pos = env.goal_distance()
        print(f"reward: {r}, goal_angle: {goal_angle}, goal_rel_pos: {goal_rel_pos}")


if __name__ == '__main__':
    main()
