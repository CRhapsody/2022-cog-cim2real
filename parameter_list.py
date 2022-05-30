env_path = "C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\win_v2.1\\cog_sim2real_env.exe"
nav_policies_path = "C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\models\\nav_policies"

num_iterations = int(1_000_000) # @param {type:"integer"}

initial_collect_steps = 2000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (512, 256, 256)
critic_joint_fc_layer_params = (512, 256, 256)

log_interval = 10 # @param {type:"integer"}

num_episodes = 10
num_eval_episodes = 1 # @param {type:"integer"}
num_eval_steps = 1000
eval_interval = 500 # @param {type:"integer"}

policy_save_interval = 500 # @param {type:"integer"}