from DRL.TD3 import TD3
from utils.drl_utils import create_directory, scale_action
from FL.fed_environment import *
from utils.options import args_parser

args = args_parser()
def main():
    env = fed()
    dir=args.ckpt_dir+args.dataset+'/'
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=env.observation_space,
                action_dim=env.action_space, actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=dir, gamma=0.99,
                tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=100000, batch_size=128)

    create_directory(path=dir, sub_path_list=['Actor', 'Critic1', 'Critic2', 'Target_actor',
                                                        'Target_critic1', 'Target_critic2'])
    #agent.load_models(30)
    total_reward_history = []
    avg_reward_history = []
    ac=[]
    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        count = 0
        while not done:
            count+=1
            print(count)
            action_set=[]
            for i in range(20):
                action = agent.choose_action(np.array(observation[i]), train=True)
                action_ = scale_action(action, low=env.action_space_low, high=env.action_space_high)
                action_set.append(action_)
            observation_, reward, done,acc= env.step(np.array(action_set))
            ac.append(acc.item())
            for i in range(20):
                agent.remember(observation[i], action_set[i], reward[i], observation_[i], done)
                agent.learn()
            total_reward += sum(reward)
            observation = observation_
            if count>=35:
                break
        total_reward_history.append(total_reward)
        avg_reward = np.mean(total_reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {} AvgReward: {}'.format(episode + 1, total_reward, avg_reward))
        print(ac)
        print(total_reward_history)
        print(avg_reward_history)
        if (episode + 1) % 9 == 0:
            agent.save_models(episode + 1)

    episodes = [i + 1 for i in range(args.max_episodes)]



if __name__ == '__main__':
    main()