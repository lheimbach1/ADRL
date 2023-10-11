import argparse
import sys
sys.path.append('../environment')
sys.path.append('../network')
from neural_net import CNN_Net
import environment
import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Simulation-predict_discrete_filters')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512, 128])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.02)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)

    net_actor = CNN_Net(state_shape = args.state_shape, device=args.device)
    actor = Actor(net_actor, action_shape = args.action_shape, hidden_sizes= args.hidden_sizes, device=args.device).to(args.device)

    net_critic = CNN_Net(state_shape = args.state_shape, device=args.device)
    critic = Critic(net_critic, hidden_sizes= args.hidden_sizes, device=args.device).to(args.device)

    actor_critic = ActorCritic(actor, critic)
    
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    dist = torch.distributions.Categorical

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
    )


    saved_state_dict = torch.load("../models/policy_network.pth", map_location=torch.device('cpu'))
    policy.load_state_dict(saved_state_dict)
    env = DummyVectorEnv([lambda: gym.make(args.task)])
    policy.eval()

    collector = Collector(policy, env)     
    result = collector.collect(n_episode=24,render=0.01)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

test_ppo()