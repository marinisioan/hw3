"""
Unified Deep RL Visualization Suite
=============================================================================
Αυτό το script αναλαμβάνει την οπτικοποίηση των Deep RL πρακτόρων (Agents)
που εκπαιδεύτηκαν κατά τη διάρκεια της εργασίας. 

Οδηγίες Εκτέλεσης (από το τερματικό):
-----------------------------------------------------------------------------
1. Standard DQN (Task 2):
   > python Visualization.py --agent dqn_full

2. Double DQN (Bonus B):
   > python Visualization.py --agent dqn_double

3. REINFORCE (Task 3):
   > python Visualization.py --agent reinforce

4. A2C Discrete (Task 3):
   > python Visualization.py --agent a2c

5. Continuous A2C (Bonus C):
   > python Visualization.py --agent continuous_a2c
=============================================================================
"""

import sys, os, argparse
import gymnasium as gym
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Unified Deep RL Visualization Suite")
    parser.add_argument("--agent", type=str, required=True,
                        choices=["dqn_full", "dqn_double", "reinforce", "a2c", "continuous_a2c"],
                        help="Επίλεξε τον Deep RL αλγόριθμο προς οπτικοποίηση.")
    return parser.parse_args()

def run_deep_rl(agent_name):
    checkpoint_path = os.path.join(BASE_DIR, "experiments", "checkpoints", f"{agent_name}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[ΣΦΑΛΜΑ] Δεν βρέθηκαν τα βάρη: {checkpoint_path}")
        return

    env_name = "LunarLanderContinuous-v3" if agent_name == "continuous_a2c" else "LunarLander-v3"
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]

    print(f"\n── Φόρτωση: {agent_name} στο {env_name} ──")

    if "dqn" in agent_name:
        from agents.dqn import DQNAgent
        mode = "double" if "double" in agent_name else "full"
        agent = DQNAgent(state_dim, env.action_space.n, mode=mode)
        agent.eps = 0.0
    elif agent_name == "a2c":
        from agents.a2c import A2CAgent
        agent = A2CAgent(state_dim, env.action_space.n)
    elif agent_name == "reinforce":
        from agents.reinforce import REINFORCEAgent
        agent = REINFORCEAgent(state_dim, env.action_space.n)
    elif agent_name == "continuous_a2c":
        from agents.continuous_a2c import ContinuousA2CAgent
        agent = ContinuousA2CAgent(state_dim, env.action_space.shape[0])

    agent.load(checkpoint_path)
    
    if hasattr(agent, "online_net"): agent.online_net.eval()
    if hasattr(agent, "ac_net"): agent.ac_net.eval()
    if hasattr(agent, "policy_net"): agent.policy_net.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0

    with torch.no_grad():
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

    print(f"Τελική Ανταμοιβή: {total_reward:.1f}")
    env.close()

def main():
    args = parse_arguments()
    # Εκτελείται πλέον απευθείας η ρουτίνα των Deep RL
    run_deep_rl(args.agent)

if __name__ == "__main__":
    main()