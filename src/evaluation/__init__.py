"""
Evaluation Pipeline — Strategy A: Sim-to-Real Transfer Evaluation.

Evaluates a pre-trained SCRIPT agent on PenGym NASim environment by:
1. Loading the trained model (PPO actor/critic + state normalization)
2. For each episode: converting PenGym obs → SCRIPT state via StateAdapter
3. Selecting actions via the policy, mapping them via ActionMapper
4. Collecting comprehensive metrics and gap analysis
"""
