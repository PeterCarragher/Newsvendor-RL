# Get the code
Download from: https://drive.google.com/drive/folders/1sLlfq-9FZqvn-S0VNlhR8ddiyBB1muKv?usp=sharing

# Custom Environment Setup
```
pip3 install gym
cd <download_directory>/or-gym
pip3 install -e .
```

# Running the experiments
For the newsvendor experiments, run the newsvendor.ipynb notebook.
For the supply chain experiments, run the supply_chain.ipynb notebook.

A reproduction of the PPO experiments on supply chains with lead times from Perez et al. is included in imo_ppo.ipynb and newsvendor_with_lead_ppo.ipynb. 
As these were not comparable to the environments where MDPs could be generated and generalised policy methods applied, they are considered out of scope of the report. 

# PPO
Custom metrics were implemented to track various inventories to help explain what PPO learns.
To investigate these during or after training the PPO policy, run:
```
tensorboard --logdir "/home/peter/ray_results"
```