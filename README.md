Monte Carlo implementation for Blackjack environment from OpenAI Gym. 
This environment is in accordance with Blackjack example in 
Reinforcement Learning: An Introduction by Sutton and Barto.
The state for this environment is characterized by tuple consisting of 
player's current sum, dealer's face-up card and if user has usable ace.
The action space consists of hit and stick.  
This implementation uses on-policy Monte Carlo algorithm with epsilon-
soft policy