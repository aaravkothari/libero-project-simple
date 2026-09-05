from agent import Agent

agent = Agent()

agent.train(epochs=100000, batch_size=32)

agent.close()

# ==========================================================================
# ORIGINAL train.py -- 10,000 iterations
# ==========================================================================
# from agent import Agent
#
# agent = Agent()
#
# agent.train(epochs=10000, batch_size=32)
#
# agent.close()
