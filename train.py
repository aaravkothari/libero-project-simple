from agent import Agent

agent = Agent()

agent.train(epochs=100000, batch_size=32)

agent.close()