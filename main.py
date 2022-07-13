from Orchestrator import Orchestrator

orchestrator = Orchestrator()

if __name__ == "__main__":
    user_input = input("Cloth number from dataset to predict: ")
    user_output = orchestrator.run(user_input)
    print(user_output)
