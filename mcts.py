import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS:
    def __init__(self, root_state, agent, intent, images, meta_data, branching_factor, max_depth, iterations, exploration):
        self.root = MCTSNode(root_state)
        self.agent = agent
        self.intent = intent
        self.images = images
        self.meta_data = meta_data
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.iterations = iterations
        self.exploration = exploration

    def search(self):
        for _ in range(self.iterations):
            node = self.select(self.root)
            score = self.simulate(node)
            self.backpropagate(node, score)
        return self.best_action(self.root)

    def select(self, node):
        while node.children:
            if len(node.children) < self.branching_factor:
                return self.expand(node)
            node = self.ucb_select(node)
        return self.expand(node)

    def expand(self, node):
        if len(node.state) // 2 >= self.max_depth:
            return node
        actions = self.agent.next_action(
            node.state,
            self.intent,
            images=self.images,
            meta_data=self.meta_data,
            branching_factor=self.branching_factor
        )
        for action in actions:
            new_state = node.state + [action, {"observation": None, "info": None, "url": None}]
            child = MCTSNode(new_state, node, action)
            node.children.append(child)
        return random.choice(node.children)

    def simulate(self, node):
        state = node.state
        depth = len(state) // 2
        while depth < self.max_depth:
            actions = self.agent.next_action(
                state,
                self.intent,
                images=self.images,
                meta_data=self.meta_data,
                branching_factor=self.branching_factor
            )
            if not actions:
                break
            action = random.choice(actions)
            state = state + [action, {"observation": None, "info": None, "url": None}]
            depth += 1
        return self.evaluate(state)

    def backpropagate(self, node, score):
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    def ucb_select(self, node):
        return max(node.children, key=lambda c: c.value / c.visits + self.exploration * math.sqrt(math.log(node.visits) / c.visits))

    def best_action(self, node):
        return max(node.children, key=lambda c: c.visits).action

    def evaluate(self, state):
        # This should be replaced with the actual evaluation function
        return random.random()