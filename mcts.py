import math
import random
import logging
from PIL import Image
from browser_env import ActionTypes, create_stop_action
from browser_env.actions import is_equivalent
from browser_env.helper_functions import get_action_description
from agent import value_function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.screenshot = None

class MCTS:
    def __init__(self, root_state, agent, intent, images, meta_data, branching_factor, max_depth, iterations, exploration, env, config_file, action_history):
        self.root = MCTSNode(root_state)
        self.agent = agent
        self.intent = intent
        self.images = images
        self.meta_data = meta_data
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.iterations = iterations
        self.exploration = exploration
        self.env = env
        self.config_file = config_file
        self.action_history = action_history
        self.early_stop_thresholds = {
            "parsing_failure": 3,
            "repeating_action": 5
        }

    def search(self):
        logger.info("Starting MCTS search")
        for i in range(self.iterations):
            logger.info(f"Iteration {i+1}/{self.iterations}")
            node = self.select(self.root)
            score = self.simulate(node)
            self.backpropagate(node, score)
        best_action, best_score = self.best_action(self.root)
        logger.info(f"MCTS search completed. Best action: {best_action}, Score: {best_score}")
        return best_action, best_score

    def select(self, node):
        logger.info(f"Selecting node. Current depth: {len(node.state) // 2}")
        while node.children:
            if len(node.children) < self.branching_factor:
                logger.info("Node not fully expanded. Expanding.")
                return self.expand(node)
            node = self.ucb_select(node)
        logger.info("Reached leaf node. Expanding.")
        return self.expand(node)

    def expand(self, node):
        logger.info(f"Expanding node at depth {len(node.state) // 2}")
        if len(node.state) // 2 >= self.max_depth:
            logger.info("Reached maximum depth. Returning without expansion.")
            return node

        self.reset_environment(node)  # Reset environment before expansion

        actions = self.agent.next_action(node.state, self.intent, images=self.images,
                                         meta_data=self.meta_data, branching_factor=self.branching_factor)
        
        if actions is None:
            logger.warning(f"next_action returned None for state: {node.state}")
            return node

        logger.info(f"Generated {len(actions)} possible actions")
        for action in actions:
            self.reset_environment(node)  # Reset environment before each child creation
            obs, _, terminated, _, info = self.env.step(action)
            new_screenshot = Image.fromarray(obs["image"])
            new_state = node.state + [action, {"observation": obs, "info": info, "url": self.env.page.url}]
            child = MCTSNode(new_state, node, action)
            child.screenshot = new_screenshot
            node.children.append(child)
            if terminated:
                break

        return random.choice(node.children) if node.children else node

    def simulate(self, node):
        self.reset_environment(node)
        logger.info("STARTING SIMULATION")
        
        state = node.state.copy()
        depth = len(state) // 2
        screenshots = [node.screenshot] if node.screenshot else []

        while depth < self.max_depth:
            early_stop_flag, stop_info = self.early_stop(state)
            if early_stop_flag:
                break

            actions = self.agent.next_action(state, self.intent, images=self.images,
                                            meta_data=self.meta_data, branching_factor=self.branching_factor)
            if not actions:
                break
            action = random.choice(actions)
            obs, _, terminated, _, info = self.env.step(action)
            screenshots.append(Image.fromarray(obs["image"]))
            state = state + [action, {"observation": obs, "info": info, "url": self.env.page.url}]
            depth += 1
            if terminated:
                break

        score = self.evaluate(state, screenshots)
        return score

    def reset_environment(self, node):
        logger.info("Resetting environment")
        _ = self.env.reset(options={"config_file": self.config_file})
        # Replay action history
        for a_hist in self.action_history:
            self.env.step(a_hist)
        # Replay actions in current node
        for a in node.state[1::2]:
            self.env.step(a)
        logger.info("Environment reset complete")

    def backpropagate(self, node, score):
        logger.info("Starting backpropagation")
        while node:
            node.visits += 1
            node.value += score
            logger.info(f"Updated node: visits={node.visits}, value={node.value}")
            node = node.parent
        logger.info("Backpropagation complete")

    def ucb_select(self, node):
        logger.info("Performing UCB selection")
        
        # Ensure all children are visited at least once
        unvisited = [c for c in node.children if c.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        
        # UCB calculation
        def ucb_value(child):
            exploitation = child.value / child.visits
            exploration = self.exploration * math.sqrt(math.log(node.visits) / child.visits)
            return exploitation + exploration

        selected = max(node.children, key=ucb_value)
        
        logger.info(f"Selected child node with UCB value: {ucb_value(selected)}")
        return selected


    def best_action(self, node):
        logger.info("Selecting best action")
        best = max(node.children, key=lambda c: c.visits)
        logger.info(f"Best action selected with {best.visits} visits")
        return best.action, best.value / best.visits

    def evaluate(self, state, screenshots):
        return value_function.evaluate_success(
            screenshots=screenshots[-(self.max_depth+1):],
            actions=[get_action_description(a, s['info']['observation_metadata'], self.agent.action_set_tag, prompt_constructor=self.agent.prompt_constructor) 
                     for a, s in zip(state[1::2], state[2::2])],
            current_url=state[-1]['url'],
            last_reasoning=state[-2]['raw_prediction'] if len(state) > 1 else "",
            intent=self.intent,
            models=["gpt-4o-2024-05-13"],
            intent_images=self.images if len(self.images) > 0 else None
        )

    def early_stop(self, trajectory):
        num_steps = (len(trajectory) - 1) / 2
        if num_steps >= self.max_depth:
            return True, f"Reach max steps {self.max_depth}"

        last_k_actions = trajectory[1::2][-self.early_stop_thresholds["parsing_failure"]:]
        if len(last_k_actions) >= self.early_stop_thresholds["parsing_failure"]:
            if all([action["action_type"] == ActionTypes.NONE for action in last_k_actions]):
                return True, f"Failed to parse actions for {self.early_stop_thresholds['parsing_failure']} times"

        last_k_actions = trajectory[1::2][-self.early_stop_thresholds["repeating_action"]:]
        action_seq = trajectory[1::2]

        if len(action_seq) == 0:
            return False, ""

        last_action = action_seq[-1]

        if last_action["action_type"] != ActionTypes.TYPE:
            if len(last_k_actions) >= self.early_stop_thresholds["repeating_action"]:
                if all([is_equivalent(action, last_action) for action in last_k_actions]):
                    return True, f"Same action for {self.early_stop_thresholds['repeating_action']} times"
        else:
            if sum([is_equivalent(action, last_action) for action in action_seq]) >= self.early_stop_thresholds["repeating_action"]:
                return True, f"Same typing action for {self.early_stop_thresholds['repeating_action']} times"

        return False, ""