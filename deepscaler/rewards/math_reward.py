"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import re

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

    
class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response

        format_score = self.format_reward(model_response)
        model_answer = extract_answer(model_response)

        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward+format_score, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward+format_score, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                if format_score >= self.config.format_correct_reward:
                    return RewardOutput(reward=self.config.correct_reward+format_score, is_correct=True)
                else:
                    return RewardOutput(reward=self.config.incorrect_reward+format_score, is_correct=True)

        return RewardOutput(reward=self.config.incorrect_reward+format_score, is_correct=False)
    

    # def format_reward(self, response, **kwargs):
    #     #pattern =  r"^<intuition>.*?</intuition>\n<think>.*?</think>.*$"
    #     pattern = r".*?</intuition>\n<think>.*?</think>.*$"
    #     match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
    #     if match:
    #         return self.config.format_correct_reward
    #     else:
    #         return self.config.format_error_reward
        

    # def format_reward(self, response, **kwargs):
    #     #pattern =  r"^<intuition>.*?</intuition>\n<think>.*?</think>.*$"
    #     pattern = r".*?</intuition>\n<think>.*?</think>.*$"
    #     match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
       
    #     if match:
    #         return self.config.format_correct_reward
    #     else:
    #         format_reward = 0
    #         if "</intuition>" in response:
    #             format_reward += self.config.format_step_reward
    #         if "<think>" in response:
    #             format_reward += self.config.format_step_reward 
    #         return format_reward
        

    def format_reward(self, response, **kwargs):
        #pattern =  r"^<intuition>.*?</intuition>\n<think>.*?</think>.*$"
        pattern = r".*?</intuition>\n<think>.*?</think>.*$"
        tags = {
            'intuition_end': ('</intuition>', 1),
            'think_start': ('<think>', 1),
        }
        match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
       
        if match:
            return self.config.format_correct_reward
        else:
            format_reward = 0
            for tag_name, (tag_str, expected_count) in tags.items():
                count = response.count(tag_str)
                if count == expected_count:
                    format_reward += self.config.format_step_reward
            return format_reward
        
        

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.reward, reward_response.is_correct

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(
        problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", 
        problem_type=RewardType.MATH, 
        model_response="<intuition>I am omniscient.\n</intuition>\n<think>I am omniscient. \n</think>The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", 
        ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
