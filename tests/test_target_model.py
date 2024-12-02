import unittest
from conversers import TargetLM



class TestJBBTargetVicuna(unittest.TestCase):
    def setUp(self):
        self.lm = TargetLM(model_name="vicuna-13b-v1.5",
            category="unittest",
            max_n_tokens=10,
            phase="dev",
            evaluate_locally = False,
            use_jailbreakbench = True)

    def test_get_response(self):
        prompts_list=["What is the meaning of life?", "Write a poem about the ocean."]
        correct_responses = ["The meaning of life is a philosophical question that", 
                             "The ocean, vast and blue,\nA body"]
        num_prompts = len(prompts_list)
        responses = self.lm.get_response(prompts_list)
                                          
        self.assertEqual(len(responses), num_prompts)
        self.assertEqual(responses,correct_responses) 

class TestJBBTargetGPT35(unittest.TestCase):
    def setUp(self):
        self.lm = TargetLM(model_name="gpt-3.5-turbo-1106",
            category="unittest",
            max_n_tokens=10,
            phase="dev",
            evaluate_locally = False,
            use_jailbreakbench = True)

    def test_get_response(self):
        prompts_list=["What is the meaning of life?", "Write a poem about the ocean."]
        correct_responses = ['The meaning of life is a deeply philosophical and personal', 'Oh mighty ocean, vast and deep\nYour waves']
        num_prompts = len(prompts_list)
        responses = self.lm.get_response(prompts_list)
                                          
        self.assertEqual(len(responses), num_prompts)
        self.assertEqual(responses,correct_responses) 


