import unittest
from conversers import TargetLM



class TestJBBTarget(unittest.TestCase):
    def setUp(self):
        self.lm = TargetLM(model_name="vicuna",
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

        #prompt_tokens = token_counts["prompt_tokens"]
        #completion_tokens = token_counts["completion_tokens"]        
        #print(responses, prompt_tokens, completion_tokens)                                               
        self.assertEqual(len(responses), num_prompts)
        #self.assertEqual(prompt_tokens, [41, 41]) # This is much larger than the prompt since it also contains the system prompt
        #self.assertEqual(completion_tokens, [9, 9])
        self.assertEqual(responses,correct_responses) 


