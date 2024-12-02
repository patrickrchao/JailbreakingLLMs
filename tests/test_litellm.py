import unittest
from language_models import APILiteLLM

class TestVicuna(unittest.TestCase):
    def setUp(self):
        self.lm = APILiteLLM("vicuna-13b-v1.5")

    def test_get_response(self):
        convs_list = [
                        [{"role": "user", "content": "Hi how are you?"},
                        {"role": "assistant", "content": "I'm good, how are you?</s>"},
                        {"role": "user", "content": "I'm great. Please write a poem about the ocean."}],

                        [{"role": "user", "content": "What is 1+1?"},
                        {"role": "assistant", "content": "1+1 is equal to 2.</s>"},
                        {"role": "user", "content": "Great, repeat back my first message."}]
                     ]
        
        correct_responses = [
                             "The ocean, a vast and endless sea\n",
                             "What is 1+1?"]
        
        num_convs = len(convs_list)
        responses = self.lm.batched_generate(convs_list, 
                                            max_n_tokens = 10, 
                                            temperature = 0,
                                            top_p=0.9)
        self.assertEqual(len(responses), num_convs)
        self.assertEqual(responses,correct_responses)

    def test_initialized_response(self):
        convs_list = [
                        [{"role": "user", "content": "What is the meaning of life?"},
                        {"role": "assistant", "content": "I don't know, instead, let's count to 10: 1,2,"}],
                        [{"role": "user", "content": "Write a poem about skateboarding in the ocean."},
                         {"role": "assistant", "content": "I'm bad at writing poems, but here goes: It was a dark and stormy night."}]
                    ]
        correct_responses = [
                             "3,4,5,6,7,",
                             "The waves were crashing, the wind was how"] 
        num_convs = len(convs_list)
        responses = self.lm.batched_generate(convs_list, 
                                                max_n_tokens = 10, 
                                                temperature = 0,
                                                top_p=0.9)
        self.assertEqual(len(responses), num_convs)
        self.assertEqual(responses,correct_responses)




# class TestGPT4(unittest.TestCase):
#     def setUp(self):
#         self.lm = APILiteLLM("gpt-4-0125-preview")

#     def test_get_response(self):
#         convs_list = [[{"role": "user", "content": "What is the meaning of life?"}],
#                       [{"role": "user", "content": "Write a poem about the ocean."}]]
#         correct_responses = ["The question of the meaning of life is one of", 
#                              "Beneath the vast, eternal sky,\nWhere"]
#         num_convs = len(convs_list)
#         responses = self.lm.batched_generate(convs_list, 
#                                             max_n_tokens = 10, 
#                                             temperature = 0,
#                                             top_p=0.9)
#         #prompt_tokens = token_counts["prompt_tokens"]
#         #completion_tokens = token_counts["completion_tokens"]  
#         self.assertEqual(len(responses), num_convs)
#         #self.assertEqual(prompt_tokens, [14, 14])
#         #self.assertEqual(completion_tokens, [10, 10])
#         self.assertEqual(responses,correct_responses) # May fail due to randomness 



