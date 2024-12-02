import unittest
from conversers import AttackLM
from fastchat.model import get_conversation_template

class TestTogetherAttacker(unittest.TestCase):
    def setUp(self):
        self.lm = AttackLM(model_name="vicuna-13b-v1.5",
                max_n_tokens=100, 
                max_n_attack_attempts=1, 
                category="unittest",
                evaluate_locally=False)

    def test_preprocess(self):
        
        conv = get_conversation_template("vicuna-13b-v1.5") 

        conv.system_message = "Write in proper JSON format with keys `improvement` and `attack`."
        prompt = "Create a JSON with the key-values (improvement,`3`) and (prompt,`5`) and nothing else."
        conv_list = [conv]
        
        processed, init_message = self.lm.preprocess_conversation(conv_list, [prompt])
        self.assertEqual(processed[0][-1],{'role': 'assistant', 'content': '{"improvement": "","prompt": "'})
        self.assertEqual(init_message,'{"improvement": "","prompt": "')

    def test_get_attack_empty(self):
        conv = get_conversation_template("vicuna-13b-v1.5") 

        conv.system_message = "Write in proper JSON format with keys `improvement` and `attack`."
        prompt = "Create a JSON with the key-values (improvement,``) and (prompt,`5`) and nothing else."
        conv_list = [conv]
        attack_dict = self.lm.get_attack(conv_list, [prompt])

        self.assertEqual(conv_list[0].messages[-1][1],'{"improvement": "","prompt": "5"}</s>')
        self.assertEqual(attack_dict, [{'improvement': '','prompt': '5'}])

    def test_get_attack(self):
        conv = get_conversation_template("vicuna-13b-v1.5") 

        conv.system_message = "Write in proper JSON format with keys `improvement` and `attack`."
        message1 = "Create a JSON with the key-values (improvement,`3`) and (prompt,`5`) and nothing else."
        conv.append_message(conv.roles[0], message1)
        conv.append_message(conv.roles[1], "{'improvement': '3','prompt': '5'}</s>")
        
        prompt = "Create a JSON with the key-values (improvement,`431`) and (prompt,`999`) and nothing else."
        conv_list = [conv]
        attack_dict = self.lm.get_attack(conv_list, [prompt])

        self.assertEqual(conv_list[0].messages[-1][1],'{"improvement": "431", "prompt": "999"}</s>')
        self.assertEqual(attack_dict, [{'improvement': '431','prompt': '999'}])
      


