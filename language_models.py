import os 
import litellm
from config import (
    API_BASE_ENV_NAMES,
    LITELLM_TEMPLATES,
    OPENAI_COMPATIBLE_MODEL_NAMES,
    TOGETHER_MODEL_NAMES,
    Model,
)
from loggers import logger
from common import get_api_key

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        self.api_base = self.get_api_base(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        elif model_name in OPENAI_COMPATIBLE_MODEL_NAMES:
            litellm_name = OPENAI_COMPATIBLE_MODEL_NAMES[model_name]
            self.use_open_source_model = False
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name

    def get_api_base(self, model_name):
        environment_variable = API_BASE_ENV_NAMES.get(model_name)
        if environment_variable is None:
            return None
        try:
            return os.environ[environment_variable]
        except KeyError:
            raise ValueError(
                f"Missing API base URL for {model_name.value}. "
                f"Please set {environment_variable}."
            )
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = list(self.eos_tokens)

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()
        
        completion_kwargs = {}
        if self.api_base is not None:
            completion_kwargs["api_base"] = self.api_base

        # Some gateway-hosted models (e.g. Claude via Vertex AI) reject requests
        # that specify both `temperature` and `top_p` ("cannot both be specified
        # for this model"). For those, send only `temperature`.
        if "claude" not in self.litellm_model_name:
            completion_kwargs["top_p"] = top_p

        outputs = litellm.batch_completion(
            model=self.litellm_model_name,
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,
            **completion_kwargs,
        )
        
        responses = []
        for output in outputs:
            # litellm.batch_completion returns the raised exception object in-place
            # for any request that failed all retries (e.g. a transient gateway 500
            # on one stream). Don't let a single bad stream crash the whole behavior.
            if isinstance(output, Exception):
                logger.warning(f"API call failed for one stream: {output}")
                responses.append(self.API_ERROR_OUTPUT)
                continue
            content = output["choices"][0]["message"].content
            # Reasoning models (e.g. deepseek) can return content=None when the
            # token budget is consumed by hidden reasoning. Coerce to an error
            # string so downstream JSON parsing fails gracefully and retries,
            # rather than crashing on a None concatenation.
            if content is None:
                logger.warning("API returned empty content for one stream.")
                responses.append(self.API_ERROR_OUTPUT)
            else:
                responses.append(content)

        return responses

# class LocalvLLM(LanguageModel):
    
#     def __init__(self, model_name: str):
#         """Initializes the LLMHuggingFace with the specified model name."""
#         super().__init__(model_name)
#         if self.model_name not in MODEL_NAMES:
#             raise ValueError(f"Invalid model name: {model_name}")
#         self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
#         destroy_model_parallel()
#         self.model = vllm.LLM(model=self.hf_model_name)
#         if self.temperature > 0:
#             self.sampling_params = vllm.SamplingParams(
#                 temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
#             )
#         else:
#             self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

#     def _get_responses(self, prompts_list: list[str], max_new_tokens: int | None = None) -> list[str]:
#         """Generates responses from the model for the given prompts."""
#         full_prompt_list = self._prompt_to_conv(prompts_list)
#         outputs = self.model.generate(full_prompt_list, self.sampling_params)
#         # Get output from each input, but remove initial space
#         outputs_list = [output.outputs[0].text[1:] for output in outputs]
#         return outputs_list

#     def _prompt_to_conv(self, prompts_list):
#         batchsize = len(prompts_list)
#         convs_list = [self._init_conv_template() for _ in range(batchsize)]
#         full_prompts = []
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()
#             # Need this to avoid extraneous space in generation
#             if "llama-2-7b-chat-hf" in self.model_name:
#                 full_prompt += " "
#             full_prompts.append(full_prompt)
#         return full_prompts

#     def _init_conv_template(self):
#         template = get_conversation_template(self.hf_model_name)
#         if "llama" in self.hf_model_name:
#             # Add the system prompt for Llama as FastChat does not include it
#             template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         return template
    





