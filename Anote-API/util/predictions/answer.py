
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

def generate_answers(question,
                     context,
                     model_type="Claude",
                     private=False):
    if private:
        # LLAMA-2 Q-A logic
        base_model_name = "meta-llama/Llama-2-7b-chat-hf"
        # Tokenizer
        llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

        #model inference using prompt template
        system_prompt = "You are a chatbot trained to answer questions based on the given context. If you do not know the answer to the question, say I do not know\n"
        user_prompt = question
        input_section = context
        text_gen = pipeline(task="text-generation", model = base_model_name, tokenizer=llama_tokenizer)
        output = text_gen(f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} {input_section} [/INST]")
        generated_text = output[0]['generated_text']

        # Extracting the part of the text after [/INST]
        response_start_idx = generated_text.find("[/INST]") + len("[/INST]")
        model_response = generated_text[response_start_idx:].strip()

        return model_response

    if model_type == 'GPT':
        openai.api_key = "Enter Your Api Key"
        system_content = "You are a chatbot trained to answer questions based on the given context. If you do not know the answer to the question, say I do not know\n."

        output = []

        completion = openai.ChatCompletion.create(
            model= "gpt-4",
            messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"{question}" }
            ])

        output.append(completion.choices[0].message.content)

    elif model_type == 'Claude':
        claude_key = "Enter API Key"
        anthropic = Anthropic(

        api_key=claude_key,
        )

        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=700,
            prompt = (
                    f"{HUMAN_PROMPT} "
                    f"You are a chatbot trained to answer questions based on the given context. If you do not know the answer to the question, say I do not know\n "
                    f"please address the question: {question}. "
                    f"Consider the provided text as evidence: {context}. "
                    f"{AI_PROMPT}") )

        output.append(completion.completion)

    return output