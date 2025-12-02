import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM

#paths and constants
BASE_MODEL_PATH = "models/gpt2-qa"   
RL_OUTPUT_PATH = "models/gpt2-qa-rl" 

PREFIX = "That is a great question. "
SUFFIX = " Let me know if you have any other questions."


def compute_reward(answer_text: str) -> float:
    """Reward how well answer matches desired format."""
    text = answer_text.strip()
    reward = 0.0

    if text.startswith(PREFIX):
        reward += 1.0
    if text.endswith(SUFFIX):
        reward += 1.0

    if text.startswith(PREFIX) and text.endswith(SUFFIX):
        middle = text[len(PREFIX): len(text) - len(SUFFIX)]
        if len(middle.split()) < 3:
            reward -= 0.2

    return reward


def generate_with_logprobs(model, tokenizer, question: str, device, max_new_tokens: int = 64):
    """Generate an answer and keep token log-probs for REINFORCE."""
    model.eval()

    prompt = f"Question: {question.strip()}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = input_ids.clone()
    logprobs = []

    for _ in range(max_new_tokens):
        outputs = model(generated_ids)
        logits = outputs.logits[:, -1, :] 

        log_probs = F.log_softmax(logits, dim=-1)        
        probs = log_probs.exp()                          

        next_token = torch.multinomial(probs, num_samples=1)        
        next_logprob = log_probs.gather(1, next_token)               

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        logprobs.append(next_logprob.squeeze(0)) 

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "Answer:" in full_text:
        answer_text = full_text.split("Answer:", 1)[1].strip()
    else:
        answer_text = full_text

    logprobs = torch.cat(logprobs, dim=0)
    return answer_text, logprobs


def main():
    #loading module 9 
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-6) 

    #questionsn from class assigment model 9
    questions = [
        "What is the capital of France?",
        "Who wrote The Count of Monte Cristo?",
        "What is 2 + 2?",
        "What is the capital of Spain?",
        "What is the capital of Italy?",
    ]

    num_epochs = 3

    for epoch in range(num_epochs):
        for q in questions:
            model.train()
            optimizer.zero_grad()

            answer_text, logprobs = generate_with_logprobs(model, tokenizer, q, device)
            reward = compute_reward(answer_text)

            loss = -reward * logprobs.sum()

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Q: {q}")
            print(f"Answer: {answer_text}")
            print(f"Reward: {reward:.2f} | Loss: {loss.item():.4f}")
            print("-" * 60)

    #saving post-trained model
    model.save_pretrained(RL_OUTPUT_PATH)
    tokenizer.save_pretrained(RL_OUTPUT_PATH)
    print(f"Saved RL post-trained model to {RL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
