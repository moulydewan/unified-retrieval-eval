import boto3
import json
import logging
import time
import random
from typing import Union, List, Optional

logger = logging.getLogger(__name__)
_bedrock_clients: dict = {}


def get_bedrock_client(region: str = "us-east-1"):
    """Initialize and cache AWS Bedrock runtime client."""
    if region not in _bedrock_clients:
        logger.info(f"Creating new Bedrock runtime client for region: {region}")
        _bedrock_clients[region] = boto3.client("bedrock-runtime", region_name=region)
    return _bedrock_clients[region]


def generate_bedrock_model(
    model_id: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: Optional[float] = 1.0,
    n: int = 1,
    max_tokens: int = 512,
    system_prompt: Optional[str] = None,
    region: str = "us-east-1",
    **kwargs,
) -> Union[str, List[str]]:
    """
    Generation function for Bedrock (supports Claude 3.x and 4.x, GPT oss 20b and 120b).
    Detects which API to call:
      - Claude 3.x models use invoke_model()
      - Claude 4.x models use converse()
      - GPT oss 20b, 120b models use invoke_model()
    """

    client = get_bedrock_client(region)
    results = []

    # Detect whether this is Claude 3.x or 4.x, GPT oss 20b, 120b
    is_claude_3 = any(
        key in model_id.lower()
        for key in ["claude-3", "sonnet-3", "haiku-3", "2024"]
    )

    is_claude_4 = any(
        key in model_id.lower()
        for key in ["claude-sonnet-4", "claude-haiku-4"]
    )

    is_gpt_oss = "gpt-oss" in model_id.lower()

    if is_claude_3:
        model_type = "Claude 3.x"
    elif is_claude_4:
        model_type = "Claude 4.x"
    elif is_gpt_oss:
        model_type = "GPT-OSS (20B/120B)"
    else:
        model_type = "Unknown"

    logger.info(f"Using {model_type} generation for model: {model_id}")

    for i in range(n):
        retries = 0
        max_retries = 6

        while retries <= max_retries:
            try:
                if is_claude_3:
                    # Claude 3.x (Invoke Model API)
                    messages = []
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}],
                        })
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    })

                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "messages": messages,
                    }

                    response = client.invoke_model(modelId=model_id, body=json.dumps(body))
                    result = json.loads(response["body"].read())
                    text = result["content"][0]["text"].strip()

                elif is_claude_4:
                    # Claude 4.x (Converse API)
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": [{"text": system_prompt}]})
                    messages.append({"role": "user", "content": [{"text": prompt}]})

                    inference_config = {"maxTokens": max_tokens}
                    if top_p is not None:
                        inference_config["topP"] = top_p
                    else:
                        inference_config["temperature"] = temperature

                    response = client.converse(
                        modelId=model_id,
                        messages=messages,
                        inferenceConfig=inference_config,
                    )

                    outputs = response.get("output", {}).get("message", {}).get("content", [])
                    text = outputs[0].get("text", "").strip() if outputs else ""

                elif is_gpt_oss:
                    # OpenAI GPT-OSS (20B or 120B) via Bedrock
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    body = {
                        "messages": messages,
                        "max_completion_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }

                    response = client.invoke_model(modelId=model_id, body=json.dumps(body))
                    result = json.loads(response["body"].read())
                    text = result["choices"][0]["message"]["content"].strip()

                else:
                    raise ValueError(f"Unknown AWS model type: {model_id}")

                results.append(text)
                break

            except Exception as e:
                if "ThrottlingException" in str(e) or "Too many requests" in str(e):
                    wait_time = min(90, 2 ** retries + random.random() * 2)
                    logger.warning(
                        f"Rate limit hit (attempt {retries+1}/{max_retries}). "
                        f"Sleeping {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"Claude generation failed: {e}")
                    raise e

        else:
            raise RuntimeError(f"Max retries exceeded for model {model_id}")

        if n > 1:
            time.sleep(0.1)

    return results[0] if n == 1 else results


def test_bedrock_connection():
    """Quick connectivity test for both Claude 3.5 and 4.5 on Bedrock."""
    try:
        # # Claude 3.5 Sonnet
        # result_sonnet_35 = generate_bedrock_model(
        #     "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0", "Hello from 3.5", max_tokens=10
        # )
        # print("Claude 3.5 Sonnet output:", result_sonnet_35)

        #  # Claude 3.5 Haiku
        # result_haiku_35 = generate_bedrock_model(
        #     "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:", "Hello from 3.5", max_tokens=10
        # )
        # print("Claude 3.5 Haiku output:", result_haiku_35)

        # Claude 4.5 Sonnet
        model_arn_sonnet_45 = (
           "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )
        result_sonnet_45 = generate_bedrock_model(
            model_arn_sonnet_45, "Hello from 4.5", max_tokens=10)
        print("Claude 4.5 Sonnet output:", result_sonnet_45)

        # Claude 4.5 Haiku
        model_arn_haiku_45 = (
            "arn:aws:bedrock:us-east-1:033792130535:inference-profile/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        result_haiku_45 = generate_bedrock_model(model_arn_haiku_45, "Hello from Haiku 4.5!", max_tokens=10)
        print("Claude 4.5 Haiku output:", result_haiku_45)

        # GPT-OSS 20B
        model_id_gpt_oss_20b = "openai.gpt-oss-20b-1:0"
        result_gpt_oss_20b = generate_bedrock_model(model_id_gpt_oss_20b, "Hello from GPT-OSS 20B!", max_tokens=10)
        print("GPT-OSS 20B output:", result_gpt_oss_20b)

        # GPT-OSS 120B
        model_id_gpt_oss_120b = "openai.gpt-oss-120b-1:0"
        result_gpt_oss_120b = generate_bedrock_model(model_id_gpt_oss_120b, "Hello from GPT-OSS 120B!", max_tokens=10)
        print("GPT-OSS 120B output:", result_gpt_oss_120b)

        logger.info("Claude Bedrock connection successful.")
        return True

    except Exception as e:
        logger.error(f"Claude test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Bedrock backend...")
    success = test_bedrock_connection()
    if success:
        print("Connection successful!")
    else:
        print("Connection failed.")
