"""
TensorRT-LLM Model Configuration
Qwen2.5-7B-Instruct configuration for complex queries and RAG
"""

QWEN2_5_7B_CONFIG = {
    "model_name": "Qwen2.5-7B-Instruct",
    "vocab_size": 151936,
    "num_layers": 28,
    "num_heads": 32,
    "num_kv_heads": 8,  # GQA (Grouped Query Attention)
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "pad_token_id": 151643,
}

# TensorRT-LLM specific settings for Qwen2.5-7B
QWEN_TENSORRT_CONFIG = {
    "max_batch_size": 1,
    "max_input_len": 4096,  # Large context for RAG
    "max_output_len": 512,
    "dtype": "float16",
    "use_gpt_attention_plugin": True,
    "use_gemm_plugin": True,
    "use_rmsnorm_plugin": True,
    "remove_input_padding": True,
    "enable_context_fmha": True,
    "enable_context_fmha_fp32_acc": False,
}

# Sampling configuration for complex queries
QWEN_SAMPLING_CONFIG = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "length_penalty": 1.0,
    "max_new_tokens": 512,
}

"""
Llama-3.2-1B-Instruct configuration for simple tasks
"""

LLAMA3_2_1B_CONFIG = {
    "model_name": "Llama-3.2-1B-Instruct",
    "vocab_size": 128256,
    "num_layers": 16,
    "num_heads": 16,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "max_position_embeddings": 8192,
    "rope_theta": 500000.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "pad_token_id": 128002,
}

# TensorRT-LLM specific settings for Llama-3.2-1B
LLAMA_TENSORRT_CONFIG = {
    "max_batch_size": 1,
    "max_input_len": 2048,
    "max_output_len": 256,
    "dtype": "float16",
    "use_gpt_attention_plugin": True,
    "use_gemm_plugin": True,
    "use_rmsnorm_plugin": True,
    "remove_input_padding": True,
    "enable_context_fmha": True,
}

# Sampling configuration for simple tasks
LLAMA_SAMPLING_CONFIG = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.85,
    "repetition_penalty": 1.05,
    "length_penalty": 1.0,
    "max_new_tokens": 256,
}

