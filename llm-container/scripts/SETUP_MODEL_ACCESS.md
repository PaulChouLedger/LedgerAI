# Setting Up HuggingFace Access for Llama Models

## Step 1: Get HuggingFace Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Choose "Read" access (sufficient for downloading)
4. Copy the token (you'll need it)

## Step 2: Request Model Access

1. Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Click "Agree and access repository" or "Request access"
3. Accept the terms if prompted
4. Wait for access approval (usually instant for Llama 3.2-1B)

## Step 3: Login on Jetson

```bash
# Login using your token
hf login
# Paste your token when prompted

# Or set token as environment variable
export HF_TOKEN="your_token_here"
hf login --token $HF_TOKEN
```

## Step 4: Verify Access

```bash
# Try to access the model info
hf repo info meta-llama/Llama-3.2-1B-Instruct
```

If this works, you have access!

## Step 5: Download Model

```bash
cd ~/LedgerAI/llm-container
mkdir -p models/Llama

hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir models/Llama/Llama-3.2-1B-Instruct
```

## Alternative: Use HuggingFace Web Interface

If command-line doesn't work:

1. Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Click "Files and versions" tab
3. Download files manually or use the download button
4. Extract to: `~/LedgerAI/llm-container/models/Llama/Llama-3.2-1B-Instruct/`

## Troubleshooting

### "Access is restricted" after login
- Make sure you clicked "Agree and access repository" on the model page
- Refresh the page and try again
- Check that your account has been granted access

### "Token is invalid"
- Generate a new token at https://huggingface.co/settings/tokens
- Make sure it has "Read" permissions
- Try logging in again

### Still can't access
- Check your internet connection
- Try logging out and back in: `hf logout` then `hf login`
- Use the web interface as a workaround

