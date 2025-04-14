# Migration from `google-generativeai` to `google-genai`

This document outlines the key differences and migration steps taken to transition from the original `google-generativeai` SDK to the newer `google-genai` SDK (version 1.10.0+) for the PPA Agent project.

## Overview

Google has deprecated the `google-generativeai` package in favor of the new `google-genai` package. The new SDK provides improved functionality and better support for the latest Gemini models, but it has a significantly different API structure.

## Key Differences

### 1. Package Imports

```python
# Old SDK
import google.generativeai as genai

# New SDK
import google.genai as genai
```

### 2. Client Initialization

```python
# Old SDK
genai.configure(api_key="YOUR_API_KEY")

# New SDK
client = genai.Client(api_key="YOUR_API_KEY")
# The SDK will automatically pick up GOOGLE_API_KEY if api_key is None/empty
```

### 3. Content Generation

```python
# Old SDK
model = genai.GenerativeModel('model-name')
response = model.generate_content('prompt')

# New SDK
response = client.models.generate_content(
    model='model-name',
    contents='prompt'
)
```

### 4. Configuration Parameters

```python
# Old SDK - Use GenerationConfig
generation_config = genai.types.GenerationConfig(
    temperature=0.7,
    response_mime_type="application/json"
)
response = model.generate_content(
    'prompt', 
    generation_config=generation_config
)

# New SDK - Use GenerateContentConfig
config = genai.types.GenerateContentConfig(
    temperature=0.7,
    responseMimeType="application/json"
)
response = client.models.generate_content(
    model='model-name',
    contents='prompt',
    config=config
)
```

### 5. Parameter Naming Convention

The new SDK uses camelCase for parameter names rather than snake_case:

- `response_mime_type` → `responseMimeType`
- `max_output_tokens` → `maxOutputTokens`
- `stop_sequences` → `stopSequences`

### 6. Error Handling

```python
# Old SDK
try:
    response = model.generate_content('prompt')
except genai.types.BlockedPromptError as e:
    print(f"Prompt blocked: {e}")

# New SDK
try:
    response = client.models.generate_content(model='model-name', contents='prompt')
except genai.errors.APIError as e:
    if "blocked" in str(e).lower() or "safety" in str(e).lower():
        print(f"Prompt likely blocked: {e}")
    else:
        print(f"API error: {e}")
        raise
```

### 7. Async Functions

```python
# Old SDK - Not directly supported
# Often implemented with run_in_executor pattern

# New SDK - Native async support
response = await client.models.generate_content_async(
    model='model-name',
    contents='prompt'
)
```

## Model Names

The default model has been updated:
- Old: `gemini-1.5-flash-latest`
- New: `gemini-2.5-pro-exp-03-25`

Refer to the [Google AI documentation](https://ai.google.dev/models) for the most up-to-date model names.

## Common Issues and Troubleshooting

1. **Validation Errors**: If you see Pydantic validation errors like `Extra inputs are not permitted`, check that you're using the correct parameter names (camelCase, not snake_case) and structure for the `GenerateContentConfig`.

2. **JSON Parsing**: When using `responseMimeType="application/json"`, be prepared to handle cases where the response might include extra data. Add robust JSON error handling.

3. **Error Types**: The new SDK has different error class names. Instead of specific errors like `BlockedPromptError`, use `APIError` and check the error message text.

4. **Configuration Structure**: Configuration in the new SDK is flatter - parameters are direct fields of `GenerateContentConfig` rather than nested in a separate configuration object.

## Reference

For comprehensive documentation, refer to:
- [Google AI Developer Documentation](https://ai.google.dev/)
- [Google Genai SDK Migration Guide](https://ai.google.dev/gemini-api/docs/migrate#python) 