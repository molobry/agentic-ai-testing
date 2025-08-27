# Troubleshooting Guide

## Common Issues in Corporate Environments

### 1. Azure OpenAI Client Initialization Errors

#### Error: `Client.__init__() got unexpected keyword argument 'proxies'`

**Cause**: Version mismatch or corporate proxy settings

**Solutions**:
```bash
# 1. Update OpenAI package
pip install --upgrade openai

# 2. Check your current version
python -c "import openai; print(openai.__version__)"

# 3. Install specific compatible version
pip install "openai>=1.3.0,<2.0.0"
```

#### Error: Connection timeout or SSL issues

**Corporate Network Solutions**:
```bash
# 1. Set proxy environment variables
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port

# 2. Install with trusted hosts
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org openai

# 3. Configure SSL verification
export SSL_CERT_FILE=/path/to/corporate/cert.pem
```

### 2. Azure OpenAI Configuration

#### Required Environment Variables:
```env
AZURE_OPENAI_API_KEY=your_actual_azure_key
AZURE_OPENAI_ENDPOINT=https://your-company-openai.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

#### Common Endpoint Formats:
- `https://your-resource-name.openai.azure.com/`
- `https://your-company-openai.openai.azure.com/`
- Custom corporate endpoints

### 3. Testing Azure OpenAI Connection

```python
# Test script to verify Azure OpenAI connection
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
)

try:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ Azure OpenAI connection successful")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### 4. Fallback Modes

The framework automatically falls back to:
1. **Basic Element Detection** - No AI required
2. **Cached Selectors** - Uses previously successful selectors
3. **Pattern Matching** - Smart element identification

### 5. Corporate Security

#### Firewall/Proxy Issues:
- Ensure outbound HTTPS access to Azure OpenAI endpoints
- Add framework to approved software list
- Configure proxy settings in environment

#### Certificate Issues:
```bash
# Trust corporate certificates
pip install --cert /path/to/corporate/cert.pem openai
```

### 6. Version Compatibility Matrix

| OpenAI Package | Azure Support | Recommended |
|----------------|---------------|-------------|
| 0.x.x         | Legacy API    | No          |
| 1.0.x - 1.2.x | Basic         | No          |
| 1.3.x+        | Full Support  | **Yes**     |

### 7. Debug Mode

Enable detailed logging:
```bash
export DEBUG=1
python run_test.py scenarios/simple_login_test.xml --headless
```

### 8. Contact IT Support

If issues persist, provide IT with:
- Error messages from console output
- OpenAI package version: `pip show openai`
- Python version: `python --version`
- Corporate network configuration requirements
- This troubleshooting guide

## Quick Fixes

```bash
# Complete environment reset
pip uninstall openai
pip install "openai>=1.3.0"
pip install --upgrade requests urllib3

# Test basic functionality
python -c "from src.ai_analyzer import AIAnalyzer; print('✅ Import successful')"
```