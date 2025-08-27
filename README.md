# Agentic AI Web Testing Framework

An intelligent BDD-driven testing framework that uses AI to analyze web pages and automatically execute test scenarios.

## Core Concept

This framework acts as an **agentic AI** that can:
1. Read BDD scenarios (Given-When-Then)
2. Intelligently analyze web pages using DOM + AI
3. Execute tests automatically with Playwright
4. Learn and cache successful element selectors
5. Generate comprehensive test reports with screenshots

## Technology Stack
- ✅ **Python 3.9+** with Playwright
- ✅ **AI-Powered DOM Analysis** (OpenAI/Anthropic integration)
- ✅ **BDD Scenario Processing** (Given-When-Then)
- ✅ **Smart Selector Caching** (learns from successful runs)
- ✅ **Screenshot Documentation** (visual test evidence)

## Key Features

### 🧠 AI-Powered Element Detection
The system intelligently identifies and interacts with:
- **Buttons**: `<button>`, `role="button"`, `input[type="button"]`
- **Links**: `<a href="...">` elements
- **Form Inputs**: Text fields, textareas, checkboxes, radio buttons
- **Dropdowns**: `<select>` elements and custom dropdowns
- **Forms**: Complete form submission workflows
- **ARIA Elements**: Advanced accessibility elements

### 🎯 Smart Selector Strategy
- **Multi-Strategy Generation**: CSS selectors, XPath, text-based, attribute-based
- **Confidence Scoring**: Ranks selector reliability
- **Adaptive Learning**: Caches successful selectors for future runs
- **Fallback Mechanisms**: Multiple selector options per element

### 📝 BDD Scenario Support
- **XML Scenario Files**: Structured Given-When-Then scenarios
- **Natural Language Processing**: Interprets human-readable test steps
- **Automatic Step Translation**: Converts BDD steps to Playwright actions

### 📊 Comprehensive Reporting
- **Step-by-Step Screenshots**: Visual documentation of each action
- **JSON Test Reports**: Detailed execution logs
- **Selector Cache Files**: Reusable element mappings
- **Execution Summaries**: Pass/fail statistics with evidence


## Use Cases

### 1. **BDD Scenario Automation**
Write natural language test scenarios that AI automatically executes:
```xml
<scenario name="SauceDemo Login Test">
  <given>I navigate to "https://www.saucedemo.com/"</given>
  <when>I login with username "standard_user" and password "secret_sauce"</when>
  <when>I add "Sauce Labs Backpack" to cart</when>
  <when>I proceed to checkout</when>
  <then>I should see the checkout page</then>
</scenario>
```

### 2. **Regression Testing**
- **Automated Test Execution**: Run entire test suites without manual intervention
- **Selector Caching**: Second runs are faster using cached element mappings
- **Visual Validation**: Screenshot comparison for UI regression detection

### 3. **Rapid Test Creation**
- **No Code Required**: Write tests in natural language
- **AI Element Discovery**: Automatically finds and interacts with page elements
- **Instant Execution**: Test scenarios run immediately after creation

### 4. **Cross-Browser Testing**
- Support for Chromium, Firefox, WebKit
- Headless execution for CI/CD pipelines
- Mobile device emulation

## Workflow Example

### Input: BDD Scenario File
```xml
<scenario name="E-commerce Purchase Flow">
  <given>I am on "https://www.saucedemo.com/"</given>
  <when>I enter "standard_user" in the username field</when>
  <when>I enter "secret_sauce" in the password field</when>
  <when>I click the login button</when>
  <when>I add the first product to cart</when>
  <when>I go to cart and checkout</when>
  <then>I should complete the purchase successfully</then>
</scenario>
```

### AI Processing Flow
1. **Parse BDD Scenario** → Extract Given-When-Then steps
2. **Load Target Page** → Navigate using Playwright
3. **AI DOM Analysis** → Identify interactive elements
4. **Execute Actions** → Click, type, navigate based on natural language
5. **Cache Selectors** → Store successful element mappings
6. **Generate Report** → Screenshots + execution log

### Output: Test Evidence
- `screenshots/step_001_navigate_to_page.png`
- `screenshots/step_002_enter_username.png` 
- `cache/saucedemo_selectors.json`
- `reports/test_execution_report.json`

## Project Structure
```
agentic_ai_testing/
├── scenarios/           # BDD test scenarios (XML)
├── src/
│   ├── ai_analyzer.py   # AI-powered DOM analysis
│   ├── bdd_parser.py    # Scenario parsing
│   ├── playwright_agent.py # Browser automation
│   └── selector_cache.py    # Learning system
├── cache/              # Cached selectors
├── screenshots/        # Step-by-step evidence
├── reports/           # Test execution reports
└── requirements.txt   # Python dependencies
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run a BDD scenario
python run_test.py scenarios/saucedemo_test.xml

# View results
open reports/latest_report.html
```

Ready to build this intelligent testing framework? 🚀