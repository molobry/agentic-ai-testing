import os
import json
from typing import List, Dict, Optional, TypedDict, Annotated
from dataclasses import dataclass
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# LLM imports based on provider
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Pydantic models for structured output
class ElementSelector(BaseModel):
    type: str = Field(description="Type of selector (id, css, xpath, text)")
    value: str = Field(description="Selector value")
    confidence: int = Field(description="Confidence score 0-100")

class WebElement(BaseModel):
    element_type: str = Field(description="HTML element type (button, input, etc.)")
    text_content: str = Field(description="Visible text content")
    selectors: List[ElementSelector] = Field(description="List of possible selectors")
    reasoning: str = Field(description="AI reasoning for element selection")
    confidence_score: int = Field(description="Overall confidence 0-100")
    context: Dict = Field(description="Surrounding context information")

class ElementAnalysisResult(BaseModel):
    elements: List[WebElement] = Field(description="List of relevant elements")
    action_strategy: str = Field(description="Recommended interaction strategy")
    fallback_options: List[str] = Field(description="Alternative approaches if primary fails")

# LangGraph State
class AnalysisState(TypedDict):
    html_content: str
    action_intent: str
    page_screenshot: Optional[str]
    extracted_elements: List[Dict]
    ai_analysis: Optional[ElementAnalysisResult]
    final_elements: List[Dict]
    error_message: Optional[str]

@dataclass
class LangChainConfig:
    provider: str = "auto"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 2000

class AdvancedElementAnalyzer:
    def __init__(self, config: Optional[LangChainConfig] = None):
        self.config = config or LangChainConfig()
        self.llm = self._initialize_llm()
        self.output_parser = PydanticOutputParser(pydantic_object=ElementAnalysisResult)
        self.workflow = self._create_analysis_workflow()
        
    def _initialize_llm(self):
        """Initialize LLM based on available providers"""
        provider = self.config.provider
        
        if provider == "auto":
            # Auto-detect based on environment variables
            if os.getenv('AZURE_OPENAI_API_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT'):
                provider = "azure_openai"
            elif os.getenv('ANTHROPIC_API_KEY'):
                provider = "anthropic"
            elif os.getenv('OPENAI_API_KEY'):
                provider = "openai"
            else:
                raise ValueError("No AI provider credentials found")
        
        try:
            if provider == "azure_openai":
                print("🧠 Using Azure OpenAI with LangChain")
                return AzureChatOpenAI(
                    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
                    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-35-turbo'),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            elif provider == "anthropic":
                print("🧠 Using Anthropic Claude with LangChain")
                return ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            elif provider == "openai":
                print("🧠 Using OpenAI with LangChain")
                return ChatOpenAI(
                    model=self.config.model_name,
                    api_key=os.getenv('OPENAI_API_KEY'),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
        except Exception as e:
            print(f"⚠️ Failed to initialize {provider}: {e}")
            raise
    
    def _create_analysis_workflow(self) -> StateGraph:
        """Create LangGraph workflow for element analysis"""
        
        def extract_dom_elements(state: AnalysisState) -> AnalysisState:
            """Extract and pre-process DOM elements"""
            try:
                soup = BeautifulSoup(state["html_content"], 'html.parser')
                elements = self._extract_interactive_elements(soup)
                state["extracted_elements"] = elements
                print(f"🔍 Extracted {len(elements)} interactive elements")
            except Exception as e:
                state["error_message"] = f"DOM extraction failed: {e}"
            return state
        
        def analyze_with_ai(state: AnalysisState) -> AnalysisState:
            """Use AI to analyze and rank elements"""
            if state.get("error_message"):
                return state
                
            try:
                # Create sophisticated prompt with context
                analysis_prompt = self._create_analysis_prompt()
                
                # Prepare input data
                input_data = {
                    "action_intent": state["action_intent"],
                    "elements": json.dumps(state["extracted_elements"], indent=2),
                    "format_instructions": self.output_parser.get_format_instructions()
                }
                
                # Create and run chain
                chain = analysis_prompt | self.llm | self.output_parser
                result = chain.invoke(input_data)
                
                state["ai_analysis"] = result
                print(f"🧠 AI analyzed {len(result.elements)} relevant elements")
                
            except Exception as e:
                state["error_message"] = f"AI analysis failed: {e}"
                print(f"⚠️ AI analysis error: {e}")
            
            return state
        
        def finalize_results(state: AnalysisState) -> AnalysisState:
            """Finalize and format results"""
            if state.get("error_message"):
                # Fallback to basic analysis
                state["final_elements"] = self._basic_element_ranking(
                    state.get("extracted_elements", []), 
                    state["action_intent"]
                )
                print("🔧 Using fallback element ranking")
            else:
                # Convert AI analysis to final format
                ai_result = state["ai_analysis"]
                state["final_elements"] = [
                    {
                        "type": elem.element_type,
                        "text": elem.text_content,
                        "selectors": {sel.type: sel.value for sel in elem.selectors},
                        "confidence_score": elem.confidence_score,
                        "ai_reasoning": elem.reasoning,
                        "context": elem.context
                    }
                    for elem in ai_result.elements
                ]
                print(f"✅ Finalized {len(state['final_elements'])} elements")
            
            return state
        
        # Build the graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("extract_dom", extract_dom_elements)
        workflow.add_node("ai_analysis", analyze_with_ai)
        workflow.add_node("finalize", finalize_results)
        
        # Add edges
        workflow.add_edge("extract_dom", "ai_analysis")
        workflow.add_edge("ai_analysis", "finalize")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("extract_dom")
        
        return workflow.compile()
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create sophisticated analysis prompt"""
        system_template = """You are an expert web automation engineer with deep knowledge of:
        - HTML/DOM structure and element identification
        - CSS selectors and XPath expressions
        - Web accessibility patterns (ARIA, semantic HTML)
        - UI/UX interaction patterns
        - Cross-browser compatibility

        Your task is to analyze web page elements and identify the most suitable elements for automated interactions.
        Consider element visibility, stability, accessibility, and interaction patterns.
        """
        
        human_template = """
        TASK: {action_intent}
        
        WEB PAGE ELEMENTS:
        {elements}
        
        Please analyze these elements and identify which ones are most suitable for the intended action.
        Consider:
        1. Element semantics and role
        2. Visibility and interaction capability  
        3. Selector stability (prefer id > data-testid > class > xpath)
        4. Text content relevance
        5. Context and surrounding elements
        6. Accessibility attributes
        
        Provide a ranked list of elements with reasoning and multiple selector strategies.
        
        {format_instructions}
        """
        
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def analyze_page_elements(self, html_content: str, action_intent: str, 
                            screenshot_path: Optional[str] = None) -> List[Dict]:
        """Main analysis method using LangGraph workflow"""
        
        # Initialize state
        initial_state: AnalysisState = {
            "html_content": html_content,
            "action_intent": action_intent,
            "page_screenshot": screenshot_path,
            "extracted_elements": [],
            "ai_analysis": None,
            "final_elements": [],
            "error_message": None
        }
        
        # Run the workflow
        try:
            result_state = self.workflow.invoke(initial_state)
            return result_state["final_elements"]
        except Exception as e:
            print(f"⚠️ Workflow execution failed: {e}")
            # Fallback to basic extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            basic_elements = self._extract_interactive_elements(soup)
            return self._basic_element_ranking(basic_elements, action_intent)
    
    def _extract_interactive_elements(self, soup: BeautifulSoup) -> List[Dict]:
        """Enhanced element extraction with more context"""
        elements = []
        
        # Enhanced selector patterns
        interactive_patterns = [
            ('button', 'button'),
            ('input[type="submit"]', 'submit_button'),
            ('input[type="button"]', 'button'),
            ('input[type="text"]', 'text_input'),
            ('input[type="email"]', 'email_input'),
            ('input[type="password"]', 'password_input'),
            ('input[type="checkbox"]', 'checkbox'),
            ('input[type="radio"]', 'radio'),
            ('textarea', 'textarea'),
            ('select', 'select'),
            ('a[href]', 'link'),
            ('[role="button"]', 'aria_button'),
            ('[role="textbox"]', 'aria_textbox'),
            ('[role="combobox"]', 'aria_combobox'),
            ('[tabindex]:not([tabindex="-1"])', 'focusable'),
            ('[onclick]', 'clickable'),
            ('[data-testid]', 'test_element')
        ]
        
        for css_selector, element_type in interactive_patterns:
            found_elements = soup.select(css_selector)
            
            for element in found_elements:
                element_data = {
                    'type': element_type,
                    'tag': element.name,
                    'text': self._extract_element_text(element),
                    'attributes': dict(element.attrs),
                    'selectors': self._generate_enhanced_selectors(element, soup),
                    'context': self._extract_enhanced_context(element),
                    'accessibility': self._extract_accessibility_info(element),
                    'position': self._get_element_position(element),
                    'confidence_score': 0  # Will be set by AI
                }
                elements.append(element_data)
        
        return elements
    
    def _extract_element_text(self, element) -> str:
        """Extract meaningful text from element"""
        # Direct text
        text = element.get_text(strip=True)
        if text:
            return text[:100]
        
        # Check attributes
        for attr in ['placeholder', 'value', 'title', 'aria-label', 'alt']:
            if element.get(attr):
                return element.get(attr)[:100]
        
        # Check nearby labels
        element_id = element.get('id')
        if element_id:
            label = element.find_parent().find('label', {'for': element_id})
            if label:
                return label.get_text(strip=True)[:100]
        
        return ""
    
    def _generate_enhanced_selectors(self, element, soup: BeautifulSoup) -> Dict[str, str]:
        """Generate multiple selector strategies"""
        selectors = {}
        
        # ID selector (highest priority)
        if element.get('id'):
            selectors['id'] = f"#{element.get('id')}"
        
        # Test ID selectors
        for test_attr in ['data-testid', 'data-test', 'data-cy', 'data-automation']:
            if element.get(test_attr):
                selectors[test_attr] = f"[{test_attr}='{element.get(test_attr)}']"
        
        # Name attribute
        if element.get('name'):
            selectors['name'] = f"[name='{element.get('name')}']"
        
        # ARIA selectors
        if element.get('role'):
            selectors['role'] = f"[role='{element.get('role')}']"
        
        if element.get('aria-label'):
            selectors['aria_label'] = f"[aria-label='{element.get('aria-label')}']"
        
        # Class selectors (with specificity)
        classes = element.get('class', [])
        if classes:
            # Use most specific classes first
            unique_classes = [cls for cls in classes if len(soup.select(f'.{cls}')) < 10]
            if unique_classes:
                selectors['class'] = f".{'.'.join(unique_classes[:3])}"
            else:
                selectors['class'] = f".{'.'.join(classes[:2])}"
        
        # Text-based selectors
        text = self._extract_element_text(element)
        if text and len(text) < 50:
            selectors['text'] = text
        
        # XPath
        selectors['xpath'] = self._generate_xpath(element, soup)
        
        # CSS path
        selectors['css_path'] = self._generate_css_path(element)
        
        return selectors
    
    def _extract_enhanced_context(self, element) -> Dict:
        """Extract rich contextual information"""
        return {
            'parent_tag': element.parent.name if element.parent else None,
            'parent_class': element.parent.get('class', []) if element.parent else [],
            'siblings_count': len(element.find_siblings()) if element.parent else 0,
            'form_context': self._get_form_context(element),
            'surrounding_text': self._get_surrounding_text(element),
            'visual_cues': self._extract_visual_cues(element)
        }
    
    def _extract_accessibility_info(self, element) -> Dict:
        """Extract accessibility information"""
        return {
            'role': element.get('role'),
            'aria_label': element.get('aria-label'),
            'aria_labelledby': element.get('aria-labelledby'),
            'aria_describedby': element.get('aria-describedby'),
            'tabindex': element.get('tabindex'),
            'title': element.get('title')
        }
    
    def _get_element_position(self, element) -> Dict:
        """Get element position information"""
        return {
            'index': element.parent.contents.index(element) if element.parent else 0,
            'depth': len(list(element.parents)),
            'siblings': len(element.find_siblings())
        }
    
    def _get_form_context(self, element) -> Dict:
        """Get form-related context"""
        form = element.find_parent('form')
        if form:
            return {
                'form_id': form.get('id'),
                'form_action': form.get('action'),
                'form_method': form.get('method', 'GET'),
                'form_fields_count': len(form.find_all(['input', 'select', 'textarea']))
            }
        return {}
    
    def _get_surrounding_text(self, element) -> Dict:
        """Get text from surrounding elements"""
        try:
            prev_text = ""
            next_text = ""
            
            # Previous sibling text
            prev_sibling = element.find_previous_sibling()
            if prev_sibling:
                prev_text = prev_sibling.get_text(strip=True)[-100:]
            
            # Next sibling text
            next_sibling = element.find_next_sibling()
            if next_sibling:
                next_text = next_sibling.get_text(strip=True)[:100]
            
            return {
                'previous': prev_text,
                'next': next_text,
                'parent': element.parent.get_text(strip=True)[:200] if element.parent else ""
            }
        except:
            return {'previous': '', 'next': '', 'parent': ''}
    
    def _extract_visual_cues(self, element) -> Dict:
        """Extract visual styling cues"""
        style = element.get('style', '')
        classes = element.get('class', [])
        
        visual_indicators = {
            'has_style': bool(style),
            'style_hints': [],
            'class_hints': []
        }
        
        # Check for common visual classes
        visual_classes = ['btn', 'button', 'primary', 'submit', 'action', 'link', 'nav']
        visual_indicators['class_hints'] = [cls for cls in classes if any(hint in cls.lower() for hint in visual_classes)]
        
        return visual_indicators
    
    def _generate_xpath(self, element, soup: BeautifulSoup) -> str:
        """Generate XPath with better uniqueness"""
        path = []
        current = element
        
        while current and current.name != 'html':
            tag = current.name
            
            # Try ID first
            if current.get('id'):
                path.insert(0, f"//{tag}[@id='{current.get('id')}']")
                break
            
            # Try unique attributes
            for attr in ['data-testid', 'name', 'class']:
                if current.get(attr):
                    attr_value = current.get(attr)
                    if isinstance(attr_value, list):
                        attr_value = ' '.join(attr_value)
                    
                    xpath_with_attr = f"//{tag}[@{attr}='{attr_value}']"
                    if len(soup.select(f"[{attr}='{attr_value}']")) == 1:
                        path.insert(0, xpath_with_attr)
                        return ''.join(path)
            
            # Fallback to position
            siblings = current.parent.find_all(tag) if current.parent else [current]
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                path.insert(0, f"/{tag}[{index}]")
            else:
                path.insert(0, f"/{tag}")
            
            current = current.parent
        
        return "/" + "/".join(path) if path else f"//{element.name}"
    
    def _generate_css_path(self, element) -> str:
        """Generate CSS path with better specificity"""
        path = []
        current = element
        
        while current and current.name != 'html' and len(path) < 4:
            selector = current.name
            
            # Add ID if available
            if current.get('id'):
                selector += f"#{current.get('id')}"
                path.insert(0, selector)
                break
            
            # Add meaningful classes
            classes = current.get('class', [])
            if classes:
                meaningful_classes = [cls for cls in classes if not cls.startswith('css-')]
                if meaningful_classes:
                    selector += "." + ".".join(meaningful_classes[:2])
            
            # Add nth-child if needed for uniqueness
            if current.parent:
                siblings = current.parent.find_all(current.name)
                if len(siblings) > 1:
                    index = siblings.index(current) + 1
                    selector += f":nth-child({index})"
            
            path.insert(0, selector)
            current = current.parent
        
        return " > ".join(path)
    
    def _basic_element_ranking(self, elements: List[Dict], action_intent: str) -> List[Dict]:
        """Fallback ranking when AI is not available"""
        for element in elements:
            score = 0
            text = element['text'].lower()
            intent_lower = action_intent.lower()
            
            # Text matching
            intent_words = intent_lower.split()
            text_matches = sum(1 for word in intent_words if word in text)
            score += text_matches * 20
            
            # Element type matching
            element_type = element['type']
            if 'click' in intent_lower and element_type in ['button', 'submit_button', 'link']:
                score += 30
            elif any(word in intent_lower for word in ['enter', 'type', 'input']) and 'input' in element_type:
                score += 30
            elif 'select' in intent_lower and element_type == 'select':
                score += 30
            
            # Selector quality
            selectors = element['selectors']
            if 'id' in selectors:
                score += 25
            if any(attr in selectors for attr in ['data-testid', 'data-test']):
                score += 20
            if 'name' in selectors:
                score += 15
            
            element['confidence_score'] = min(score, 100)
        
        return sorted(elements, key=lambda x: x['confidence_score'], reverse=True)

if __name__ == "__main__":
    # Example usage
    analyzer = AdvancedElementAnalyzer()
    
    sample_html = """
    <html>
    <body>
        <form id="login-form">
            <label for="username">Username:</label>
            <input type="text" id="username" name="username" data-testid="username-input" placeholder="Enter your username">
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" data-testid="password-input" placeholder="Enter your password">
            <button type="submit" id="login-btn" data-testid="login-button">Sign In</button>
        </form>
    </body>
    </html>
    """
    
    try:
        elements = analyzer.analyze_page_elements(sample_html, "click the login button")
        print(f"Found {len(elements)} elements:")
        for i, elem in enumerate(elements[:3]):
            print(f"{i+1}. {elem['type']} - '{elem['text']}' (Score: {elem['confidence_score']})")
            if elem.get('ai_reasoning'):
                print(f"   Reasoning: {elem['ai_reasoning']}")
    except Exception as e:
        print(f"Error: {e}")