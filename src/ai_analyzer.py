import json
import os
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

class AIAnalyzer:
    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        self.provider = provider
        self.client = None
        
        # Auto-detect provider based on available API keys
        if provider == "auto":
            if os.getenv('AZURE_OPENAI_API_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT'):
                self.provider = "azure_openai"
            elif os.getenv('ANTHROPIC_API_KEY'):
                self.provider = "anthropic"
            elif os.getenv('OPENAI_API_KEY'):
                self.provider = "openai"
            else:
                self.provider = None
                print("⚠️  No AI API key found. Using basic element detection only.")
        
        # Initialize the appropriate client
        if self.provider == "azure_openai":
            try:
                import openai
                self.client = openai.AzureOpenAI(
                    api_key=api_key or os.getenv('AZURE_OPENAI_API_KEY'),
                    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
                )
                self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-35-turbo')
            except ImportError:
                print("⚠️  openai package not installed. Install with: pip install openai")
                self.provider = None
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
                )
            except ImportError:
                print("⚠️  anthropic package not installed. Install with: pip install anthropic")
                self.provider = None
        elif self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=api_key or os.getenv('OPENAI_API_KEY')
                )
            except ImportError:
                print("⚠️  openai package not installed. Install with: pip install openai")
                self.provider = None
        
    def analyze_page_elements(self, html_content: str, action_intent: str) -> List[Dict]:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract all interactive elements
        interactive_elements = self._extract_interactive_elements(soup)
        
        # Use AI to identify the most relevant elements for the action
        relevant_elements = self._ai_element_selection(
            interactive_elements, 
            action_intent
        )
        
        return relevant_elements
    
    def _extract_interactive_elements(self, soup: BeautifulSoup) -> List[Dict]:
        elements = []
        
        # Define interactive element selectors
        selectors = [
            ('button', 'button'),
            ('input', 'input'),
            ('select', 'select'),
            ('textarea', 'textarea'),
            ('a[href]', 'link'),
            ('[role="button"]', 'button'),
            ('[onclick]', 'clickable'),
            ('[tabindex]', 'focusable')
        ]
        
        for css_selector, element_type in selectors:
            found_elements = soup.select(css_selector)
            
            for element in found_elements:
                element_info = {
                    'type': element_type,
                    'tag': element.name,
                    'text': self._get_element_text(element),
                    'attributes': dict(element.attrs),
                    'selectors': self._generate_selectors(element, soup),
                    'context': self._get_element_context(element),
                    'confidence_score': 0
                }
                elements.append(element_info)
        
        return elements
    
    def _get_element_text(self, element) -> str:
        text = element.get_text(strip=True)
        if not text:
            # Check for common text attributes
            for attr in ['placeholder', 'value', 'title', 'aria-label']:
                if element.get(attr):
                    text = element.get(attr)
                    break
        return text[:100]  # Limit text length
    
    def _get_element_context(self, element) -> Dict:
        return {
            'parent_text': element.parent.get_text(strip=True)[:50] if element.parent else '',
            'preceding_text': self._get_preceding_text(element),
            'following_text': self._get_following_text(element),
            'form_context': self._get_form_context(element)
        }
    
    def _get_preceding_text(self, element) -> str:
        prev_sibling = element.find_previous_sibling()
        if prev_sibling:
            return prev_sibling.get_text(strip=True)[-50:]
        return ''
    
    def _get_following_text(self, element) -> str:
        next_sibling = element.find_next_sibling()
        if next_sibling:
            return next_sibling.get_text(strip=True)[:50]
        return ''
    
    def _get_form_context(self, element) -> Dict:
        form = element.find_parent('form')
        if form:
            return {
                'form_id': form.get('id', ''),
                'form_action': form.get('action', ''),
                'form_method': form.get('method', 'GET')
            }
        return {}
    
    def _generate_selectors(self, element, soup: BeautifulSoup) -> Dict[str, str]:
        selectors = {}
        
        # ID selector (most reliable)
        element_id = element.get('id')
        if element_id:
            selectors['id'] = f"#{element_id}"
        
        # Class selector
        classes = element.get('class', [])
        if classes:
            selectors['class'] = f".{'.'.join(classes)}"
        
        # Attribute selectors
        for attr in ['name', 'data-testid', 'data-test']:
            if element.get(attr):
                selectors[attr] = f"[{attr}='{element.get(attr)}']"
        
        # Text-based selector
        text = self._get_element_text(element)
        if text:
            selectors['text'] = text
        
        # XPath
        selectors['xpath'] = self._generate_xpath(element, soup)
        
        # CSS selector path
        selectors['css_path'] = self._generate_css_path(element)
        
        return selectors
    
    def _generate_xpath(self, element, soup: BeautifulSoup) -> str:
        path = []
        current = element
        
        while current and current.name != 'html':
            tag = current.name
            
            # Check if element has unique ID
            if current.get('id'):
                path.insert(0, f"//{tag}[@id='{current.get('id')}']")
                break
            
            # Count siblings of the same tag
            siblings = current.parent.find_all(tag) if current.parent else [current]
            if len(siblings) > 1:
                index = siblings.index(current) + 1
                path.insert(0, f"{tag}[{index}]")
            else:
                path.insert(0, tag)
            
            current = current.parent
        
        return "/" + "/".join(path) if path else f"//{element.name}"
    
    def _generate_css_path(self, element) -> str:
        path = []
        current = element
        
        while current and current.name != 'html':
            selector = current.name
            
            # Add ID if available
            if current.get('id'):
                selector += f"#{current.get('id')}"
                path.insert(0, selector)
                break
            
            # Add classes
            classes = current.get('class', [])
            if classes:
                selector += "." + ".".join(classes)
            
            path.insert(0, selector)
            current = current.parent
            
            if len(path) >= 3:  # Limit path depth
                break
        
        return " > ".join(path)
    
    def _ai_element_selection(self, elements: List[Dict], action_intent: str) -> List[Dict]:
        if not elements:
            return []
        
        # If no AI provider available, use basic confidence scoring
        if not self.client or not self.provider:
            print("🔍 Using basic element detection (no AI)")
            for element in elements:
                element['confidence_score'] = self._calculate_basic_confidence(element, action_intent)
            return sorted(elements, key=lambda x: x['confidence_score'], reverse=True)
        
        # Create a simplified representation for AI analysis
        element_summaries = []
        for i, element in enumerate(elements):
            summary = {
                'index': i,
                'type': element['type'],
                'text': element['text'],
                'attributes': {k: v for k, v in element['attributes'].items() 
                             if k in ['id', 'class', 'name', 'placeholder', 'type', 'role']},
                'context': element['context']
            }
            element_summaries.append(summary)
        
        prompt = f"""
        Analyze the following web page elements and identify which ones are most relevant for this action: "{action_intent}"
        
        Elements found on the page:
        {json.dumps(element_summaries, indent=2)}
        
        Please respond with a JSON array of element indices (0-based) ranked by relevance, along with confidence scores (0-100).
        
        Format:
        [
            {{"index": 0, "confidence": 95, "reasoning": "This is the login button based on text and context"}},
            {{"index": 2, "confidence": 70, "reasoning": "Alternative submit button"}}
        ]
        
        Consider:
        - Element text content and context
        - HTML attributes (id, class, name, etc.)
        - Element type and role
        - Surrounding text and form context
        """
        
        try:
            if self.provider == "azure_openai":
                print("🧠 Using Azure OpenAI (in-house) for element analysis")
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing web page elements for UI automation. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                ai_response = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                print("🧠 Using Anthropic AI for element analysis")
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": f"You are an expert at analyzing web page elements for UI automation. Return only valid JSON.\n\n{prompt}"}
                    ]
                )
                ai_response = response.content[0].text
                
            elif self.provider == "openai":
                print("🧠 Using OpenAI for element analysis")
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing web page elements for UI automation. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                ai_response = response.choices[0].message.content
            
            selected_elements = json.loads(ai_response)
            
            # Apply AI confidence scores to elements
            result_elements = []
            for selection in selected_elements:
                if 0 <= selection['index'] < len(elements):
                    element = elements[selection['index']].copy()
                    element['confidence_score'] = selection['confidence']
                    element['ai_reasoning'] = selection['reasoning']
                    result_elements.append(element)
            
            return result_elements
            
        except Exception as e:
            print(f"AI analysis failed, using basic detection: {e}")
            # Fallback: return elements with basic confidence scoring
            for element in elements:
                element['confidence_score'] = self._calculate_basic_confidence(element, action_intent)
            
            return sorted(elements, key=lambda x: x['confidence_score'], reverse=True)
    
    def _calculate_basic_confidence(self, element: Dict, action_intent: str) -> int:
        score = 0
        text = element['text'].lower()
        intent_lower = action_intent.lower()
        
        # Text matching
        if any(word in text for word in intent_lower.split()):
            score += 30
        
        # Element type matching
        if 'click' in intent_lower and element['type'] in ['button', 'link']:
            score += 25
        elif 'enter' in intent_lower and element['type'] in ['input', 'textarea']:
            score += 25
        elif 'select' in intent_lower and element['type'] == 'select':
            score += 25
        
        # ID/name matching
        attrs = element['attributes']
        if attrs.get('id') and any(word in attrs['id'].lower() for word in intent_lower.split()):
            score += 20
        if attrs.get('name') and any(word in attrs['name'].lower() for word in intent_lower.split()):
            score += 20
        
        return min(score, 100)

if __name__ == "__main__":
    # Example usage
    analyzer = AIAnalyzer()
    
    sample_html = """
    <html>
    <body>
        <form>
            <input type="text" id="username" placeholder="Enter username" />
            <input type="password" id="password" placeholder="Enter password" />
            <button type="submit" id="login-btn">Login</button>
        </form>
    </body>
    </html>
    """
    
    elements = analyzer.analyze_page_elements(sample_html, "click the login button")
    for element in elements[:3]:  # Top 3 elements
        print(f"Element: {element['text']} (Score: {element['confidence_score']})")
        print(f"Selectors: {element['selectors']}")
        print("---")