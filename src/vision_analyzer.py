import base64
import io
import os
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw
import numpy as np

from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class BoundingBox(BaseModel):
    x: int = Field(description="X coordinate of top-left corner")
    y: int = Field(description="Y coordinate of top-left corner") 
    width: int = Field(description="Width of the bounding box")
    height: int = Field(description="Height of the bounding box")

class VisualElement(BaseModel):
    element_type: str = Field(description="Type of UI element (button, input, link, etc.)")
    text_content: str = Field(description="Visible text content")
    bounding_box: BoundingBox = Field(description="Location on screen")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    interaction_hint: str = Field(description="How to interact with this element")
    visual_description: str = Field(description="Visual characteristics")

class VisionAnalysisResult(BaseModel):
    elements: List[VisualElement] = Field(description="Detected UI elements")
    page_layout: str = Field(description="Overall page layout description")
    recommended_action: str = Field(description="Recommended next action")

class VisionBasedAnalyzer:
    """Enhanced element detection using vision models and LangChain"""
    
    def __init__(self, provider: str = "auto"):
        self.provider = self._detect_provider(provider)
        self.vision_model = self._initialize_vision_model()
        
    def _detect_provider(self, provider: str) -> str:
        """Auto-detect available vision model provider"""
        if provider == "auto":
            # Check for GPT-4V availability
            if os.getenv('OPENAI_API_KEY') or (os.getenv('AZURE_OPENAI_API_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT')):
                return "openai"
            elif os.getenv('ANTHROPIC_API_KEY'):
                return "anthropic"
            else:
                raise ValueError("No vision model provider available")
        return provider
    
    def _initialize_vision_model(self):
        """Initialize vision-capable model"""
        try:
            if self.provider == "openai":
                print("👁️ Using GPT-4V for vision analysis")
                if os.getenv('AZURE_OPENAI_API_KEY'):
                    from langchain_openai import AzureChatOpenAI
                    return AzureChatOpenAI(
                        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
                        deployment_name=os.getenv('AZURE_OPENAI_VISION_DEPLOYMENT', 'gpt-4v'),
                        temperature=0.1,
                        max_tokens=2000
                    )
                else:
                    return ChatOpenAI(
                        model="gpt-4-vision-preview",
                        api_key=os.getenv('OPENAI_API_KEY'),
                        temperature=0.1,
                        max_tokens=2000
                    )
            elif self.provider == "anthropic":
                print("👁️ Using Claude Vision for vision analysis")
                return ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                    temperature=0.1,
                    max_tokens=2000
                )
        except Exception as e:
            print(f"⚠️ Vision model initialization failed: {e}")
            raise
    
    async def analyze_screenshot(self, screenshot_path: str, action_intent: str) -> VisionAnalysisResult:
        """Analyze screenshot to identify interactive elements"""
        try:
            # Load and process image
            image = Image.open(screenshot_path)
            image_base64 = self._encode_image(image)
            
            # Create vision analysis prompt
            prompt = self._create_vision_prompt(action_intent)
            
            if self.provider == "anthropic":
                # Claude's format for images
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
                response = self.vision_model.invoke(messages)
                
            else:  # OpenAI format
                messages = [
                    HumanMessage(content=[
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ])
                ]
                response = self.vision_model.invoke(messages)
            
            # Parse response to structured format
            result = self._parse_vision_response(response.content, image.size)
            
            print(f"👁️ Vision analysis found {len(result.elements)} interactive elements")
            return result
            
        except Exception as e:
            print(f"❌ Vision analysis failed: {e}")
            # Return empty result
            return VisionAnalysisResult(
                elements=[],
                page_layout="Analysis failed",
                recommended_action="Fallback to DOM analysis"
            )
    
    def _create_vision_prompt(self, action_intent: str) -> str:
        """Create comprehensive vision analysis prompt"""
        return f"""
        Analyze this web page screenshot to help with automated testing. I need to: "{action_intent}"

        Please identify ALL interactive elements on the page including:
        1. Buttons (primary, secondary, icon buttons)
        2. Form inputs (text fields, checkboxes, dropdowns, radio buttons)
        3. Links and navigation elements
        4. Interactive images/icons
        5. Menu items and tabs
        6. Any clickable elements

        For each element, provide:
        - Element type and purpose
        - Exact visible text (if any)
        - Approximate pixel coordinates (x, y, width, height)
        - Visual characteristics (color, size, styling)
        - How it relates to the intended action
        - Confidence level (0.0-1.0)

        Focus especially on elements that would be relevant for: "{action_intent}"

        Also provide:
        - Overall page layout description
        - Recommended approach for the intended action
        - Any potential challenges or alternatives

        Return your analysis in a structured format that can be easily parsed.
        """
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64"""
        # Resize if too large (vision models have size limits)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _parse_vision_response(self, response_text: str, image_size: Tuple[int, int]) -> VisionAnalysisResult:
        """Parse vision model response into structured format"""
        # This would need more sophisticated parsing based on the actual response format
        # For now, providing a basic structure
        
        try:
            # Extract elements from response (would need more sophisticated NLP parsing)
            elements = self._extract_elements_from_text(response_text, image_size)
            
            # Extract page layout description
            page_layout = self._extract_page_layout(response_text)
            
            # Extract recommended action
            recommended_action = self._extract_recommended_action(response_text)
            
            return VisionAnalysisResult(
                elements=elements,
                page_layout=page_layout,
                recommended_action=recommended_action
            )
            
        except Exception as e:
            print(f"⚠️ Failed to parse vision response: {e}")
            return VisionAnalysisResult(
                elements=[],
                page_layout="Could not parse layout",
                recommended_action="Manual analysis needed"
            )
    
    def _extract_elements_from_text(self, text: str, image_size: Tuple[int, int]) -> List[VisualElement]:
        """Extract element information from text response"""
        elements = []
        
        # This would need sophisticated text parsing to extract structured data
        # For demo purposes, showing the structure
        
        # Look for element descriptions in the text
        lines = text.split('\n')
        current_element = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple pattern matching - in practice, would use more sophisticated NLP
            if any(word in line.lower() for word in ['button', 'input', 'link', 'field']):
                if current_element:
                    # Save previous element
                    if self._is_complete_element(current_element):
                        elements.append(self._create_visual_element(current_element))
                    current_element = {}
                
                current_element['description'] = line
            elif 'coordinates' in line.lower() or 'location' in line.lower():
                # Extract coordinates
                coords = self._extract_coordinates(line)
                if coords:
                    current_element['coordinates'] = coords
            elif 'text' in line.lower() and ':' in line:
                # Extract text content
                text_content = line.split(':', 1)[1].strip()
                current_element['text'] = text_content
        
        # Don't forget the last element
        if current_element and self._is_complete_element(current_element):
            elements.append(self._create_visual_element(current_element))
        
        return elements
    
    def _extract_coordinates(self, text: str) -> Optional[Tuple[int, int, int, int]]:
        """Extract coordinates from text"""
        import re
        
        # Look for patterns like "(100, 200, 150, 50)" or "x:100 y:200 w:150 h:50"
        coord_pattern = r'(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)'
        match = re.search(coord_pattern, text)
        
        if match:
            return tuple(map(int, match.groups()))
        
        return None
    
    def _is_complete_element(self, element_data: Dict) -> bool:
        """Check if element has minimum required data"""
        return 'description' in element_data and ('coordinates' in element_data or 'text' in element_data)
    
    def _create_visual_element(self, element_data: Dict) -> VisualElement:
        """Create VisualElement from parsed data"""
        coords = element_data.get('coordinates', (0, 0, 0, 0))
        
        return VisualElement(
            element_type=self._determine_element_type(element_data['description']),
            text_content=element_data.get('text', ''),
            bounding_box=BoundingBox(
                x=coords[0],
                y=coords[1], 
                width=coords[2],
                height=coords[3]
            ),
            confidence=0.8,  # Would be determined by parsing confidence indicators
            interaction_hint=self._determine_interaction(element_data['description']),
            visual_description=element_data['description']
        )
    
    def _determine_element_type(self, description: str) -> str:
        """Determine element type from description"""
        desc_lower = description.lower()
        
        if 'button' in desc_lower:
            return 'button'
        elif 'input' in desc_lower or 'field' in desc_lower:
            return 'input'
        elif 'link' in desc_lower:
            return 'link'
        elif 'checkbox' in desc_lower:
            return 'checkbox'
        elif 'dropdown' in desc_lower or 'select' in desc_lower:
            return 'select'
        else:
            return 'interactive'
    
    def _determine_interaction(self, description: str) -> str:
        """Determine how to interact with element"""
        element_type = self._determine_element_type(description)
        
        interaction_map = {
            'button': 'click',
            'link': 'click',
            'input': 'type_text',
            'checkbox': 'toggle',
            'select': 'select_option'
        }
        
        return interaction_map.get(element_type, 'click')
    
    def _extract_page_layout(self, text: str) -> str:
        """Extract page layout description from response"""
        # Look for layout-related sentences
        lines = text.split('\n')
        layout_lines = []
        
        for line in lines:
            if any(word in line.lower() for word in ['layout', 'page', 'structure', 'organized']):
                layout_lines.append(line.strip())
        
        return ' '.join(layout_lines) if layout_lines else "Standard web page layout"
    
    def _extract_recommended_action(self, text: str) -> str:
        """Extract recommended action from response"""
        # Look for recommendation-related sentences
        lines = text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should', 'best']):
                return line.strip()
        
        return "Proceed with DOM-based element selection"
    
    def create_annotated_screenshot(self, screenshot_path: str, elements: List[VisualElement], 
                                  output_path: str = None) -> str:
        """Create annotated screenshot with detected elements highlighted"""
        try:
            image = Image.open(screenshot_path)
            draw = ImageDraw.Draw(image)
            
            # Draw bounding boxes for each element
            for i, element in enumerate(elements):
                bbox = element.bounding_box
                
                # Choose color based on element type
                color_map = {
                    'button': 'red',
                    'input': 'blue',
                    'link': 'green',
                    'select': 'orange',
                    'checkbox': 'purple'
                }
                color = color_map.get(element.element_type, 'yellow')
                
                # Draw rectangle
                draw.rectangle([
                    bbox.x, bbox.y,
                    bbox.x + bbox.width, bbox.y + bbox.height
                ], outline=color, width=2)
                
                # Add label
                label = f"{i+1}: {element.element_type}"
                draw.text((bbox.x, bbox.y - 15), label, fill=color)
            
            # Save annotated image
            if not output_path:
                base, ext = os.path.splitext(screenshot_path)
                output_path = f"{base}_annotated{ext}"
            
            image.save(output_path)
            print(f"💾 Annotated screenshot saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create annotated screenshot: {e}")
            return screenshot_path

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_vision_analyzer():
        analyzer = VisionBasedAnalyzer()
        
        # This would work with an actual screenshot
        # result = await analyzer.analyze_screenshot("test_screenshot.png", "click the login button")
        # print(f"Found {len(result.elements)} elements")
        # 
        # # Create annotated version
        # annotated = analyzer.create_annotated_screenshot("test_screenshot.png", result.elements)
        
        print("Vision analyzer initialized successfully")
    
    asyncio.run(test_vision_analyzer())