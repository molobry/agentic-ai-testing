import asyncio
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from PIL import Image
import io

from .ai_analyzer import AIAnalyzer
from .selector_cache import SelectorCache

class PlaywrightAgent:
    def __init__(self, headless: bool = False, browser_type: str = "chromium", record_video: bool = False):
        self.headless = headless
        self.browser_type = browser_type
        self.record_video = record_video
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.ai_analyzer = AIAnalyzer()
        self.selector_cache = SelectorCache()
        self.screenshot_counter = 1
        self.current_scenario = ""
        self.video_path = None
        
    async def start_browser(self):
        self.playwright = await async_playwright().start()
        
        if self.browser_type == "chromium":
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
        elif self.browser_type == "firefox":
            self.browser = await self.playwright.firefox.launch(headless=self.headless)
        elif self.browser_type == "webkit":
            self.browser = await self.playwright.webkit.launch(headless=self.headless)
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        
        # Set up video recording if requested
        context_options = {
            "viewport": {"width": 1280, "height": 720},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        if self.record_video:
            # Ensure recordings directory exists
            os.makedirs("recordings", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_name = (
                self.current_scenario.replace(" ", "_").lower()
                if self.current_scenario else "test"
            )
            self.video_path = f"recordings/{scenario_name}_{timestamp}.webm"
            
            context_options["record_video_dir"] = "recordings/"
        
        self.context = await self.browser.new_context(**context_options)
        
        self.page = await self.context.new_page()
        
        # Enable request/response logging
        self.page.on("request", lambda request: print(f"→ {request.method} {request.url}"))
        self.page.on("response", lambda response: print(f"← {response.status} {response.url}"))
    
    async def close_browser(self):
        # Save video if recording was enabled
        if self.record_video and self.page:
            try:
                # Close page to finalize video
                await self.page.close()
                await self.context.close()
                
                # Get the actual video path from Playwright
                video_path = await self.page.video.path() if self.page.video else None
                if video_path and os.path.exists(video_path):
                    # Move video to our desired location
                    import shutil
                    final_path = self.video_path or f"recordings/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
                    shutil.move(video_path, final_path)
                    print(f"🎥 Video recording saved: {final_path}")
                    self.video_path = final_path
                else:
                    print("⚠️ Video recording was enabled but no video file was created")
            except Exception as e:
                print(f"⚠️ Error saving video: {e}")
        
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def navigate_to(self, url: str) -> bool:
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await self.take_screenshot(f"navigate_to_{self._clean_url_for_filename(url)}")
            return True
        except Exception as e:
            print(f"Navigation failed: {e}")
            return False
    
    async def execute_step(self, step_description: str, step_number: int) -> Dict:
        result = {
            "step_number": step_number,
            "description": step_description,
            "success": False,
            "action_taken": None,
            "element_found": None,
            "screenshot": None,
            "error": None,
            "selectors_used": [],
            "cached_selector_used": False
        }
        
        try:
            # Check if we have cached selectors for this step
            cached_selector = self.selector_cache.get_selector(
                self.current_scenario, 
                step_description
            )
            
            if cached_selector:
                success = await self._try_cached_selector(cached_selector, step_description, result)
                if success:
                    result["cached_selector_used"] = True
                    await self.take_screenshot(f"step_{step_number:03d}_cached_success")
                    result["screenshot"] = f"step_{step_number:03d}_cached_success.png"
                    result["success"] = True
                    return result
            
            # If no cached selector or cached selector failed, use AI analysis
            html_content = await self.page.content()
            relevant_elements = self.ai_analyzer.analyze_page_elements(html_content, step_description)
            
            if not relevant_elements:
                result["error"] = "No relevant elements found for this action"
                return result
            
            # Try elements in order of confidence
            for element in relevant_elements:
                success, action_taken = await self._try_interact_with_element(element, step_description)
                
                if success:
                    result["success"] = True
                    result["action_taken"] = action_taken
                    result["element_found"] = element
                    result["selectors_used"] = list(element["selectors"].keys())
                    
                    # Cache successful selector
                    best_selector = self._get_best_selector(element)
                    if best_selector:
                        self.selector_cache.save_selector(
                            self.current_scenario,
                            step_description,
                            best_selector
                        )
                    
                    # Take success screenshot
                    await self.take_screenshot(f"step_{step_number:03d}_success")
                    result["screenshot"] = f"step_{step_number:03d}_success.png"
                    break
            
            if not result["success"]:
                result["error"] = "Could not successfully interact with any found elements"
                await self.take_screenshot(f"step_{step_number:03d}_failed")
                result["screenshot"] = f"step_{step_number:03d}_failed.png"
                
        except Exception as e:
            result["error"] = str(e)
            await self.take_screenshot(f"step_{step_number:03d}_error")
            result["screenshot"] = f"step_{step_number:03d}_error.png"
        
        return result
    
    async def _try_cached_selector(self, cached_selector: Dict, step_description: str, result: Dict) -> bool:
        try:
            selector_type = cached_selector["type"]
            selector_value = cached_selector["value"]
            
            if selector_type == "id":
                element = self.page.locator(f"#{selector_value}")
            elif selector_type == "css":
                element = self.page.locator(selector_value)
            elif selector_type == "xpath":
                element = self.page.locator(f"xpath={selector_value}")
            elif selector_type == "text":
                element = self.page.get_by_text(selector_value)
            else:
                return False
            
            # Wait for element to be available
            await element.wait_for(state="visible", timeout=5000)
            
            # Determine action type and execute
            action_taken = await self._execute_action_on_element(element, step_description)
            result["action_taken"] = action_taken
            
            return True
            
        except Exception as e:
            print(f"Cached selector failed: {e}")
            return False
    
    async def _try_interact_with_element(self, element: Dict, step_description: str) -> Tuple[bool, Optional[str]]:
        selectors = element["selectors"]
        
        # Try selectors in order of preference: ID > text > class > css_path > xpath
        selector_preference = ["id", "text", "name", "data-testid", "class", "css_path", "xpath"]
        
        for selector_type in selector_preference:
            if selector_type not in selectors or not selectors[selector_type]:
                continue
                
            try:
                selector_value = selectors[selector_type]
                
                if selector_type == "id":
                    playwright_element = self.page.locator(selector_value)
                elif selector_type == "text":
                    playwright_element = self.page.get_by_text(selector_value, exact=False)
                elif selector_type == "xpath":
                    playwright_element = self.page.locator(f"xpath={selector_value}")
                else:
                    playwright_element = self.page.locator(selector_value)
                
                # Wait for element to be available
                await playwright_element.wait_for(state="visible", timeout=5000)
                
                # Execute the appropriate action
                action_taken = await self._execute_action_on_element(playwright_element, step_description)
                
                return True, action_taken
                
            except Exception as e:
                print(f"Failed with selector {selector_type}={selectors[selector_type]}: {e}")
                continue
        
        return False, None
    
    async def _execute_action_on_element(self, element, step_description: str) -> str:
        description_lower = step_description.lower()
        
        if any(word in description_lower for word in ["click", "press", "tap", "select"]):
            await element.click()
            return "click"
            
        elif any(word in description_lower for word in ["enter", "type", "input", "fill"]):
            # Extract text to enter from step description
            text_to_enter = self._extract_text_from_description(step_description)
            if text_to_enter:
                await element.fill(text_to_enter)
                return f"fill: {text_to_enter}"
            else:
                await element.click()  # Focus the element
                return "focus"
                
        elif any(word in description_lower for word in ["check", "uncheck"]):
            if "uncheck" in description_lower:
                await element.uncheck()
                return "uncheck"
            else:
                await element.check()
                return "check"
                
        elif any(word in description_lower for word in ["hover", "mouse over"]):
            await element.hover()
            return "hover"
            
        else:
            # Default action is click
            await element.click()
            return "click"
    
    def _extract_text_from_description(self, description: str) -> Optional[str]:
        # Look for quoted text in the description
        import re
        
        # Match text in quotes
        quoted_match = re.search(r'"([^"]*)"', description)
        if quoted_match:
            return quoted_match.group(1)
        
        quoted_match = re.search(r"'([^']*)'", description)
        if quoted_match:
            return quoted_match.group(1)
        
        # Look for patterns like "enter X in" or "type X in"
        patterns = [
            r"(?:enter|type|input|fill)\s+([^\s]+)\s+(?:in|into)",
            r"(?:with|using)\s+([^\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _get_best_selector(self, element: Dict) -> Optional[Dict]:
        selectors = element["selectors"]
        
        # Preference order for caching
        if selectors.get("id"):
            return {"type": "id", "value": selectors["id"].replace("#", "")}
        elif selectors.get("data-testid"):
            return {"type": "css", "value": selectors["data-testid"]}
        elif selectors.get("name"):
            return {"type": "css", "value": selectors["name"]}
        elif selectors.get("text") and len(selectors["text"]) < 50:
            return {"type": "text", "value": selectors["text"]}
        elif selectors.get("class"):
            return {"type": "css", "value": selectors["class"]}
        elif selectors.get("css_path"):
            return {"type": "css", "value": selectors["css_path"]}
        elif selectors.get("xpath"):
            return {"type": "xpath", "value": selectors["xpath"]}
        
        return None
    
    async def take_screenshot(self, filename: str):
        if not self.page:
            return
        
        try:
            screenshot_path = f"screenshots/{filename}.png"
            await self.page.screenshot(path=screenshot_path, full_page=True)
            print(f"📸 Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"Screenshot failed: {e}")
    
    def _clean_url_for_filename(self, url: str) -> str:
        import re
        clean = re.sub(r'[^\w\-_.]', '_', url)
        return clean[:50]  # Limit filename length
    
    def set_current_scenario(self, scenario_name: str):
        self.current_scenario = scenario_name
        self.screenshot_counter = 1
    
    async def wait_for_page_load(self, timeout: int = 10000):
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=timeout)
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception as e:
            print(f"Page load wait timeout: {e}")

if __name__ == "__main__":
    # Example usage
    async def test_agent():
        agent = PlaywrightAgent(headless=False)
        await agent.start_browser()
        
        try:
            await agent.navigate_to("https://www.saucedemo.com/")
            
            result = await agent.execute_step("enter 'standard_user' in username field", 1)
            print(json.dumps(result, indent=2))
            
            result = await agent.execute_step("enter 'secret_sauce' in password field", 2)
            print(json.dumps(result, indent=2))
            
            result = await agent.execute_step("click login button", 3)
            print(json.dumps(result, indent=2))
            
        finally:
            await agent.close_browser()
    
    asyncio.run(test_agent())