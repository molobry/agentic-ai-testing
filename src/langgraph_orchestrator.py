import asyncio
import json
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from dataclasses import dataclass
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver

from .langchain_analyzer import AdvancedElementAnalyzer, LangChainConfig
from .bdd_parser import BDDScenario, TestStep
from .playwright_agent import PlaywrightAgent

# State management for the orchestrator
class TestExecutionState(TypedDict):
    scenario: BDDScenario
    current_step: int
    step_results: List[Dict[str, Any]]
    browser_agent: Optional[PlaywrightAgent]
    element_analyzer: Optional[AdvancedElementAnalyzer]
    page_context: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    error_context: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int

class StepExecutionResult(BaseModel):
    step_number: int
    step_type: str
    description: str
    success: bool
    action_taken: Optional[str] = None
    elements_found: List[Dict] = Field(default_factory=list)
    reasoning: Optional[str] = None
    screenshot_path: Optional[str] = None
    execution_time: float
    retry_count: int = 0

class TestOrchestrator:
    """Advanced test orchestrator using LangGraph for intelligent test execution"""
    
    def __init__(self, config: Optional[LangChainConfig] = None, 
                 browser_config: Optional[Dict] = None):
        self.config = config or LangChainConfig()
        self.browser_config = browser_config or {"headless": True, "record_video": False}
        
        # Initialize components
        self.element_analyzer = AdvancedElementAnalyzer(self.config)
        self.checkpointer = SqliteSaver.from_conn_string("memory:///checkpoints")
        
        # Create the orchestration workflow
        self.workflow = self._create_orchestration_workflow()
    
    def _create_orchestration_workflow(self) -> StateGraph:
        """Create LangGraph workflow for intelligent test orchestration"""
        
        async def initialize_execution(state: TestExecutionState) -> TestExecutionState:
            """Initialize browser and test environment"""
            try:
                print(f"🚀 Initializing test execution for: {state['scenario'].name}")
                
                # Initialize browser agent
                browser_agent = PlaywrightAgent(
                    headless=self.browser_config.get("headless", True),
                    browser_type=self.browser_config.get("browser", "chromium"),
                    record_video=self.browser_config.get("record_video", False)
                )
                
                browser_agent.set_current_scenario(state['scenario'].name)
                await browser_agent.start_browser()
                
                state["browser_agent"] = browser_agent
                state["element_analyzer"] = self.element_analyzer
                state["current_step"] = 0
                state["step_results"] = []
                state["page_context"] = {}
                state["execution_metadata"] = {
                    "start_time": datetime.now().isoformat(),
                    "scenario_name": state['scenario'].name,
                    "total_steps": len(state['scenario'].steps)
                }
                state["retry_count"] = 0
                state["max_retries"] = 3
                
                print(f"✅ Environment initialized successfully")
                
            except Exception as e:
                state["error_context"] = {
                    "phase": "initialization",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"❌ Initialization failed: {e}")
            
            return state
        
        async def plan_step_execution(state: TestExecutionState) -> TestExecutionState:
            """Plan the execution of the current step using AI"""
            if state.get("error_context"):
                return state
            
            try:
                current_step_idx = state["current_step"]
                if current_step_idx >= len(state["scenario"].steps):
                    return state
                
                current_step = state["scenario"].steps[current_step_idx]
                print(f"\n📋 Planning step {current_step_idx + 1}: {current_step.description}")
                
                # Get current page context
                browser_agent = state["browser_agent"]
                if browser_agent and browser_agent.page:
                    # Capture current page state
                    html_content = await browser_agent.page.content()
                    page_url = browser_agent.page.url
                    page_title = await browser_agent.page.title()
                    
                    state["page_context"] = {
                        "url": page_url,
                        "title": page_title,
                        "html_length": len(html_content),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Use AI to analyze elements for this step
                    if self._is_interaction_step(current_step):
                        print("🧠 Using AI to analyze page elements...")
                        relevant_elements = self.element_analyzer.analyze_page_elements(
                            html_content, current_step.description
                        )
                        state["page_context"]["analyzed_elements"] = relevant_elements
                        print(f"🎯 Found {len(relevant_elements)} relevant elements")
                    
            except Exception as e:
                state["error_context"] = {
                    "phase": "planning",
                    "step": current_step_idx,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"❌ Planning failed: {e}")
            
            return state
        
        async def execute_step(state: TestExecutionState) -> TestExecutionState:
            """Execute the current step with intelligent error handling"""
            if state.get("error_context"):
                return state
            
            step_start_time = datetime.now()
            current_step_idx = state["current_step"]
            
            if current_step_idx >= len(state["scenario"].steps):
                return state
            
            current_step = state["scenario"].steps[current_step_idx]
            browser_agent = state["browser_agent"]
            
            try:
                print(f"🎬 Executing: {current_step.step_type.upper()} - {current_step.description}")
                
                if self._is_navigation_step(current_step):
                    # Handle navigation
                    url = self._extract_url_from_step(current_step.description)
                    success = await browser_agent.navigate_to(url)
                    
                    step_result = StepExecutionResult(
                        step_number=current_step_idx + 1,
                        step_type=current_step.step_type,
                        description=current_step.description,
                        success=success,
                        action_taken=f"navigate_to: {url}",
                        execution_time=(datetime.now() - step_start_time).total_seconds()
                    )
                    
                else:
                    # Handle interaction steps with AI assistance
                    step_result_dict = await browser_agent.execute_step(
                        current_step.description, 
                        current_step_idx + 1
                    )
                    
                    step_result = StepExecutionResult(
                        step_number=current_step_idx + 1,
                        step_type=current_step.step_type,
                        description=current_step.description,
                        success=step_result_dict.get("success", False),
                        action_taken=step_result_dict.get("action_taken"),
                        elements_found=step_result_dict.get("elements_found", []),
                        screenshot_path=step_result_dict.get("screenshot"),
                        execution_time=(datetime.now() - step_start_time).total_seconds(),
                        retry_count=state.get("retry_count", 0)
                    )
                
                # Add AI reasoning if available
                if "analyzed_elements" in state.get("page_context", {}):
                    analyzed = state["page_context"]["analyzed_elements"]
                    if analyzed and len(analyzed) > 0:
                        step_result.reasoning = analyzed[0].get("ai_reasoning", "")
                
                state["step_results"].append(step_result.dict())
                
                if step_result.success:
                    print(f"✅ Step completed successfully")
                    state["current_step"] += 1
                    state["retry_count"] = 0  # Reset retry count on success
                else:
                    print(f"❌ Step failed")
                    # Will be handled by retry logic
                
            except Exception as e:
                step_result = StepExecutionResult(
                    step_number=current_step_idx + 1,
                    step_type=current_step.step_type,
                    description=current_step.description,
                    success=False,
                    execution_time=(datetime.now() - step_start_time).total_seconds(),
                    retry_count=state.get("retry_count", 0)
                )
                
                state["step_results"].append(step_result.dict())
                state["error_context"] = {
                    "phase": "execution",
                    "step": current_step_idx,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"❌ Step execution failed: {e}")
            
            return state
        
        async def handle_step_failure(state: TestExecutionState) -> TestExecutionState:
            """Intelligent failure handling with retry strategies"""
            current_step_idx = state["current_step"]
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 3)
            
            if retry_count < max_retries:
                print(f"🔄 Retry attempt {retry_count + 1}/{max_retries}")
                state["retry_count"] = retry_count + 1
                
                # Add intelligent retry strategies here
                # - Wait for dynamic content
                # - Try alternative selectors
                # - Refresh page if needed
                
                browser_agent = state["browser_agent"]
                if browser_agent:
                    # Wait a bit for dynamic content
                    await asyncio.sleep(2)
                    await browser_agent.wait_for_page_load()
                
                # Don't increment step counter - retry current step
                return state
            else:
                print(f"❌ Step failed after {max_retries} retries")
                # Move to next step or fail scenario
                state["current_step"] += 1
                state["retry_count"] = 0
                return state
        
        async def finalize_execution(state: TestExecutionState) -> TestExecutionState:
            """Finalize test execution and cleanup"""
            try:
                # Close browser
                browser_agent = state.get("browser_agent")
                if browser_agent:
                    await browser_agent.close_browser()
                
                # Update metadata
                execution_metadata = state.get("execution_metadata", {})
                execution_metadata.update({
                    "end_time": datetime.now().isoformat(),
                    "total_steps_executed": len(state.get("step_results", [])),
                    "successful_steps": sum(1 for result in state.get("step_results", []) if result["success"]),
                    "failed_steps": sum(1 for result in state.get("step_results", []) if not result["success"])
                })
                state["execution_metadata"] = execution_metadata
                
                success_rate = (execution_metadata["successful_steps"] / 
                              max(execution_metadata["total_steps_executed"], 1)) * 100
                
                print(f"\n🏁 Test execution completed")
                print(f"📊 Success rate: {success_rate:.1f}%")
                print(f"⏱️  Total execution time: {self._calculate_execution_time(state):.2f}s")
                
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
            
            return state
        
        def should_retry_step(state: TestExecutionState) -> str:
            """Decision node: should we retry the current step?"""
            if state.get("error_context"):
                return "finalize"
            
            current_results = state.get("step_results", [])
            if not current_results:
                return "plan_step"
            
            last_result = current_results[-1]
            if not last_result["success"] and state.get("retry_count", 0) < state.get("max_retries", 3):
                return "retry_step"
            else:
                return "continue"
        
        def should_continue(state: TestExecutionState) -> str:
            """Decision node: should we continue with next step?"""
            if state.get("error_context"):
                return "finalize"
            
            current_step = state.get("current_step", 0)
            total_steps = len(state.get("scenario", {}).get("steps", []))
            
            if current_step < total_steps:
                return "plan_step"
            else:
                return "finalize"
        
        # Build the workflow graph
        workflow = StateGraph(TestExecutionState)
        
        # Add nodes
        workflow.add_node("initialize", initialize_execution)
        workflow.add_node("plan_step", plan_step_execution)
        workflow.add_node("execute_step", execute_step)
        workflow.add_node("retry_step", handle_step_failure)
        workflow.add_node("finalize", finalize_execution)
        
        # Add edges
        workflow.add_edge("initialize", "plan_step")
        workflow.add_edge("plan_step", "execute_step")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "execute_step",
            should_retry_step,
            {
                "retry_step": "retry_step",
                "continue": "plan_step",
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "retry_step",
            should_continue,
            {
                "plan_step": "plan_step",
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "plan_step",
            should_continue,
            {
                "plan_step": "execute_step",
                "finalize": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def execute_scenario(self, scenario: BDDScenario) -> Dict[str, Any]:
        """Execute a BDD scenario using the LangGraph orchestrator"""
        
        initial_state: TestExecutionState = {
            "scenario": scenario,
            "current_step": 0,
            "step_results": [],
            "browser_agent": None,
            "element_analyzer": None,
            "page_context": {},
            "execution_metadata": {},
            "error_context": None,
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Execute the workflow
        config = {"configurable": {"thread_id": f"test_{datetime.now().timestamp()}"}}
        
        try:
            final_state = await self.workflow.ainvoke(initial_state, config)
            
            # Convert to result format
            return {
                "scenario_name": scenario.name,
                "success": self._calculate_overall_success(final_state),
                "steps": final_state.get("step_results", []),
                "execution_metadata": final_state.get("execution_metadata", {}),
                "error_context": final_state.get("error_context"),
                "page_context": final_state.get("page_context", {})
            }
            
        except Exception as e:
            return {
                "scenario_name": scenario.name,
                "success": False,
                "error": str(e),
                "steps": [],
                "execution_metadata": {},
                "timestamp": datetime.now().isoformat()
            }
    
    def _is_navigation_step(self, step: TestStep) -> bool:
        """Check if step is a navigation action"""
        description_lower = step.description.lower()
        return any(keyword in description_lower for keyword in [
            "navigate to", "go to", "visit", "open", "load", "browse to", "i am on"
        ])
    
    def _is_interaction_step(self, step: TestStep) -> bool:
        """Check if step requires element interaction"""
        return not self._is_navigation_step(step)
    
    def _extract_url_from_step(self, description: str) -> str:
        """Extract URL from step description"""
        import re
        
        # Look for URLs in quotes
        url_match = re.search(r'"(https?://[^"]+)"', description)
        if url_match:
            return url_match.group(1)
        
        url_match = re.search(r"'(https?://[^']+)'", description)
        if url_match:
            return url_match.group(1)
        
        # Look for URLs without quotes
        url_match = re.search(r'(https?://\S+)', description)
        if url_match:
            return url_match.group(1)
        
        return ""
    
    def _calculate_overall_success(self, state: TestExecutionState) -> bool:
        """Calculate if the overall test was successful"""
        step_results = state.get("step_results", [])
        if not step_results:
            return False
        
        # All steps must succeed for overall success
        return all(result["success"] for result in step_results)
    
    def _calculate_execution_time(self, state: TestExecutionState) -> float:
        """Calculate total execution time"""
        metadata = state.get("execution_metadata", {})
        start_time = metadata.get("start_time")
        end_time = metadata.get("end_time")
        
        if start_time and end_time:
            from datetime import datetime
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            return (end - start).total_seconds()
        
        return 0.0

if __name__ == "__main__":
    # Example usage
    from .bdd_parser import BDDParser
    
    async def test_orchestrator():
        # Parse a scenario
        parser = BDDParser()
        scenario = parser.parse_scenario_file("scenarios/simple_login_test.xml")
        
        # Create orchestrator with LangChain integration
        config = LangChainConfig(provider="auto", temperature=0.1)
        browser_config = {"headless": True, "record_video": True}
        
        orchestrator = TestOrchestrator(config, browser_config)
        
        # Execute the scenario
        result = await orchestrator.execute_scenario(scenario)
        
        print(f"\n🎯 Final Result:")
        print(f"Scenario: {result['scenario_name']}")
        print(f"Success: {result['success']}")
        print(f"Steps executed: {len(result['steps'])}")
        
        return result
    
    # Run the example
    # asyncio.run(test_orchestrator())