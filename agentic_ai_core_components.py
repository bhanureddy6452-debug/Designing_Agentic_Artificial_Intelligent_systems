"""
Agentic AI Architecture: Core Components Showcase
===================================================
This script demonstrates the 5 foundational modules of an Autonomous Agent:
1. Perception Module      -> Senses & extracts structured observations from raw input.
2. Memory Module          -> Manages Working (Short-term) and Episodic/Semantic (Long-term) memory.
3. Goal Management Module -> Defines, decomposes, prioritizes, and tracks goal states.
4. Planning Module        -> Generates step-by-step plans (ReAct/CoT style) and handles re-planning.
5. Action Module          -> Executes tools/actions and returns feedback into the agent loop.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import time
import json


# ============================================================================
# 1. PERCEPTION MODULE
# ============================================================================
@dataclass
class Observation:
    """Represents processed, structured perceptual information."""
    raw_input: str
    intent: str
    entities: Dict[str, Any]
    environment_state: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

class PerceptionModule:
    """
    Perception senses the external environment (user input, sensor feeds, API events),
    filters noise, and transforms raw unstructured data into structured observations.
    """
    def __init__(self, agent_name: str = "Agent-01"):
        self.agent_name = agent_name

    def perceive(self, raw_input: str, environment_state: Dict[str, Any]) -> Observation:
        print(f"\n[1. PERCEPTION] Ingesting raw signal: '{raw_input}'")
        
        # Rule-based / NLU parsing simulation
        lowered = raw_input.lower()
        if "analyze" in lowered or "research" in lowered:
            intent = "RESEARCH_AND_REPORT"
        elif "buy" in lowered or "order" in lowered:
            intent = "EXECUTE_PURCHASE"
        elif "status" in lowered:
            intent = "CHECK_SYSTEM_STATUS"
        else:
            intent = "GENERAL_ASSISTANCE"

        # Entity extraction simulation
        entities = {}
        if "market" in lowered or "sales" in lowered:
            entities["domain"] = "sales_market"
        if "competitor" in lowered:
            entities["target"] = "competitor_data"
        if "urgent" in lowered:
            entities["priority"] = "HIGH"

        observation = Observation(
            raw_input=raw_input,
            intent=intent,
            entities=entities,
            environment_state=environment_state
        )
        print(f"      -> Structured Observation: Intent='{intent}', Entities={entities}")
        return observation


# ============================================================================
# 2. MEMORY MODULE
# ============================================================================
@dataclass
class MemoryRecord:
    key: str
    content: Any
    category: str  # 'working', 'episodic', 'semantic'
    timestamp: float = field(default_factory=time.time)

class MemoryModule:
    """
    Memory provides temporal consistency across agent operations:
    - Working Memory (Short-Term): Scratchpad, current context, recent observations.
    - Long-Term Memory:
        - Episodic: Past experiences, past successful/failed action sequences.
        - Semantic: Knowledge base, domain facts, user preferences.
    """
    def __init__(self):
        self.working_memory: List[MemoryRecord] = []
        self.episodic_memory: List[MemoryRecord] = []
        self.semantic_memory: Dict[str, Any] = {
            "user_name": "Dr. Alex",
            "preferred_format": "bullet_points",
            "access_tier": "admin",
            "known_tools": ["WebSearchTool", "DataAnalysisTool", "EmailNotificationTool"]
        }

    def record_working_memory(self, key: str, content: Any):
        rec = MemoryRecord(key=key, content=content, category="working")
        self.working_memory.append(rec)
        print(f"[2. MEMORY] [Working] Cached '{key}': {content}")

    def record_episodic_memory(self, experience: str, outcome: str):
        rec = MemoryRecord(key=experience, content=outcome, category="episodic")
        self.episodic_memory.append(rec)
        print(f"[2. MEMORY] [Episodic] Learned from experience: '{experience}' => '{outcome}'")

    def query_semantic(self, key: str) -> Optional[Any]:
        return self.semantic_memory.get(key)

    def search_episodic(self, query: str) -> List[MemoryRecord]:
        return [m for m in self.episodic_memory if query.lower() in m.key.lower()]

    def get_context_snapshot(self) -> Dict[str, Any]:
        """Returns active memory state to help Planning & Goal Management."""
        return {
            "working_memory_items": len(self.working_memory),
            "recent_context": [m.content for m in self.working_memory[-3:]],
            "user_preferences": self.semantic_memory
        }


# ============================================================================
# 3. GOAL MANAGEMENT MODULE
# ============================================================================
class GoalStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class SubGoal:
    id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    result: Optional[Any] = None

@dataclass
class Goal:
    id: str
    title: str
    priority: int  # 1 = Highest
    subgoals: List[SubGoal] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING

class GoalManagementModule:
    """
    Goal Management formulates overarching objectives from perception,
    breaks them down into manageable sub-goals, prioritizes, and tracks state.
    """
    def __init__(self):
        self.goals: List[Goal] = []

    def create_goal_from_observation(self, observation: Observation) -> Goal:
        goal_id = f"GOAL-{len(self.goals)+1:03d}"
        priority = 1 if observation.entities.get("priority") == "HIGH" else 2
        
        goal = Goal(
            id=goal_id,
            title=f"Resolve {observation.intent}: {observation.raw_input}",
            priority=priority
        )

        # Decompose into sub-goals
        if observation.intent == "RESEARCH_AND_REPORT":
            goal.subgoals = [
                SubGoal(id=f"{goal_id}.1", description="Gather raw data on market and competitors"),
                SubGoal(id=f"{goal_id}.2", description="Analyze trends and extract key metrics"),
                SubGoal(id=f"{goal_id}.3", description="Format and deliver final report to user")
            ]
        else:
            goal.subgoals = [
                SubGoal(id=f"{goal_id}.1", description="Execute generic task step 1"),
                SubGoal(id=f"{goal_id}.2", description="Summarize outcome")
            ]

        self.goals.append(goal)
        print(f"\n[3. GOAL MANAGEMENT] Created {goal.id} (Priority {goal.priority}): '{goal.title}'")
        for sg in goal.subgoals:
            print(f"      |-- Sub-Goal {sg.id}: {sg.description} [{sg.status.value}]")
        return goal

    def update_subgoal_status(self, subgoal: SubGoal, status: GoalStatus, result: Any = None):
        subgoal.status = status
        subgoal.result = result
        print(f"[3. GOAL MANAGEMENT] Sub-Goal {subgoal.id} status updated -> {status.value}")


# ============================================================================
# 4. PLANNING MODULE
# ============================================================================
@dataclass
class PlanStep:
    step_number: int
    action_name: str
    parameters: Dict[str, Any]
    rationale: str

class PlanningModule:
    """
    Planning synthesizes the goal, perceptual observation, and memory context
    to generate an ordered trajectory of executable actions.
    Supports dynamic re-planning on failures.
    """
    def __init__(self):
        pass

    def generate_plan(self, goal: Goal, memory_snapshot: Dict[str, Any]) -> List[PlanStep]:
        print(f"\n[4. PLANNING] Devising strategic plan for {goal.id}...")
        plan: List[PlanStep] = []

        pref = memory_snapshot["user_preferences"].get("preferred_format", "standard")

        for idx, sg in enumerate(goal.subgoals, start=1):
            if "Gather" in sg.description:
                step = PlanStep(
                    step_number=idx,
                    action_name="WebSearchTool",
                    parameters={"query": "2026 Q3 AI sales competitor analysis"},
                    rationale="Episodic memory indicates fresh search yields the most accurate data."
                )
            elif "Analyze" in sg.description:
                step = PlanStep(
                    step_number=idx,
                    action_name="DataAnalysisTool",
                    parameters={"metric": "growth_rate", "confidence_threshold": 0.85},
                    rationale="Required to compute trends prior to final presentation."
                )
            elif "Format" in sg.description:
                step = PlanStep(
                    step_number=idx,
                    action_name="ReportGeneratorTool",
                    parameters={"format": pref, "recipient": memory_snapshot["user_preferences"].get("user_name")},
                    rationale=f"Align output with user preference ({pref})."
                )
            else:
                step = PlanStep(
                    step_number=idx,
                    action_name="DefaultTool",
                    parameters={},
                    rationale="Standard fallback step."
                )
            plan.append(step)

        print(f"      Plan generated with {len(plan)} steps:")
        for step in plan:
            print(f"      Step {step.step_number}: Call '{step.action_name}' (Rationale: {step.rationale})")
        return plan


# ============================================================================
# 5. ACTION MODULE
# ============================================================================
class ToolRegistry:
    """Simulated environment tools callable by the Action module."""
    @staticmethod
    def WebSearchTool(query: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "data": f"Found 3 key reports on '{query}': Competitor A grew 15%, Market share is 42%."
        }

    @staticmethod
    def DataAnalysisTool(metric: str, confidence_threshold: float) -> Dict[str, Any]:
        return {
            "status": "success",
            "insights": f"Metric '{metric}' analyzed with {confidence_threshold*100:.0f}% confidence: Strong upward momentum."
        }

    @staticmethod
    def ReportGeneratorTool(format: str, recipient: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "report": f"Formatted ({format}) Report delivered to {recipient}:\n"
                      f"  * Market growth remains resilient.\n"
                      f"  * Competitor A expanded by 15%.\n"
                      f"  * Recommendation: Capitalize on untapped market sectors."
        }

class ActionModule:
    """
    Action is the effectuator. It interfaces with external APIs, databases,
    tools, or human interfaces to effectuate the plan steps in the environment.
    """
    def __init__(self):
        self.tools = {
            "WebSearchTool": ToolRegistry.WebSearchTool,
            "DataAnalysisTool": ToolRegistry.DataAnalysisTool,
            "ReportGeneratorTool": ToolRegistry.ReportGeneratorTool
        }

    def execute(self, step: PlanStep) -> Dict[str, Any]:
        print(f"\n[5. ACTION] Executing Step {step.step_number}: {step.action_name}...")
        tool = self.tools.get(step.action_name)
        
        if not tool:
            error_result = {"status": "error", "message": f"Tool '{step.action_name}' not found"}
            print(f"      -> Execution Failed: {error_result}")
            return error_result

        # Call the tool with parameters
        result = tool(**step.parameters)
        print(f"      -> Action Output: {json.dumps(result)}")
        return result


# ============================================================================
# ORCHESTRATOR: AGENTIC AI FEEDBACK LOOP
# ============================================================================
class AutonomousAgent:
    """
    Ties all 5 core components together in the continuous Sense-Plan-Act feedback loop.
    """
    def __init__(self, name: str):
        self.name = name
        self.perception = PerceptionModule(agent_name=name)
        self.memory = MemoryModule()
        self.goal_manager = GoalManagementModule()
        self.planner = PlanningModule()
        self.action = ActionModule()

    def run_agent_loop(self, user_instruction: str, env_state: Dict[str, Any]):
        print("=" * 80)
        print(f"[AGENT] STARTING AGENTIC RUN: '{self.name}'")
        print("=" * 80)

        # 1. PERCEPTION
        observation = self.perception.perceive(user_instruction, env_state)
        self.memory.record_working_memory("current_observation", observation.intent)

        # 2. GOAL MANAGEMENT
        goal = self.goal_manager.create_goal_from_observation(observation)
        goal.status = GoalStatus.IN_PROGRESS

        # 3. PLANNING
        memory_snapshot = self.memory.get_context_snapshot()
        plan = self.planner.generate_plan(goal, memory_snapshot)

        # 4. ACTION & FEEDBACK LOOP
        for step, subgoal in zip(plan, goal.subgoals):
            # Execute step
            action_result = self.action.execute(step)
            
            # Store in Working Memory
            self.memory.record_working_memory(
                key=f"step_{step.step_number}_result",
                content=action_result
            )

            # Update Goal state based on execution
            if action_result.get("status") == "success":
                self.goal_manager.update_subgoal_status(subgoal, GoalStatus.COMPLETED, action_result)
            else:
                self.goal_manager.update_subgoal_status(subgoal, GoalStatus.FAILED, action_result)
                print(f"[RE-PLANNING TRIGGERED] Step {step.step_number} failed. Re-evaluating...")
                break

        # Final Goal Completion Check
        all_completed = all(sg.status == GoalStatus.COMPLETED for sg in goal.subgoals)
        if all_completed:
            goal.status = GoalStatus.COMPLETED
            # Consolidate into Long-term Episodic Memory for future retrieval
            self.memory.record_episodic_memory(
                experience=f"Executed task: {goal.title}",
                outcome="Successfully produced and delivered final analysis"
            )
            print("\n" + "=" * 80)
            print(f"[SUCCESS] GOAL ACHIEVED: {goal.id} is {goal.status.value}!")
            print("=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    agent = AutonomousAgent("Nexus-Analyst")

    # Environmental input simulating a user command
    user_prompt = "Urgent: Please analyze market competitor data and prepare a summary report."
    initial_environment = {
        "network_available": True,
        "database_connected": True,
        "battery_level": 98
    }

    # Run the agent through all 5 modules
    agent.run_agent_loop(user_prompt, initial_environment)
