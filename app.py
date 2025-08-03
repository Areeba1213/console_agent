from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    RunConfig,
    RunContextWrapper,
    TResponseInputItem
)
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
import os
import asyncio

# ================= Load API Key ======================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = AsyncOpenAI(api_key=openai_api_key)

model = OpenAIChatCompletionsModel(
   model="gpt-4o-mini",
    openai_client=client
)

config = RunConfig(
    model=model,
    tracing_disabled=True,
    model_provider=client)

# ================= Pydantic Context =====================
class user_info(BaseModel):
    name: str
    is_premium: bool
    issue_type: str  # 'technical', 'billing', 'refund'

# ================= Tools ================================

@function_tool(
    is_enabled=lambda ctx, agent: ctx.context.issue_type == "refund"
)
def refund(ctx: RunContextWrapper[user_info]) -> str:
    """Process a refund only if the user is premium."""
    if ctx.context.is_premium:
        return f"Refund processed successfully for {ctx.context.name}."
    return f"{ctx.context.name}, you need a premium subscription to request a refund."

@function_tool(
    is_enabled=lambda ctx, agent: not ctx.context.is_premium
)
def check_issue_type(ctx: RunContextWrapper[user_info]) -> str:
    """Return issue type to help route non-premium users."""
    return ctx.context.issue_type

@function_tool(
    is_enabled=lambda ctx, agent: ctx.context.issue_type == "technical")
def restart_service(ctx: RunContextWrapper[user_info]) -> str:
    """Restart the user's service (technical support)."""
    return f"Technical service has been restarted for {ctx.context.name}."

# ================= Main CLI App ============================
async def main():
    # Specialized Agents
    technical_agent = Agent(
        name="technical_agent",
        instructions="You handle technical issues like restarting services, bugs, or errors.",
        tools=[restart_service]
    )

    billing_agent = Agent(
        name="billing_agent",
        instructions="You handle billing questions including payments and charges."
    )

    refund_agent = Agent(
        name="refund_agent",
        instructions="You handle refund-related queries. Only serve premium users.",
        tools=[refund]
    )

    # Triage Agent
    support_agent = Agent(
        name="customer_support_agent",
        instructions="""
    You are a helpful and polite customer support triage agent.

    Your job is to:
    - Read the context.issue_type (technical, billing, refund).
    - Based on that, call the `handoff()` function to pass the conversation to the correct agent:
        - If it's 'technical', handoff to 'technical_agent'
        - If it's 'billing', handoff to 'billing_agent'
        - If it's 'refund', handoff to 'refund_agent'

    Never respond directly. Always use handoff() or tools to handle the issue.
    """,
        handoffs=[technical_agent, billing_agent, refund_agent],
        handoff_description="Delegate to the correct agent using issue_type in context.",
        tools=[check_issue_type],
    )

    print("\n Console Support Agent System Started!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input(" User Input: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print(" Exiting. Thank you!")
            break

        # Ask for user context
        issue_type = input("🔧 Enter issue type (technical / billing / refund): ").strip().lower()
        premium_input = input(" Are you a premium user? (yes / no): ").strip().lower()
        is_premium = premium_input in ["yes", "y"]

        user_data = user_info(
            name="Areeba",
            is_premium=is_premium,
            issue_type=issue_type
        )

        print("\n Agent response:\n")

        result =  Runner.run_streamed(
            support_agent,
            input=user_input,
            context=user_data,
            run_config=config
        )

        print(result.final_output)

        async for event in result.stream():
            if hasattr(event, "name") and event.name:
                print(f"\nEvent Triggered: {event.name}")
            if hasattr(event, "delta") and event.delta:
                print(event.delta, end="", flush=True)
                print("\n" + "-" * 60 + "\n")

# ================= Entry ================================
if __name__ == "__main__":
    asyncio.run(main())


