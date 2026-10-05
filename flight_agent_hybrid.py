"""BTVN#3 - Hybrid flight agent: Plan first, then adapt with a ReAct loop.

Flow:
  1. Search and let a planner propose a complete booking plan.
  2. A human reviews the plan.
  3. The harness checks each action before execution.
  4. If an action is denied, hand the observed state to a ReAct agent so it
     can inspect the options and recover (or request human help).

All tools and models are mocked; this demo does not need an API key.
Run from the repository root: python BTVN03/flight_agent_hybrid.py
"""
import json
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel


@dataclass
class Constraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000

    def to_prompt(self) -> str:
        return (f"Book one ticket {self.origin} -> {self.destination} on {self.date}, "
                f"departing before {self.depart_before}, price at most {self.max_price:,} VND.")

    def is_ok(self, flight: dict) -> bool:
        return (flight["depart"].startswith(self.date)
                and flight["depart"][11:16] < self.depart_before
                and flight["price"] <= self.max_price)


CONSTRAINTS = Constraints()
FLIGHTS_BY_CASE = {
    "success": [
        {"flight": "VN122", "depart": "2026-10-07T08:10", "price": 1_850_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
    # Same inventory as success, but the planner initially chooses the wrong flight.
    "recovery": [
        {"flight": "VN122", "depart": "2026-10-07T08:10", "price": 1_850_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
    "fail": [
        {"flight": "VJ604", "depart": "2026-10-07T08:10", "price": 2_480_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
}
CASE = "success"
FLIGHTS = []
BOOKINGS = {}
LOG = []


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search mock flights by route and travel date."""
    return {"status": "ok", "flights": FLIGHTS}


@tool
def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Does not charge the customer."""
    info = next((f for f in FLIGHTS if f["flight"] == flight), None)
    if info is None:
        return {"status": "not_found", "flight": flight}
    code = f"{flight}-12A"
    BOOKINGS[code] = {"code": code, "paid": False, **info}
    return {"status": "ok", **BOOKINGS[code]}


@tool
def pay(code: str) -> dict:
    """Pay for a held booking."""
    if code not in BOOKINGS:
        return {"status": "not_found", "code": code}
    BOOKINGS[code]["paid"] = True
    return {"status": "ok", **BOOKINGS[code]}


@tool
def get_booking(code: str) -> dict:
    """Read a booking back from the mock booking database."""
    return {"status": "ok", **BOOKINGS[code]} if code in BOOKINGS else {"status": "not_found"}


TOOLS = [search_flights, book_seat, pay, get_booking]
TOOL_MAP = {t.name: t for t in TOOLS}


class Step(BaseModel):
    tool: str
    args: dict


class Plan(BaseModel):
    steps: list[Step]


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Create a complete flight booking plan as tool steps. "
                "Only select a flight meeting every constraint; otherwise return an empty plan. "
                "Use $booking_code in pay/get_booking steps."),
    ("human", "Goal: {goal}\nAvailable flights: {flights}"),
])


def check_permission(tool_name: str, args: dict) -> str | None:
    """Authorize a tool action against constraint data before it runs."""
    if tool_name not in TOOL_MAP:
        return f"Unknown tool: {tool_name}"
    if tool_name == "book_seat":
        flight = next((f for f in FLIGHTS if f["flight"] == args.get("flight")), None)
        if flight is None or not CONSTRAINTS.is_ok(flight):
            return f"{args.get('flight')} does not meet: {CONSTRAINTS.to_prompt()}"
    if tool_name == "pay" and args.get("code") not in BOOKINGS:
        return f"Cannot pay: booking {args.get('code')} does not exist."
    return None


def is_done() -> bool:
    """The harness verifies the persisted booking instead of trusting the model."""
    for code in BOOKINGS:
        booking = TOOL_MAP["get_booking"].invoke({"code": code})
        if booking.get("paid") and CONSTRAINTS.is_ok(booking):
            return True
    return False


def handoff(plan_rejected: bool = False) -> dict:
    """The question is built from the current state, so it never tells the user something false."""
    tried = {args.get("flight") for name, args, _ in LOG if name == "book_seat"}
    valid = [f["flight"] for f in FLIGHTS if CONSTRAINTS.is_ok(f)]
    untried = [f for f in valid if f not in tried]
    if plan_rejected and valid:
        question = (f"The plan was rejected. Valid flight(s): {', '.join(valid)}. "
                    "Book one of these, or change the constraints?")
    elif untried:
        question = f"Valid flight(s) not tried yet: {', '.join(untried)}. Book {untried[0]}?"
    else:
        question = "No flight satisfies all constraints. May I relax the departure time or price limit?"
    return {
        "done_so_far": [f"{code}: paid={booking['paid']}" for code, booking in BOOKINGS.items()]
        or ["Nothing booked, nothing paid"],
        "tried": [f"{name}({args}) -> {result.get('status')}" for name, args, result in LOG],
        "question": question,
    }


def fill_placeholders(args: dict) -> dict:
    code = next((r["code"] for n, _, r in reversed(LOG)
                 if n == "book_seat" and r.get("status") == "ok"), None)
    return {k: (code if v == "$booking_code" else v) for k, v in args.items()}


@wrap_tool_call
def harness(request, handler):
    """All ReAct tool calls pass through permission checks and are logged."""
    call = request.tool_call
    name, args = call["name"], call["args"]
    reason = check_permission(name, args)
    if reason:
        result = {"status": "denied", "reason": reason}
    else:
        result = TOOL_MAP[name].invoke(args)
    LOG.append((name, args, result))
    return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])


def scripted_models(case: str):
    """Deterministic stand-ins for planner and ReAct models."""
    from langchain_core.language_models import GenericFakeChatModel

    flight = {"success": "VN122", "recovery": "QH118", "fail": "VJ604"}[case]
    plan = Plan(steps=[
        Step(tool="book_seat", args={"flight": flight}),
        Step(tool="pay", args={"code": "$booking_code"}),
        Step(tool="get_booking", args={"code": "$booking_code"}),
    ])

    class PlannerModel(GenericFakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: plan)

    planner_model = PlannerModel(messages=iter([]))

    def call(name: str, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}-{len(LOG)}-{len(BOOKINGS)}"}])

    # On recovery, ReAct searches again and chooses the remaining valid option.
    fallback_scripts = {
        "success": [],
        "recovery": [
            call("search_flights", origin=CONSTRAINTS.origin,
                 destination=CONSTRAINTS.destination, date=CONSTRAINTS.date),
            call("book_seat", flight="VN122"),
            call("pay", code="VN122-12A"),
            call("get_booking", code="VN122-12A"),
            AIMessage(content="Recovered by selecting valid flight VN122."),
        ],
        "fail": [
            call("search_flights", origin=CONSTRAINTS.origin,
                 destination=CONSTRAINTS.destination, date=CONSTRAINTS.date),
            AIMessage(content="No flight meets all constraints; human help is required."),
        ],
    }

    class ReActModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return planner_model, ReActModel(messages=iter(fallback_scripts[case]))


def ask_human(plan: Plan) -> bool:
    """The human reviews the plan AFTER it has been printed."""
    answer = ""
    while answer not in ("y", "n"):
        answer = input("Human reviewer - approve this plan? (y/n): ").strip().lower()
    return answer == "y"


def run_case(case: str, approve=ask_human) -> dict:
    """approve: a function plan -> bool (default: ask a human), or a fixed True/False for tests."""
    global CASE, FLIGHTS
    CASE = case
    FLIGHTS = FLIGHTS_BY_CASE[case]
    BOOKINGS.clear()
    LOG.clear()

    c = CONSTRAINTS
    search_args = {"origin": c.origin, "destination": c.destination, "date": c.date}
    found = TOOL_MAP["search_flights"].invoke(search_args)
    LOG.append(("search_flights", search_args, found))

    planner_model, react_model = scripted_models(case)
    planner = PLANNER_PROMPT | planner_model.with_structured_output(Plan)
    plan = planner.invoke({"goal": c.to_prompt(), "flights": json.dumps(found["flights"])})

    print(f"\n--- plan ({case}) ---")
    for i, step in enumerate(plan.steps, 1):
        print(f"  Step {i}: {step.tool}({step.args})")

    approved = approve(plan) if callable(approve) else approve
    if not approved:
        print("Human rejected the plan; handing off for approval/replanning.")
        output = {"result": "FAILED", "reason": "plan_rejected", "handoff": handoff(plan_rejected=True)}
        print(json.dumps(output, indent=2))
        return output

    recovery_needed = False
    recovery_reason = "The planned step did not complete successfully."
    for i, step in enumerate(plan.steps, 1):
        args = fill_placeholders(step.args)
        reason = check_permission(step.tool, args)
        if reason:
            result = {"status": "denied", "reason": reason}
            LOG.append((step.tool, args, result))
            print(f"  Plan step {i} blocked: {reason}")
            recovery_needed = True
            recovery_reason = reason
            break
        result = TOOL_MAP[step.tool].invoke(args)
        LOG.append((step.tool, args, result))
        print(f"  Plan step {i}: {step.tool} -> {result['status']}")
        if result.get("status") != "ok":
            recovery_needed = True
            recovery_reason = f"{step.tool} returned {result}."
            break

    if recovery_needed:
        print("\n--- ReAct recovery ---")
        agent = create_agent(
            model=react_model,
            tools=TOOLS,
            system_prompt=("You are a flight booking agent. Search available flights, "
                           "obey all constraints, and ask for human help if none qualify."),
            middleware=[harness, ModelCallLimitMiddleware(run_limit=10)],
        )
        recovery_request = (f"{c.to_prompt()} The approved plan could not continue: {recovery_reason} "
                            "Inspect the available flights and recover if a valid option remains.")
        out = agent.invoke({"messages": [{"role": "user", "content": recovery_request}]})
        print("ReAct response:", out["messages"][-1].content)

    if is_done():
        output = {"result": "DONE", "booking": list(BOOKINGS.values())}
    else:
        output = {"result": "FAILED", "handoff": handoff()}
    print("\n--- trace ---")
    for i, (name, args, result) in enumerate(LOG, 1):
        print(f"[{i}] {name}({args}) -> {result.get('status')}")
    print("\n--- result ---")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    choices = {"1": "success", "2": "recovery", "3": "fail"}
    print("Choose a scenario:")
    print("  1. success  - plan is valid")
    print("  2. recovery - plan is blocked, ReAct finds a valid alternative")
    print("  3. fail     - no flight satisfies all constraints")
    choice = ""
    while choice not in choices:
        choice = input("Your choice (1-3): ").strip()
    run_case(choices[choice])          # the plan is printed first, then the human approves it
