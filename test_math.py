 .t"""Quick test for deterministic math patterns."""
from agent.evaluator import try_solve_locally

tests = [
    ("math", "A shop offers 20% discount on an item priced at $150. What is the sale price?"),
    ("math", "What is the area of a rectangle with length 8 and width 5?"),
    ("math", "If a rectangle has length 12 cm and width 8 cm, what is its perimeter?"),
    ("math", "If a bacteria population doubles every 12 hours, and starts with 100 bacteria, how many will there be after exactly 3 days?"),
    ("math", "If a train leaves Station A at 60 mph and another leaves Station B at 90 mph heading towards each other, and they are 300 miles apart, how long until they meet in hours?"),
    ("math", "I have 60 more apples than oranges. If I have 100 fruits in total, how many oranges do I have?"),
    ("math", "Solve for x in the equation: 3(x - 4) + 5 = 2(x + 1)"),
]

for domain, prompt in tests:
    result = try_solve_locally(domain, prompt)
    print(f"{prompt[:60]:60s} => {result}")