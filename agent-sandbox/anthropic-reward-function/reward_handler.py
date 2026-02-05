import json

{generated_code}

TESTS = [
    ("basic sort", [3, 1, 2], [1, 2, 3]),
    ("empty list", [], []),
    ("already sorted", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ("reverse sorted", [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ("duplicates", [3, 1, 2, 1, 3], [1, 1, 2, 3, 3]),
]

def handler(event, context):
    results = []
    for name, input_list, expected in TESTS:
        try:
            assert sort_list(input_list) == expected
            results.append({{"test": name, "passed": True}})
        except Exception as e:
            results.append({{"test": name, "passed": False, "error": str(e)}})

    score = sum(1 for r in results if r["passed"])
    reward = score / len(TESTS)

    return {{
        "statusCode": 200,
        "headers": {{"Content-Type": "application/json"}},
        "body": json.dumps({{"reward": reward, "score": f"{{score}}/{{len(TESTS)}}", "tests": results}})
    }}
