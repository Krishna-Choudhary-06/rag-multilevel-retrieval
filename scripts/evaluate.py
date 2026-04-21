import json
from src.pipeline.rag_pipeline import RAGPipeline


def evaluate():
    rag = RAGPipeline()

    with open("tests/eval_data.json", "r") as f:
        data = json.load(f)

    total = len(data)
    correct = 0

    for item in data:
        query = item["query"]
        expected = item["expected_keywords"]

        print(f"\n[QUERY] {query}")

        response = rag.run(query)
        answer = response["answer"].lower()

        match_count = sum(1 for word in expected if word in answer)

        print("Matched keywords:", match_count, "/", len(expected))

        if match_count >= len(expected) // 2:
            correct += 1

    accuracy = correct / total

    print("\n=== FINAL SCORE ===")
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    evaluate()