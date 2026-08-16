from app.rag import rag
eval_set = [
    {
        "question": "What is machine learning?",
        "reference": (
            "Machine learning is the study of computer programs that "
            "improve their performance on a task through experience."
        ),
    },
    {
        "question": "What is supervised learning?",
        "reference": (
            "Supervised learning is learning a function that maps inputs "
            "to outputs from a set of labeled training examples."
        ),
    },
    {
        "question": "What is unsupervised learning?",
        "reference": (
            "Unsupervised learning is learning patterns or structure "
            "from data that has no labeled outputs."
        ),
    },
    {
        "question": "What is reinforcement learning?",
        "reference": (
            "Reinforcement learning is learning what actions to take, "
            "given a state, in order to maximize a numerical reward signal over time."
        ),
    },
    {
        "question": "What are the main applications of machine learning?",
        "reference": (
            "Machine learning is applied in areas such as data mining, "
            "speech and image recognition, fraud detection, autonomous "
            "vehicles, and information-filtering systems."
        ),
    },
]

eval_rows = []

for item in eval_set:
    answer, sources = rag(item["question"])

    eval_rows.append({
        "user_input": item["question"],
        "response": answer,
        "retrieved_contexts": [doc.page_content for doc in sources],
        "reference": item["reference"],
    })

print(f"Evaluation samples: {len(eval_rows)}")
