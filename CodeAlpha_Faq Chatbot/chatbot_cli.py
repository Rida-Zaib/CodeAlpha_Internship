"""Simple terminal chatbot. Run with: python chatbot_cli.py"""

from matcher import FAQMatcher

FAQ_FILE = "faqs.csv"


def main():
    print("TechNest FAQ Bot — ask me a question (type 'exit' to quit)\n")
    matcher = FAQMatcher(FAQ_FILE)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye!")
            break
        if not user_input:
            continue

        r = matcher.get_answer(user_input)
        print(f"Bot: {r['answer']}")
        if r["matched_question"]:
            tag = f"[{r['category']}] confidence={r['confidence']} score={r['score']} ({r['response_ms']} ms)"
            print(f"     {tag}")
            if r["ambiguous"] and r["alternatives"]:
                alts = ", ".join(a["question"] for a in r["alternatives"])
                print(f"     Did you also mean: {alts}?")
        print()


if __name__ == "__main__":
    main()
