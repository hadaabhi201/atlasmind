import argparse
import sys
from atlasmind.agents.basic_agent import BasicAgent
from atlasmind.utils.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AtlasMind agent with a question and optional file."
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the agent (required)."
    )

    parser.add_argument(
        "--file_path",
        type=str,
        required=False,
        default=None,
        help="Optional path to an attached file (image, audio, code, excel, etc)."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    agent = BasicAgent(timeout=90, verbose=True)

    try:
        result = agent(
            question=args.question,
            file_path=args.file_path
        )

        # result is StopEvent.result dict
        

        if result:
            print("\n=== Final Answer ===\n")
            print(result)
        else:
            print("\nNo model answer returned.\n")

    except Exception as e:
        logger.error(f"[Main] Agent failed: {e}")
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
