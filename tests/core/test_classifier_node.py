import pytest
from atlasmind.core.classifier_node import ClassifierNode
from atlasmind.core.classification.base import ReasoningCategory


@pytest.fixture
def node():
    """Fixture for initializing ClassifierNode."""
    return ClassifierNode()


@pytest.mark.parametrize(
    "name,task,expected_type",
    [
        (
            "Knowledge Retrieval",
            {
                "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?",
            },
            ReasoningCategory.KNOWLEDGE_RETRIEVAL,
        ),
        (
            "Audio Reasoning",
            {
                "question": "Listen to the attached mp3 and summarize it.",
            },
            ReasoningCategory.AUDIO_REASONING,
        ),
        (
            "Audio Reasoning – file extension",
            {"question": "Transcribe this audio.", "file_path": "sample/homework.mp3"},
            ReasoningCategory.AUDIO_REASONING,
        ),

        (
            "Video Reasoning",
            {
                "question": "In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what happens at the end?",
            },
            ReasoningCategory.VIDEO_REASONING,
        ),
        (
            "Structured Data",
            {
                "question": "The attached Excel file contains menu sales. Calculate total food revenue.",
            },
            ReasoningCategory.STRUCTURED_DATA,
        ),
        (
            "Code Execution",
            {
                "question": "Run the attached Python file and return its numeric output.",
            },
            ReasoningCategory.CODE_EXECUTION,
        ),
        (
            "Code Execution – file extension",
            {"question": "Execute this file.", "file_path": "tmp/code.py"},
            ReasoningCategory.CODE_EXECUTION,
        ),
        (
            "Structured Data – file extension",
            {"question": "Analyze this spreadsheet.", "file_path": "tmp/sales.xlsx"},
            ReasoningCategory.STRUCTURED_DATA,
        ),
        (
            "Visual Reasoning",
            {
                "question": "Review the chess position provided in the attached image and find the best move.",
            },
            ReasoningCategory.VISUAL_REASONING,
        ),
        (
            "Visual Reasoning – file extension",
            {"question": "How many pieces are on this chessboard?", "file_path": "tmp/chess_position.png"},
            ReasoningCategory.VISUAL_REASONING,
        ),
        (
            "Some Random Logic",
            {
                "question": "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
            }
            ,
            ReasoningCategory.FALLBACK_SEARCH,
        ),
        (
            "Media Data Lookup",
            {
                "question": "Who played Ray in the Polish-language version of Everybody Loves Raymond?",
            },
            ReasoningCategory.KNOWLEDGE_RETRIEVAL,
        ),
        (
            "Semantic Categorization",
            {
                "question": (
                    "I'm making a grocery list for my mom, but she's a professor of botany "
                    "and she's a real stickler when it comes to categorizing things. "
                    "I need to add different foods to different categories on the grocery list, "
                    "but if I make a mistake, she won't buy anything inserted in the wrong category. "
                    "Here's the list I have so far:\n\nmilk, eggs, flour, whole bean coffee, Oreos, "
                    "sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, "
                    "whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts\n\n"
                    "I need to make headings for the fruits and vegetables. Could you please "
                    "create a list of just the vegetables from my list? Please alphabetize them."
                ),
            },
            ReasoningCategory.SEMANTIC_CATEGORIZATION,
        ),
        (
            "Unknown Fallback",
            {
                "question": "Tell me something random with no clue about type.",
            },
            ReasoningCategory.FALLBACK_SEARCH,
        ),
    ],
)
def test_all_classifiers(node, name, task, expected_type):
    """Test all classifiers through ClassifierNode using parameterized test cases."""
    result = node.classify(task)

    print(result)
    assert result.reasoning_type == expected_type, f"{name} failed: wrong reasoning_type"
    
