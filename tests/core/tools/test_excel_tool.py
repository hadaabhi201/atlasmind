import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from atlasmind.core.tools.excel_tool import ExcelTool
from atlasmind.core.planning.base_plan import PlanTemplate, ToolType
from atlasmind.core.tools.base_tool import ExecutionStatus


@pytest.fixture
def sample_excel_file(tmp_path):
    """Create a valid Excel file with sample data."""
    df = pd.DataFrame({
        "Item": ["A", "B", "C"],
        "Category": ["Food", "Drink", "Food"],
        "Sales (USD)": [100.5, 50.25, 75.75]
    })
    file_path = tmp_path / "sample_data.xlsx"
    df.to_excel(file_path, index=False, engine="openpyxl")
    return str(file_path)


@pytest.fixture
def sample_plan(sample_excel_file):
    """Attach the Excel file to the plan as file_path."""
    return PlanTemplate(
        question="Summarize the Excel data.",
        tool=ToolType.EXCEL_READER,
        plan_steps=["Load Excel", "Summarize"],
        file_path=sample_excel_file
    )


def test_excel_tool_success(sample_plan):
    """ExcelTool should load, summarize, and return structured Excel metadata."""
    tool = ExcelTool()
    result = tool.execute(sample_plan)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.tool == ToolType.EXCEL_READER
    assert result.data is not None

    data = result.data

    assert "columns" in data
    assert "preview_rows" in data
    assert "numeric_summary" in data
    assert data["row_count"] == 3

    assert "Sales (USD)" in data["columns"]
    assert data["preview_rows"][0]["Item"] == "A"
    assert "Sales (USD)" in data["numeric_summary"]

def test_excel_tool_invalid_file_path():
    tool = ExcelTool()

    plan = PlanTemplate(
        question="Test missing Excel",
        tool=ToolType.EXCEL_READER,
        file_path=None
    )

    with pytest.raises(RuntimeError, match="Invalid file_path"):
        tool.execute(plan)


def test_excel_tool_nonexistent_file():
    tool = ExcelTool()

    plan = PlanTemplate(
        question="Test nonexistent file",
        tool=ToolType.EXCEL_READER,
        file_path="/tmp/this_file_does_not_exist.xlsx"
    )

    with pytest.raises(RuntimeError, match="Invalid file_path"):
        tool.execute(plan)

def test_excel_tool_empty_excel(tmp_path):
    """ExcelTool must raise error for empty Excel sheet."""
    empty_file = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(empty_file, index=False, engine="openpyxl")

    plan = PlanTemplate(
        question="Summarize empty Excel",
        tool=ToolType.EXCEL_READER,
        file_path=str(empty_file)
    )

    tool = ExcelTool()

    with pytest.raises(RuntimeError, match="Excel file is empty"):
        tool.execute(plan)