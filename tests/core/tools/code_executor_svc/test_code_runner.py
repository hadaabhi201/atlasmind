import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from atlasmind.core.tools.code_executor_svc.code_runner import CodeRunner


def test_code_runner_init():
    """Test CodeRunner initialization."""
    runner = CodeRunner()
    assert hasattr(runner, 'url')
    assert hasattr(runner, 'host')
    assert hasattr(runner, 'key')


@patch('atlasmind.core.tools.code_executor_svc.code_runner.requests.post')
def test_run_success(mock_post):
    """Test successful code execution."""
    runner = CodeRunner()
    
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("hello world")')
        temp_path = f.name
    
    try:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": {"description": "Accepted"},
            "stdout": "hello world\n",
            "stderr": "",
            "exit_code": 0
        }
        mock_post.return_value = mock_response
        
        result = runner.run(temp_path)
        
        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "base64_encoded=false&wait=true" in call_args[0][0]
        assert call_args[1]['headers']['Content-Type'] == 'application/json'
        assert call_args[1]['headers']['X-RapidAPI-Key'] == runner.key
        assert call_args[1]['timeout'] == 30
        
        # Verify payload
        payload = call_args[1]['json']
        assert payload['language_id'] == 71  # Python
        assert 'print("hello world")' in payload['source_code']
        
        # Verify result
        assert result['status']['description'] == 'Accepted'
        assert result['stdout'] == 'hello world\n'
        
    finally:
        # Cleanup
        os.unlink(temp_path)


@patch('atlasmind.core.tools.code_executor_svc.code_runner.requests.post')
def test_run_http_error(mock_post):
    """Test HTTP error during code execution."""
    runner = CodeRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("test")')
        temp_path = f.name
    
    try:
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_post.return_value = mock_response
        
        with pytest.raises(RuntimeError, match="CodeRunner failed:"):
            runner.run(temp_path)
            
    finally:
        os.unlink(temp_path)


@patch('atlasmind.core.tools.code_executor_svc.code_runner.requests.post')
def test_run_network_error(mock_post):
    """Test network error during code execution."""
    runner = CodeRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('print("test")')
        temp_path = f.name
    
    try:
        # Mock network error
        mock_post.side_effect = Exception("Network timeout")
        
        with pytest.raises(RuntimeError, match="CodeRunner failed:"):
            runner.run(temp_path)
            
    finally:
        os.unlink(temp_path)


def test_run_file_not_found():
    """Test error when code file doesn't exist."""
    runner = CodeRunner()
    
    with pytest.raises(RuntimeError, match="CodeRunner failed:"):
        runner.run("/nonexistent/path/file.py")


@patch('atlasmind.core.tools.code_executor_svc.code_runner.requests.post')
def test_run_with_different_code(mock_post):
    """Test execution with different Python code."""
    runner = CodeRunner()
    
    test_code = '''
def add(a, b):
    return a + b

result = add(5, 3)
print(f"Result: {result}")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_path = f.name
    
    try:
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": {"description": "Accepted"},
            "stdout": "Result: 8\n",
            "stderr": "",
            "exit_code": 0
        }
        mock_post.return_value = mock_response
        
        result = runner.run(temp_path)
        
        # Verify the code was sent correctly
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        # Check that the source code contains the function definition
        assert 'def add(a, b):' in payload['source_code']
        assert 'result = add(5, 3)' in payload['source_code']
        assert 'print(f"Result: {result}")' in payload['source_code']
        
        # Check that the response contains the expected output
        assert result['stdout'] == 'Result: 8\n'
        
    finally:
        os.unlink(temp_path)