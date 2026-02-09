"""Tests for the lightweight tool system and local file tools."""

import pytest

from cicaddy.tools import (
    Tool,
    ToolRegistry,
    create_local_file_registry,
    glob_files,
    read_file,
    set_working_directory,
    tool,
)


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_basic(self):
        """Test basic tool decoration."""

        @tool
        def my_tool(x: str) -> str:
            """A simple tool."""
            return f"Result: {x}"

        assert isinstance(my_tool, Tool)
        assert my_tool.name == "my_tool"
        assert "simple tool" in my_tool.description

    def test_tool_decorator_with_args(self):
        """Test tool decoration with custom name and description."""

        @tool(name="custom_name", description="Custom description")
        def another_tool(x: str) -> str:
            return x

        assert another_tool.name == "custom_name"
        assert another_tool.description == "Custom description"

    def test_tool_schema_generation(self):
        """Test that input schema is correctly generated."""

        @tool
        def search(query: str, limit: int = 10, exact: bool = False) -> str:
            """Search for items.

            Args:
                query: Search query string
                limit: Maximum results to return
                exact: Whether to match exactly
            """
            return f"Found results for {query}"

        schema = search.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "exact" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["exact"]["type"] == "boolean"
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]  # Has default

    def test_tool_to_mcp_format(self):
        """Test conversion to MCP format."""

        @tool
        def my_tool(x: str) -> str:
            """Tool description."""
            return x

        mcp_format = my_tool.to_mcp_format(server_name="test_server")
        assert mcp_format["name"] == "my_tool"
        assert mcp_format["server"] == "test_server"
        assert "inputSchema" in mcp_format
        assert mcp_format["inputSchema"]["type"] == "object"

    def test_tool_invocation(self):
        """Test tool invocation."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add.invoke({"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_tool_async_invocation(self):
        """Test async tool invocation."""

        @tool
        async def async_add(a: int, b: int) -> int:
            """Add two numbers asynchronously."""
            return a + b

        result = await async_add.ainvoke({"a": 2, "b": 3})
        assert result == 5


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_registry_creation(self):
        """Test registry creation."""
        registry = ToolRegistry(server_name="test")
        assert len(registry) == 0
        assert registry.server_name == "test"

    def test_registry_register_tool(self):
        """Test registering a tool."""

        @tool
        def my_tool(x: str) -> str:
            return x

        registry = ToolRegistry()
        registry.register(my_tool)
        assert "my_tool" in registry
        assert len(registry) == 1

    def test_registry_register_function(self):
        """Test registering a plain function."""
        registry = ToolRegistry()

        def plain_func(x: str) -> str:
            """A plain function."""
            return x

        registry.register(plain_func, name="custom_name")
        assert "custom_name" in registry

    def test_registry_get_tools(self):
        """Test getting tools in MCP format."""

        @tool
        def tool_a(x: str) -> str:
            return x

        @tool
        def tool_b(y: int) -> int:
            return y

        registry = ToolRegistry(server_name="local")
        registry.register(tool_a)
        registry.register(tool_b)

        tools = registry.get_tools()
        assert len(tools) == 2
        # Should be sorted alphabetically
        assert tools[0]["name"] == "tool_a"
        assert tools[1]["name"] == "tool_b"
        assert all(t["server"] == "local" for t in tools)

    @pytest.mark.asyncio
    async def test_registry_call_tool(self):
        """Test calling a tool through the registry."""

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        registry = ToolRegistry()
        registry.register(multiply)

        result = await registry.call_tool("multiply", {"a": 3, "b": 4})
        assert result["status"] == "success"
        assert result["content"] == "12"
        assert result["tool"] == "multiply"

    @pytest.mark.asyncio
    async def test_registry_call_unknown_tool(self):
        """Test calling an unknown tool."""
        registry = ToolRegistry()
        result = await registry.call_tool("unknown", {})
        assert result["status"] == "error"
        assert "not found" in result["content"]


class TestLocalFileTools:
    """Tests for local file tools (glob_files, read_file)."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace with test files."""
        # Create directory structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "subdir").mkdir()
        (tmp_path / "tests").mkdir()

        # Create test files
        (tmp_path / "file1.py").write_text("# Python file 1\nprint('hello')")
        (tmp_path / "file2.py").write_text("# Python file 2\nprint('world')")
        (tmp_path / "src" / "main.py").write_text(
            "# Main module\ndef main():\n    pass"
        )
        (tmp_path / "src" / "subdir" / "nested.py").write_text("# Nested file")
        (tmp_path / "tests" / "test_main.py").write_text("# Test file")
        (tmp_path / "readme.md").write_text("# README\n\nThis is a test.")
        (tmp_path / "config.json").write_text('{"key": "value"}')

        # Set working directory
        set_working_directory(str(tmp_path))
        return tmp_path

    def test_glob_files_basic(self, temp_workspace):
        """Test basic glob pattern matching."""
        result = glob_files("*.py")
        assert "Found 2 file(s)" in result
        assert "file1.py" in result
        assert "file2.py" in result

    def test_glob_files_recursive(self, temp_workspace):
        """Test recursive glob pattern."""
        result = glob_files("**/*.py")
        assert "Found 5 file(s)" in result
        assert "file1.py" in result
        assert "src/main.py" in result or "src\\main.py" in result
        assert "tests/test_main.py" in result or "tests\\test_main.py" in result

    def test_glob_files_with_path(self, temp_workspace):
        """Test glob with subdirectory path."""
        result = glob_files("*.py", path="src")
        assert "Found 1 file(s)" in result
        assert "main.py" in result

    def test_glob_files_no_match(self, temp_workspace):
        """Test glob with no matches."""
        result = glob_files("*.xyz")
        assert "No files found" in result

    def test_glob_files_max_results(self, temp_workspace):
        """Test max_results limit."""
        result = glob_files("**/*.py", max_results=2)
        # Should only return 2 files even though 5 exist
        lines = [
            line for line in result.split("\n") if line and not line.startswith("Found")
        ]
        assert len(lines) == 2

    def test_glob_files_path_traversal(self, temp_workspace):
        """Test that path traversal is blocked."""
        result = glob_files("*.py", path="../..")
        assert "Error" in result
        assert "traversal" in result.lower() or "escapes" in result.lower()

    def test_read_file_basic(self, temp_workspace):
        """Test basic file reading."""
        result = read_file("file1.py")
        assert "# File: file1.py" in result
        assert "Python file 1" in result
        assert "print('hello')" in result

    def test_read_file_with_line_numbers(self, temp_workspace):
        """Test that line numbers are included."""
        result = read_file("src/main.py")
        assert "1\t" in result or "     1\t" in result
        assert "2\t" in result or "     2\t" in result

    def test_read_file_offset_and_limit(self, temp_workspace):
        """Test reading with offset and limit."""
        # Create a file with many lines
        many_lines = "\n".join([f"Line {i}" for i in range(1, 101)])
        (temp_workspace / "many_lines.txt").write_text(many_lines)

        result = read_file("many_lines.txt", offset=10, limit=5)
        assert "lines 10-14" in result
        assert "Line 10" in result
        assert "Line 14" in result
        assert "Line 9" not in result
        assert "Line 15" not in result

    def test_read_file_not_found(self, temp_workspace):
        """Test reading non-existent file."""
        result = read_file("nonexistent.py")
        assert "Error" in result
        assert "not found" in result

    def test_read_file_path_traversal(self, temp_workspace):
        """Test that path traversal is blocked."""
        result = read_file("../../etc/passwd")
        assert "Error" in result
        assert "traversal" in result.lower() or "escapes" in result.lower()

    def test_read_file_empty_file(self, temp_workspace):
        """Test reading an empty file."""
        (temp_workspace / "empty.txt").write_text("")
        result = read_file("empty.txt")
        assert "empty" in result.lower()


class TestLocalFileRegistry:
    """Tests for create_local_file_registry factory."""

    def test_create_registry(self, tmp_path):
        """Test creating a local file registry."""
        registry = create_local_file_registry(str(tmp_path))
        assert len(registry) == 2
        assert "glob_files" in registry
        assert "read_file" in registry

    def test_registry_mcp_format(self, tmp_path):
        """Test that registry produces MCP-compatible tool definitions."""
        registry = create_local_file_registry(str(tmp_path))
        tools = registry.get_tools()

        assert len(tools) == 2
        for tool_def in tools:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "inputSchema" in tool_def
            assert "server" in tool_def
            assert tool_def["server"] == "local"

    @pytest.mark.asyncio
    async def test_registry_execute_glob(self, tmp_path):
        """Test executing glob_files through registry."""
        # Create a test file
        (tmp_path / "test.py").write_text("print('test')")

        registry = create_local_file_registry(str(tmp_path))
        result = await registry.call_tool("glob_files", {"pattern": "*.py"})

        assert result["status"] == "success"
        assert "test.py" in result["content"]

    @pytest.mark.asyncio
    async def test_registry_execute_read(self, tmp_path):
        """Test executing read_file through registry."""
        # Create a test file
        (tmp_path / "hello.txt").write_text("Hello, World!")

        registry = create_local_file_registry(str(tmp_path))
        result = await registry.call_tool("read_file", {"file_path": "hello.txt"})

        assert result["status"] == "success"
        assert "Hello, World!" in result["content"]
