"""Test module ``icon_timeseries``."""
# Third-party
from click.testing import CliRunner

# First-party
from icon_timeseries import cli


class _TestCLI:
    """Base class to test the command line interface."""

    def call(self, args=None, *, exit_=0):
        runner = CliRunner()
        result = runner.invoke(cli.main, args)
        assert result.exit_code == exit_
        return result


class TestNoCmd(_TestCLI):
    """Test CLI without commands."""

    def test_default(self):
        result = self.call()
        assert result.output.startswith("Usage: ")
        assert "Show this message and exit." in result.output

    def test_help(self):
        result = self.call(["--help"])
        assert result.output.startswith("Usage: ")
        assert "Show this message and exit." in result.output

    def test_version(self):
        result = self.call(["-V"])
        assert cli.__version__ in result.output


class TestCmd(_TestCLI):
    """Test CLI with some commands."""

    def test_meanmax(self):
        result = self.call(["meanmax", "-h"])
        assert result.output.startswith("Usage: ")
        assert "Show this message and exit." in result.output
