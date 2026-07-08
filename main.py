"""Top-level entry point. Delegates to arena.main."""

# Must run before importing arena.main (which imports arena.agents), since
# arena.agents reads CRUCIBLE_PROVIDER / API keys from os.environ at module
# load time. Loading .env any later means those reads see stale defaults.
from dotenv import load_dotenv

load_dotenv()

from arena.main import main  # noqa: E402

if __name__ == "__main__":
    main()
