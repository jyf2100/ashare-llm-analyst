```markdown
# ashare-llm-analyst Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill introduces the core development patterns and workflows used in the `ashare-llm-analyst` Python codebase. You'll learn the project's coding conventions, how to contribute new or improved data providers with robust anti-crawler and retry logic, and how to write and run comprehensive tests. The guide also documents common commands and best practices for maintainable, testable code.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files and modules.
  - Example: `anti_crawler.py`, `providers.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .cache import CacheManager
    from ..data.providers import Provider
    ```

### Export Style
- Use **named exports** (i.e., explicitly define what is exported from a module).
  - Example:
    ```python
    # src/core/anti_crawler.py
    class AntiCrawler:
        ...
    __all__ = ["AntiCrawler"]
    ```

### Commit Messages
- Follow **conventional commit** style.
  - Prefixes: `feat`, `refactor`, `test`
  - Example: `feat: add retry logic to provider manager`

## Workflows

### Feature Development with Hardened Provider and Shared Logic
**Trigger:** When you need to add or harden a data provider with shared anti-crawler/retry logic and ensure robust, offline-testable behavior.  
**Command:** `/harden-provider`

**Step-by-step Instructions:**

1. **Refactor Core Logic**
   - Move or update anti-crawler and retry management logic into a reusable module.
   - Example:
     ```python
     # src/core/anti_crawler.py
     class AntiCrawler:
         def handle(self, response):
             # logic here
     ```

2. **Update Provider Implementation**
   - Refactor or implement the provider to use the shared logic.
   - Add new features as needed (rate limiting, retry, caching, shared login).
   - Example:
     ```python
     # src/data/providers.py
     from ..core.anti_crawler import AntiCrawler

     class MyProvider:
         def fetch(self, url):
             response = self.downloader.get(url)
             return AntiCrawler().handle(response)
     ```

3. **Update or Extend Core Utilities**
   - Modify or add to utility modules as needed (e.g., caching, exceptions).
   - Example:
     ```python
     # src/core/cache.py
     class CacheManager:
         ...
     ```

4. **Write or Update Comprehensive Tests**
   - Ensure tests cover offline scenarios and edge cases.
   - Place tests in `tests/test_providers_harden.py` or `tests/test_providers.py`.
   - Example:
     ```python
     # tests/test_providers_harden.py
     def test_provider_handles_captcha():
         ...
     ```

5. **Register Tests in the Test Runner**
   - Ensure new/updated tests are included in `tests/run_tests.py`.
   - Example:
     ```python
     # tests/run_tests.py
     from .test_providers_harden import *
     ```

**Files Involved:**
- `src/core/anti_crawler.py`
- `src/data/downloaders.py`
- `src/data/providers.py`
- `src/core/cache.py`
- `src/core/exceptions.py`
- `tests/test_providers_harden.py`
- `tests/test_providers.py`
- `tests/run_tests.py`

**Frequency:** ~1-2x/month

## Testing Patterns

- **Framework:** Not explicitly detected; tests are Python scripts in `tests/`.
- **Naming:** Test files use `test_*.py` naming convention.
- **Coverage:** Tests should cover offline behavior and edge cases.
- **Example:**
  ```python
  # tests/test_providers.py
  def test_provider_retry_logic():
      # Arrange
      # Act
      # Assert
  ```

## Commands

| Command           | Purpose                                                                 |
|-------------------|-------------------------------------------------------------------------|
| /harden-provider  | Start the workflow to add or harden a provider with shared core logic.  |

```