# ML4RG-group-17

## Getting started

- Create a virtual environment:
  ```bash
  python -m venv venv
  ```
  Notes:
  - On Windows: Use `python` or `py -3` if you have the Python Launcher
  - On macOS/Linux: Use `python3` if `python` points to Python 2
  - Alternative Windows command: `py -m venv venv`

- Activate the virtual environment:
  - On Windows:
    ```bash
    venv\Scripts\activate
    ```
  - On macOS and Linux:
    ```bash
    source venv/bin/activate
    ```

- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Linting and Formatting

- **Black**: Code formatter (run `black app` to format the code)
- **Isort**: Import sorter (run `isort app` to sort imports)
- **Flake8**: Linter (run `flake8` to check for linting issues)

### VSCode

Install the required extensions:
  * [Black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  * [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)
  * [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)

Change the settings in VSCode to look for the config files of Linter and Formatter at the root directory of the project. Then turn on “format on save”, also in the settings.

### PyCharm

Install __black__ & __isort__ and add them to the FileWatchers.
This automatically formats the code according to PEP 8 when saving the files.

## Code Style

Adhere to the [PEP 8](https://peps.python.org/pep-0008/) style guide:

* **Indentation**: 4 spaces per indentation level.  
* **Line Length**: Limit lines to 79 characters.  
* **Imports**: Group and order imports as follows:  
  * Standard library imports.  
  * Related third-party imports.  
  * Local application/library-specific imports

Each group should be separated by a blank line.

* **Naming Conventions**:  
  * Variables, functions, methods: `snake_case`  
  * Classes: `PascalCase`  
  * Constants: `UPPER_CASE`

## Comments & Docstrings

* **Docstrings**: Use triple double quotes (`"""`) for module, class, and function docstrings. Follow the [PEP 257](https://peps.python.org/pep-0257/) conventions.  
* **Inline Comments**:  
  * Use sparingly to explain non-obvious code.  
  * Separate inline comments by at least two spaces from the statement.  
  * Start with a `#` and a single space.  
  * Ensure they are complete sentences and start with a capital letter.

## Version Control & Collaboration

Use feature branches named using the pattern `type/short-description`, e.g., `feat/pytorch-geometric-setup`. The type can be:
  * `feat`: the branch introduces a new feature.
  * `fix`: the branch fixes a bug.
  * `docs`: the branch contains documentation
  * `chore`: the branch contains clean up or config stuff

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:  
  * Format: `<type>(<scope>): <description>`  
  * Example: `feat(auth): add login functionality`  

## Documentation

Reflect any updates to the steps required for setup or any part of it, core parts of the functionality or usability as described, or technologies used in the README.md file. The README.md file should grow as the project does and always provide an up-to-date overview of the project, how to set it up locally, and how to use it.