# Contributing

This project uses **GitHub Flow**: `main` is always deployable, and all changes reach it through pull requests.

## Workflow

1. Update `main`:
   ```bash
   git checkout main
   git pull
   ```
2. Create a branch from `main` with a short descriptive name:
   ```bash
   git checkout -b feature/<short-description>
   ```
   Use prefixes such as `feature/`, `fix/`, `docs/`, or `chore/`.
3. Make small, focused commits with clear messages.
4. Push the branch and open a pull request against `main`:
   ```bash
   git push -u origin feature/<short-description>
   ```
5. Get the pull request reviewed. Address feedback with new commits on the same branch.
6. Merge the pull request once it is approved. Delete the branch afterwards.

## Rules

- Do not push directly to `main`. Every change goes through a pull request.
- Keep pull requests small and focused on one change.
- Describe what changed and why in the pull request.
