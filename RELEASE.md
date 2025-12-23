# Workflow for Releasing to PyPI

1. Ensure `main` is ready to release
2. Make sure `CHANGELOG.md` is up to date (review commit history to verify, open a Release PR if changes are needed)
3. Verify the version in `CHANGELOG.md` matches the version you're about to tag
4. Create and push the release tag ([see here](#valid-version-numbers-and-their-meaning) for version syntax)
```bash
$ git tag vX.Y.Z
$ git push origin vX.Y.Z
```
5. Monitor the GitHub Actions workflow to ensure the build and publish succeed
6. Add a GitHub Release with a small description of changes from the last release.

**Note:** Tags must start with "v" for the publishing Github Action to begin.

## Valid version numbers and their meaning

- For version number we follow [SemVer](https://semver.org/) (major.minor.patch).
- For pre-release tags, we follow the [PEP440](https://peps.python.org/pep-0440/) syntax: 
    - v0.1.3rc1: First release candidate for v0.1.3
    - v0.1.3a3: 3rd alpha version of v0.1.3
    - v0.1.3b4: 4th Beta version of v0.1.3
