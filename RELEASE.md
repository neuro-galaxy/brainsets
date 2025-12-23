# Workflow for Releasing to PyPI

1. Ensure `main` is ready to release
2. Make sure `CHANGELOG.md` is up to date (review commit history to verify, open a Release PR if changes are needed)
3. Create and push the release tag
```bash
$ git tag vX.Y.Z
$ git push origin vX.Y.Z
```
4. Add a Github Release with a small description of changes from the last release.

## Pre-release versions

If the tag contains pre-release strings, e.g. `v0.1.4-rc.1`, or `v0.1.4-alpha`,
then the build would automatically be published to TestPyPI instead of PyPI.