# Workflow for Releasing to PyPI and TestPyPI

1. Ensure `main` is ready to release
2. Make sure `CHANGELOG.md` is up to date (review commit history to verify, open a Release PR if changes are needed)
3. Verify the version in `CHANGELOG.md` matches the version you're about to tag
4. Create and push the release tag
```bash
$ git tag vX.Y.Z
$ git push origin vX.Y.Z
```
5. Monitor the GitHub Actions workflow to ensure the build and publish succeed
6. Add a Github Release with a small description of changes from the last release.

## Pre-release versions go to TestPyPI

If the tag contains pre-release strings (`rc`, `a`, `b`, or `dev`), e.g.:
- `v0.1.4-rc.1` or `v0.1.4rc1`
- `v0.1.4-alpha` or `v0.1.4a1`
- `v0.1.4-beta` or `v0.1.4b1`
- `v0.1.4-dev`

then the build will automatically be published to TestPyPI instead of PyPI. 
The GitHub Actions workflow triggers automatically when you push a tag matching `v*`.