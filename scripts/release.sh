#!/bin/bash

set -e

# Function to extract version components from version.py using regex
get_version_from_file() {
    VERSION_FILE="myocr/version.py"

    if [[ ! -f "$VERSION_FILE" ]]; then
        echo "Error: $VERSION_FILE does not exist."
        exit 1
    fi

    _MAJOR=$(grep -E '^_MAJOR\s*=\s*"([^"]+)"' "$VERSION_FILE" | sed -nE 's/^_MAJOR[[:space:]]*=[[:space:]]*"([^"]+)"$/\1/p')
    if [[ -z "$_MAJOR" ]]; then
        echo "Error: Could not extract _MAJOR from $VERSION_FILE."
        exit 1
    fi

    _MINOR=$(grep -E '^_MINOR\s*=\s*"([^"]+)"' "$VERSION_FILE" | sed -nE 's/^_MINOR[[:space:]]*=[[:space:]]*"([^"]+)"$/\1/p')
    if [[ -z "$_MINOR" ]]; then
        echo "Error: Could not extract _MINOR from $VERSION_FILE."
        exit 1
    fi

    _PATCH=$(grep -E '^_PATCH\s*=\s*"([^"]+)"' "$VERSION_FILE" | sed -nE 's/^_PATCH[[:space:]]*=[[:space:]]*"([^"]+)"$/\1/p')
    if [[ -z "$_PATCH" ]]; then
        echo "Error: Could not extract _PATCH from $VERSION_FILE."
        exit 1
    fi

    # Extract _SUFFIX (optional)
    _SUFFIX=$(grep -E '^_SUFFIX\s*=\s*"([^"]*)"' "$VERSION_FILE" | sed -nE 's/^_SUFFIX[[:space:]]*=[[:space:]]*"([^"]+)"$/\1/p')
    if [[ -z "$_SUFFIX" ]]; then
        _SUFFIX=""
    fi


    VERSION_PY="${_MAJOR}.${_MINOR}.${_PATCH}${_SUFFIX}"
    echo "$VERSION_PY"
}


TAG=$(python -c 'from myocr.version import VERSION; print("v" + VERSION)')

# Get the VERSION from version.py
VERSION_PY=$(get_version_from_file)

# Compare the two versions
if [[ "v$VERSION_PY" != "$TAG" ]]; then
    echo "Version mismatch detected:"
    echo "  Python reported version: $TAG"
    echo "  version.py contains: v$VERSION_PY"
    echo
    read -p "The versions do not match. Please run 'pip install -e .' to synchronize versions. Do you want to continue? [Y/n] " prompt

    if [[ ! "$prompt" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Release process aborted due to version mismatch."
        exit 1
    else
        echo "Proceeding with the release despite the version mismatch."
    fi
fi

read -p "Creating new release for $TAG. Do you want to continue? [Y/n] " prompt

if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]; then
    python scripts/prepare_changelog.py
    git add -A
    git commit -m "Bump version to $TAG for release" || true && git push
    echo "Creating new git tag $TAG"
    git tag "$TAG" -m "$TAG"
    git push --tags
else
    echo "Cancelled"
    exit 1
fi
