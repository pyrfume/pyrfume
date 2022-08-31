#!/usr/bin/env bash

set -eux

name="pyrfume"
docker build --cache-from "$name" --file dev.Dockerfile --tag "$name" .
docker run --interactive --tty --rm --name "$name" --volume "$(pwd):/workspace/$name" "$name"
