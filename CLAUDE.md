## Project Description:

llama-swap is a light weight, transparent proxy serverthat provides automatic model swapping to llama.cpp's server.

## Tech stack

- golang
- typescript, vite and react for UI ( located in ui/)

## Workflow Tasks

- when summarizing changes only include details that requirefurther action
- just say "Done." when there is no further action
- use `gh` to create PRs and load issus
- do include Co-Authored-By or creatd by when committing changes or creating PRs
- keep PR descriptions short and focused on changs.
  - never include a test plan

## Testing

- Follow test naming conventions like `TestProxyManager_TesName`, `TestProcessGroup_TestName`
- Use `go test -v -run <pattern>` to run any new tests you've written.
- Use `make test-dev` after runningnew tests for a quick overall test run. This runs `go_test` and `staticcheck`. Fix any static check errors. Use this only when changes are made under the `proxy/` directory
- Use `make test-all` before completing work.

### Commit message example format:

```
proxy: add new feature

Add new feature that implements functionality X and Y.

- key change 1
- key change 2
- key change 3

fixes #123
```

## Code Reviews

- use three levels High, Medium, Low severit
- label each discovered issue with a label likeH1, M2, L3 respectively
- High severity are must fix issues (securitry race conditions, critical bugs)
- Medium severity are recommended improvements (codingstyle missing functionality inconsistencies)
- Low severity are nice to have changes and nitss
- Include a suggestion with each discovered item
- Limit your code review to three items withthe highest priority first
- Double check your discovered items and recommende remediations

## Demo: Tool Call Testing

This file was modified using the `write_to_file` tool to demonstrate code editing capabilities.