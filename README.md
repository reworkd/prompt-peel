# Prompt-peel 🍌
A Python prompt design library heavily based on Priompt from Cursor/Anysphere. 
Build declarative prompts that automatically select the "optimal" prompt based on priority

## What's wrong with prompt design today?
TODO

## How does priompt/prompt-peel aim to fix it?
TODO

## DSL
Top level
- `system_prompt(*children)`: Self explanatory
- `user_prompt(*children)`: Self explanatory 
- `assistant_prompt(*children)`: Self explanatory
- `scope(*children)`: Create a new scope 
- `top_k(*children, top_k_value=N)`
- `empty(tokens=N)`: Empty cell used to  to define how many tokens you require

# Getting started
## Using the library
```
poetry add prompt-peel
```

## Contributing to the library
```
TODO
git checkout ____
```
- Look to the tests to get the best understanding of library features and practices.
Ensure tests pass before PR-ing
- Before PRs, run linting via `./lint.sh`

# TODO
- [x] Token counting logic
- [x] Binary search for optimal priority
- [x] Empty node to save space for N tokens
- [x] Top K node to only take top k elements from a list
- [ ] Accept function calling
- [ ] Allow images in prompts

# Caveats
- JSX is much more ergonomic than python strings. Automatic node splitting (when you embed elements amonst strings),
automatic spacing on new line, automatic de-tabbing, etc. 
You must actively account for this in python (as seen in the examples)
- The aim is not to have feature parity with Priompt or even to follow their architecture in the long run.
We think they've done a great job and currently provide the functionality we ourselves need, 

## Contributing
Contributions are welcome. Please open an issue or a pull request. Test cases are required.

### Relevant reading
- [Priompt](https://github.com/anysphere/priompt): What this library is based off. 
A good read to understand their foundational principals.
- [Writing DSLs](https://weblog.jamisbuck.org/2006/4/20/writing-domain-specific-languages): 
A short primer for what a DSL is and why you'd want to write one
- [Build your own React](https://pomb.us/build-your-own-react/): 
A good look into how the DSL of React/JSX is implemented and handled
